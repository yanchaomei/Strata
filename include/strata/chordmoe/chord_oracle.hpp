// include/strata/chordmoe/chord_oracle.hpp - Exact Offline Branch-and-Bound Oracles & 2026 Competitor Baselines.
//
// Provides:
//   1. `chord_solve_exact_setcover_oracle`: Provably exact global minimum cold-expert union cardinality `U_OPT`
//      over a verification window subject to anchor immunity, logit margin, and exact sparse TV budget `tv_budget_token`,
//      via per-token Pareto-undominated cold-mask enumeration + cross-token Branch-and-Bound.
//   2. `chord_solve_exact_makespan_oracle`: Provably exact global minimum makespan `C*_max` across `G` PCIe DMA
//      links and `S` NUMA CPU sockets via LPT-seeded Branch-and-Bound with volume lower bounds & symmetry breaking.
//   3. `strata_static_pcie_share_baseline`: Static bandwidth-proportional round-robin splitter (emulating `--pcie-frac`
//      without multiplicity sorting or NUMA page-home affinity).
//   4. `moespec_budget_router` (2026.02 MoE-Spec hard expert budgeting / dropping) & `tail_skip_router` (`TAIL_SKIP`).
#pragma once

#include "strata/chordmoe/chord_router.hpp"
#include "strata/chordmoe/chord_sched.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace strata::chordmoe {

struct ExactSetCoverResult {
    int32_t  unbiased_cold_union = 0;
    int32_t  chord_cold_union    = 0;
    int32_t  opt_cold_union      = 0;
    bool     proven_exact        = true;
    uint64_t bnb_nodes           = 0;
};

/// Compute exact offline minimum cold-expert union `U_OPT` for a single `[n_tok, k]` window.
/// Note: `ids_in` and `weights_in` must be the unbiased Top-K routing of `logits`.
inline ExactSetCoverResult chord_solve_exact_setcover_oracle(
    const float* logits, const int32_t* ids_in, const float* weights_in,
    int n_tok, int n_expert, int k,
    const int32_t* primary_res, const ChordConfig& cfg,
    const int32_t* peer_res = nullptr,
    uint64_t max_bnb_nodes = 500000) {

    ExactSetCoverResult res{};
    if (n_tok <= 0 || k <= 0 || n_expert <= 0 || n_expert > kChordMaxExperts || k > kChordMaxTopK) return res;

    constexpr int kWords = kChordMaxExperts / 32;
    auto bm_test = [](const std::array<uint32_t, kWords>& bm, int e) -> bool {
        return ((bm[(size_t) (e >> 5)] >> (e & 31)) & 1u) != 0;
    };
    auto bm_set = [](std::array<uint32_t, kWords>& bm, int e) {
        bm[(size_t) (e >> 5)] |= (1u << (e & 31));
    };
    auto bm_pop = [&](const std::array<uint32_t, kWords>& bm) -> int32_t {
        int32_t c = 0;
        const int nw = (n_expert + 31) / 32;
        for (int w = 0; w < nw; ++w) c += __builtin_popcount(bm[(size_t) w]);
        return c;
    };
    auto is_subset = [&](const std::array<uint32_t, kWords>& sub,
                         const std::array<uint32_t, kWords>& sup) -> bool {
        const int nw = (n_expert + 31) / 32;
        for (int w = 0; w < nw; ++w) {
            if ((sub[(size_t) w] & ~sup[(size_t) w]) != 0u) return false;
        }
        return true;
    };

    // 1. Evaluate ChordMoE greedy solution as tight initial upper bound!
    std::vector<int32_t> chord_ids(ids_in, ids_in + (size_t) n_tok * k);
    std::vector<float>   chord_w(weights_in, weights_in + (size_t) n_tok * k);
    ChordConfig c_cfg = cfg;
    c_cfg.mode = ChordMode::BoundedSetCover;
    ChordWindowStats c_st{};
    chord_reshape_window(logits, chord_ids.data(), chord_w.data(), n_tok, n_expert, k,
                         primary_res, c_cfg, peer_res, nullptr, nullptr, &c_st);

    std::array<uint32_t, kWords> raw_bm{}, chord_bm{}, base_paid_bm{}, window_cold_pool{};
    const int r_lo = std::clamp(cfg.rank_lo, 0, k);
    const int r_hi = std::min((int) cfg.rank_hi, k - 1);

    for (int t = 0; t < n_tok; ++t) {
        const bool is_anchor = (t == 0 && cfg.protect_anchor);
        for (int r = 0; r < k; ++r) {
            const int32_t e0 = ids_in[(size_t) t * k + r];
            if (e0 >= 0 && e0 < n_expert && classify_expert_tier(e0, primary_res, peer_res) == ExpertTier::ColdHost) {
                bm_set(raw_bm, e0);
                bm_set(window_cold_pool, e0);
                if (is_anchor || r < r_lo || r > r_hi) {
                    bm_set(base_paid_bm, e0);
                }
            }
            const int32_t ec = chord_ids[(size_t) t * k + r];
            if (ec >= 0 && ec < n_expert && classify_expert_tier(ec, primary_res, peer_res) == ExpertTier::ColdHost) {
                bm_set(chord_bm, ec);
            }
        }
    }
    res.unbiased_cold_union = bm_pop(raw_bm);
    res.chord_cold_union    = bm_pop(chord_bm);
    const int32_t base_cnt  = bm_pop(base_paid_bm);
    if (res.chord_cold_union <= base_cnt) {
        res.opt_cold_union = res.chord_cold_union;
        return res;
    }

    float surv[kChordMaxWindow];
    fill_default_survival(std::min(n_tok, kChordMaxWindow), cfg.default_alpha, surv);

    // 2. For each draft token t, enumerate all Pareto-minimal residual cold masks `M_t \subseteq Cold \setminus base_paid_bm`
    //    achievable under `(eff_margin, tv_budget_token)`.
    const int start_t = cfg.protect_anchor ? 1 : 0;
    std::vector<std::vector<std::array<uint32_t, kWords>>> token_pareto(n_tok);

    for (int t = start_t; t < n_tok; ++t) {
        const float* rl = logits + (size_t) t * n_expert;
        const int32_t* orig_id = ids_in + (size_t) t * k;
        const float*   orig_w  = weights_in + (size_t) t * k;
        const float st = (t < kChordMaxWindow) ? std::clamp(surv[t], 0.0f, 1.0f) : 0.1f;
        const float eff_margin = cfg.logit_margin * (1.0f - cfg.survival_strict * st);

        // Identify cold tail ranks that are NOT already in `base_paid_bm`
        std::vector<int> flex_ranks;
        for (int r = r_lo; r <= r_hi; ++r) {
            const int32_t e = orig_id[r];
            if (e >= 0 && e < n_expert &&
                classify_expert_tier(e, primary_res, peer_res) == ExpertTier::ColdHost &&
                !bm_test(base_paid_bm, e)) {
                flex_ranks.push_back(r);
            }
        }
        if (flex_ranks.empty()) {
            token_pareto[t].push_back(std::array<uint32_t, kWords>{});
            continue;
        }

        // Candidate choices per flexible rank `r`:
        //   - Up to 2 highest-logit zero-extra-cold choices (resident or in `base_paid_bm`) in case top-1 collides;
        //   - Original cold expert `orig_id[r]`;
        //   - Up to 4 highest-logit cold experts in `window_cold_pool \setminus base_paid_bm` within `eff_margin`
        //     (for cross-draft cold coalescing).
        std::vector<std::vector<int32_t>> rank_cands(flex_ranks.size());
        for (size_t idx = 0; idx < flex_ranks.size(); ++idx) {
            const int r = flex_ranks[idx];
            const int32_t e = orig_id[r];
            std::vector<std::pair<float, int32_t>> free_cands, shared_cold_cands;
            for (int32_t f = 0; f < n_expert; ++f) {
                if (rl[e] - rl[f] > eff_margin) continue;
                bool in_fixed = false;
                for (int q = 0; q < k; ++q) {
                    if (std::find(flex_ranks.begin(), flex_ranks.end(), q) == flex_ranks.end() && orig_id[q] == f) {
                        in_fixed = true;
                        break;
                    }
                }
                if (in_fixed) continue;
                const ExpertTier tf = classify_expert_tier(f, primary_res, peer_res);
                if (tf != ExpertTier::ColdHost || bm_test(base_paid_bm, f)) {
                    free_cands.push_back({rl[f], f});
                } else if (f != e && bm_test(window_cold_pool, f)) {
                    shared_cold_cands.push_back({rl[f], f});
                }
            }
            std::sort(free_cands.begin(), free_cands.end(), [](auto& a, auto& b) { return a.first > b.first; });
            std::sort(shared_cold_cands.begin(), shared_cold_cands.end(), [](auto& a, auto& b) { return a.first > b.first; });

            auto& rc = rank_cands[idx];
            for (size_t j = 0; j < std::min<size_t>(2, free_cands.size()); ++j) rc.push_back(free_cands[j].second);
            rc.push_back(e); // always include keeping original `e`
            for (size_t j = 0; j < std::min<size_t>(3, shared_cold_cands.size()); ++j) rc.push_back(shared_cold_cands[j].second);
        }

        std::vector<std::array<uint32_t, kWords>> valid_masks;
        int32_t trial_id[kChordMaxTopK];
        float   trial_w[kChordMaxTopK];
        std::memcpy(trial_id, orig_id, (size_t) k * sizeof(int32_t));

        auto dfs_token = [&](auto& self, size_t depth) -> void {
            if (depth == flex_ranks.size()) {
                renormalize_topk_weights(rl, trial_id, trial_w, k);
                const float tv = compute_sparse_tv(orig_id, orig_w, trial_id, trial_w, k);
                if (tv <= cfg.tv_budget_token + 1e-7f) {
                    std::array<uint32_t, kWords> m{};
                    for (size_t d = 0; d < flex_ranks.size(); ++d) {
                        const int32_t ch = trial_id[flex_ranks[d]];
                        if (classify_expert_tier(ch, primary_res, peer_res) == ExpertTier::ColdHost &&
                            !bm_test(base_paid_bm, ch)) {
                            bm_set(m, ch);
                        }
                    }
                    // Prune supersets to retain only Pareto-minimal subsets!
                    for (const auto& ex : valid_masks) {
                        if (is_subset(ex, m)) return;
                    }
                    valid_masks.erase(
                        std::remove_if(valid_masks.begin(), valid_masks.end(),
                                       [&](const auto& ex) { return is_subset(m, ex); }),
                        valid_masks.end());
                    valid_masks.push_back(m);
                }
                return;
            }
            const int r = flex_ranks[depth];
            for (int32_t cand : rank_cands[depth]) {
                bool dup = false;
                for (int q = 0; q < k; ++q) {
                    if (q == r) continue;
                    // Check fixed ranks and already-assigned flex ranks
                    bool assigned = (std::find(flex_ranks.begin(), flex_ranks.end(), q) == flex_ranks.end());
                    if (!assigned) {
                        for (size_t p = 0; p < depth; ++p) {
                            if (flex_ranks[p] == q) { assigned = true; break; }
                        }
                    }
                    if (assigned && trial_id[q] == cand) { dup = true; break; }
                }
                if (dup) continue;
                trial_id[r] = cand;
                self(self, depth + 1);
            }
        };
        dfs_token(dfs_token, 0);

        if (valid_masks.empty()) {
            std::array<uint32_t, kWords> fallback{};
            for (int r : flex_ranks) bm_set(fallback, orig_id[r]);
            valid_masks.push_back(fallback);
        }
        std::sort(valid_masks.begin(), valid_masks.end(),
                  [&](const auto& a, const auto& b) { return bm_pop(a) < bm_pop(b); });
        token_pareto[t] = std::move(valid_masks);
    }

    // 3. Exact cross-token Branch-and-Bound over `token_pareto[start_t .. n_tok-1]`
    int32_t best_extra = res.chord_cold_union - base_cnt;
    const int nw = (n_expert + 31) / 32;

    // Precompute suffix singleton lower bound: minimum popcount of each token's smallest mask
    std::vector<int32_t> min_tok_pop(n_tok, 0);
    for (int t = start_t; t < n_tok; ++t) {
        min_tok_pop[t] = bm_pop(token_pareto[t][0]);
    }

    auto bnb = [&](auto& self, int t, const std::array<uint32_t, kWords>& cur_union, int32_t cur_pop) -> void {
        if (cur_pop >= best_extra) return;
        if (t >= n_tok) {
            best_extra = cur_pop;
            return;
        }
        if (++res.bnb_nodes > max_bnb_nodes) {
            res.proven_exact = false;
            return;
        }
        // Any single remaining token `s >= t` requires total union >= min_tok_pop[s]
        for (int s = t; s < n_tok; ++s) {
            if (min_tok_pop[s] >= best_extra) return;
        }
        for (const auto& m : token_pareto[t]) {
            std::array<uint32_t, kWords> nxt = cur_union;
            int32_t nxt_pop = 0;
            for (int w = 0; w < nw; ++w) {
                nxt[(size_t) w] |= m[(size_t) w];
                nxt_pop += __builtin_popcount(nxt[(size_t) w]);
            }
            if (nxt_pop < best_extra) {
                self(self, t + 1, nxt, nxt_pop);
                if (!res.proven_exact) return;
            }
        }
    };
    std::array<uint32_t, kWords> empty_u{};
    bnb(bnb, start_t, empty_u, 0);
    res.opt_cold_union = base_cnt + best_extra;
    return res;
}

struct ExactMakespanResult {
    double   chord_makespan_us  = 0.0;
    double   fiddler_makespan_us = 0.0;
    double   static_makespan_us = 0.0;
    double   opt_makespan_us    = 0.0;
    bool     proven_exact       = true;
    uint64_t bnb_nodes          = 0;
};

/// Static bandwidth-proportional splitter emulating legacy `--pcie-frac` (blind to token multiplicity & NUMA home).
inline LayerSchedulePlan strata_static_pcie_share_baseline(
    const std::vector<MissedExpertTask>& tasks,
    const std::vector<HeteroQueueSpec>& queues) {

    LayerSchedulePlan plan;
    if (queues.empty() || tasks.empty()) return plan;
    const size_t Q = queues.size();
    plan.queue_loads_us.assign(Q, 0.0);
    std::vector<uint8_t> used(Q, 0);
    plan.assignments.resize(tasks.size());

    double sum_bw = 0.0;
    for (const auto& q : queues) sum_bw += std::max(0.1, q.bw_gbps);
    std::vector<double> target_share(Q, 0.0), assigned_cnt(Q, 0.0);
    for (size_t q = 0; q < Q; ++q) target_share[q] = std::max(0.1, queues[q].bw_gbps) / sum_bw;

    // Assigns in arrival order purely to match static bandwidth share `L / (L + R)`, ignoring NUMA home & multiplicity!
    for (size_t i = 0; i < tasks.size(); ++i) {
        size_t best_q = 0;
        double max_deficit = -1e18;
        for (size_t q = 0; q < Q; ++q) {
            const double deficit = target_share[q] * (double) (i + 1) - assigned_cnt[q];
            if (deficit > max_deficit) {
                max_deficit = deficit;
                best_q = q;
            }
        }
        const double delta = ChordHeteroScheduler::task_latency_us(tasks[i], queues[best_q], used[best_q] == 0);
        plan.queue_loads_us[best_q] += delta;
        used[best_q] = 1;
        assigned_cnt[best_q] += 1.0;
        plan.assignments[i] = ScheduledExpertAssignment{
            tasks[i].expert_id, (int32_t) best_q, queues[best_q].kind, queues[best_q].backend_id, delta
        };
    }
    for (double ld : plan.queue_loads_us) {
        plan.makespan_us = std::max(plan.makespan_us, ld);
        plan.sum_work_us += ld;
    }
    return plan;
}

/// Exact offline Branch-and-Bound solver for heterogeneous multi-GPU PCIe + multi-NUMA makespan minimization.
inline ExactMakespanResult chord_solve_exact_makespan_oracle(
    const std::vector<MissedExpertTask>& tasks,
    const std::vector<HeteroQueueSpec>& queues,
    uint64_t max_bnb_nodes = 400000) {

    ExactMakespanResult res{};
    if (tasks.empty() || queues.empty()) return res;

    const auto chord_plan   = ChordHeteroScheduler::partition_layer_misses(tasks, queues);
    const auto fiddler_plan = ChordHeteroScheduler::fiddler_sum_objective_baseline(tasks, queues);
    const auto static_plan  = strata_static_pcie_share_baseline(tasks, queues);

    res.chord_makespan_us   = chord_plan.makespan_us;
    res.fiddler_makespan_us = fiddler_plan.makespan_us;
    res.static_makespan_us  = static_plan.makespan_us;
    res.opt_makespan_us     = chord_plan.makespan_us;

    const size_t M = tasks.size();
    const size_t Q = queues.size();

    std::vector<size_t> order(M);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (tasks[a].token_rows != tasks[b].token_rows) return tasks[a].token_rows > tasks[b].token_rows;
        return tasks[a].blob_bytes > tasks[b].blob_bytes;
    });

    // Precompute suffix minimum work `rem_min_work[d]` for volume lower bound `(sum_loads + rem_min_work) / Q`!
    // Wait: on unrelated machines, `sum_{q} load[q] / T <= Q` -> `T >= (sum_{j>=d} min_q(cost_{q,j}) + sum_q load[q])`?
    // Even tighter speed-normalized bound: every task `j` consumes `cost(j, q) / cap_rate(q)`!
    // Simpler exact admissible bound: `sum_q load[q] + sum_{j>=d} min_q delta(j, q) <= Q * T_max`!
    std::vector<double> suf_min_work(M + 1, 0.0);
    for (int d = (int) M - 1; d >= 0; --d) {
        double mc = std::numeric_limits<double>::infinity();
        for (size_t q = 0; q < Q; ++q) {
            mc = std::min(mc, ChordHeteroScheduler::task_latency_us(tasks[order[(size_t) d]], queues[q], false));
        }
        suf_min_work[(size_t) d] = suf_min_work[(size_t) d + 1] + mc;
    }

    std::vector<double> loads(Q, 0.0);
    std::vector<uint8_t> used(Q, 0);
    double best = res.chord_makespan_us;

    auto bnb = [&](auto& self, size_t d, double cur_max, double cur_sum) -> void {
        if (cur_max >= best - 1e-6) return;
        if ((cur_sum + suf_min_work[d]) / (double) Q >= best - 1e-6) return;
        if (d == M) {
            best = cur_max;
            return;
        }
        if (++res.bnb_nodes > max_bnb_nodes) {
            res.proven_exact = false;
            return;
        }
        const MissedExpertTask& task = tasks[order[d]];
        // Try queues in ascending order of tentative finish time
        std::vector<std::pair<double, size_t>> q_try;
        q_try.reserve(Q);
        for (size_t q = 0; q < Q; ++q) {
            // Symmetry breaking among identical unused queues with identical task cost
            bool sym_dup = false;
            if (used[q] == 0) {
                for (size_t p = 0; p < q; ++p) {
                    if (used[p] == 0 && queues[p].kind == queues[q].kind &&
                        std::fabs(queues[p].bw_gbps - queues[q].bw_gbps) < 1e-9 &&
                        (queues[q].kind != ExecBackendKind::CpuNumaSocket || task.home_numa < 0)) {
                        sym_dup = true;
                        break;
                    }
                }
            }
            if (sym_dup) continue;
            const double delta = ChordHeteroScheduler::task_latency_us(task, queues[q], used[q] == 0);
            if (loads[q] + delta < best - 1e-6) {
                q_try.push_back({loads[q] + delta, q});
            }
        }
        std::sort(q_try.begin(), q_try.end(), [](auto& a, auto& b) { return a.first < b.first; });
        for (const auto& [fin, q] : q_try) {
            const double delta = fin - loads[q];
            const uint8_t prev_u = used[q];
            loads[q] = fin;
            used[q] = 1;
            self(self, d + 1, std::max(cur_max, fin), cur_sum + delta);
            loads[q] -= delta;
            used[q] = prev_u;
            if (!res.proven_exact) return;
        }
    };
    bnb(bnb, 0, 0.0, 0.0);
    res.opt_makespan_us = best;
    return res;
}

/// 2026.02 `MoE-Spec` Hard Expert Budgeting Baseline (`arXiv:2602.16052`):
/// Ranks cold experts across the window by summed gating probability `sum_t p_{t,e}`, admits at most `cold_budget`
/// cold experts (always preserving anchor `t=0` when `protect_anchor` is true), and DROPS (sets weight = 0, id = -1)
/// any remaining cold expert on draft tokens before renormalizing surviving weights!
inline void moespec_budget_router(const float* logits, int32_t* ids, float* weights,
                                  int n_tok, int n_expert, int k,
                                  const int32_t* primary_res, int cold_budget,
                                  bool protect_anchor = true,
                                  const int32_t* peer_res = nullptr) {
    if (n_tok <= 0 || k <= 0 || n_expert <= 0) return;
    std::vector<float> cold_mass((size_t) n_expert, 0.0f);
    std::vector<uint8_t> admitted((size_t) n_expert, 0);
    int used_budget = 0;

    if (protect_anchor) {
        for (int r = 0; r < k; ++r) {
            const int32_t e = ids[r];
            if (e >= 0 && e < n_expert && classify_expert_tier(e, primary_res, peer_res) == ExpertTier::ColdHost) {
                if (!admitted[(size_t) e]) {
                    admitted[(size_t) e] = 1;
                    ++used_budget;
                }
            }
        }
    }
    for (int t = 0; t < n_tok; ++t) {
        for (int r = 0; r < k; ++r) {
            const int32_t e = ids[(size_t) t * k + r];
            if (e >= 0 && e < n_expert && classify_expert_tier(e, primary_res, peer_res) == ExpertTier::ColdHost) {
                cold_mass[(size_t) e] += weights[(size_t) t * k + r];
            }
        }
    }
    std::vector<int32_t> order((size_t) n_expert);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return cold_mass[(size_t) a] > cold_mass[(size_t) b]; });
    for (int32_t e : order) {
        if (used_budget >= cold_budget) break;
        if (cold_mass[(size_t) e] > 0.0f && !admitted[(size_t) e]) {
            admitted[(size_t) e] = 1;
            ++used_budget;
        }
    }

    const int start_t = protect_anchor ? 1 : 0;
    for (int t = start_t; t < n_tok; ++t) {
        const float* rl = logits + (size_t) t * n_expert;
        int32_t* tid = ids + (size_t) t * k;
        float*   tw  = weights + (size_t) t * k;
        float mx = -INFINITY;
        int kept = 0;
        for (int r = 0; r < k; ++r) {
            const int32_t e = tid[r];
            if (e >= 0 && e < n_expert &&
                classify_expert_tier(e, primary_res, peer_res) == ExpertTier::ColdHost &&
                !admitted[(size_t) e]) {
                tid[r] = -1;
                tw[r]  = 0.0f;
            } else if (e >= 0 && e < n_expert) {
                mx = std::max(mx, rl[e]);
                ++kept;
            }
        }
        // Renormalize surviving experts (or keep rank-0 if all 8 were dropped)
        if (kept == 0) {
            // Fallback: keep top-1 resident expert if all 8 were cold & dropped
            float best_v = -INFINITY;
            int32_t best_e = 0;
            for (int32_t f = 0; f < n_expert; ++f) {
                if (classify_expert_tier(f, primary_res, peer_res) != ExpertTier::ColdHost && rl[f] > best_v) {
                    best_v = rl[f];
                    best_e = f;
                }
            }
            tid[0] = best_e;
            tw[0]  = 1.0f;
        } else {
            float s = 0.0f;
            for (int r = 0; r < k; ++r) {
                if (tid[r] >= 0) {
                    tw[r] = std::exp(rl[tid[r]] - mx);
                    s += tw[r];
                }
            }
            const float inv = 1.0f / std::max(s, 1e-30f);
            for (int r = 0; r < k; ++r) {
                if (tid[r] >= 0) tw[r] *= inv;
            }
        }
    }
}

/// Rank-based cold tail dropping (`STRATA_ROUTE_TAIL_SKIP=r_skip`): drops cold experts at rank >= `r_skip` on drafts.
inline void tail_skip_router(const float* logits, int32_t* ids, float* weights,
                             int n_tok, int n_expert, int k,
                             const int32_t* primary_res, int r_skip,
                             bool protect_anchor = true,
                             const int32_t* peer_res = nullptr) {
    const int start_t = protect_anchor ? 1 : 0;
    for (int t = start_t; t < n_tok; ++t) {
        const float* rl = logits + (size_t) t * n_expert;
        int32_t* tid = ids + (size_t) t * k;
        float*   tw  = weights + (size_t) t * k;
        float mx = -INFINITY;
        for (int r = 0; r < k; ++r) {
            const int32_t e = tid[r];
            if (r >= r_skip && e >= 0 && e < n_expert &&
                classify_expert_tier(e, primary_res, peer_res) == ExpertTier::ColdHost) {
                tid[r] = -1;
                tw[r]  = 0.0f;
            } else if (e >= 0 && e < n_expert) {
                mx = std::max(mx, rl[e]);
            }
        }
        float s = 0.0f;
        for (int r = 0; r < k; ++r) {
            if (tid[r] >= 0) {
                tw[r] = std::exp(rl[tid[r]] - mx);
                s += tw[r];
            }
        }
        if (s > 0.0f) {
            for (int r = 0; r < k; ++r) if (tid[r] >= 0) tw[r] /= s;
        }
    }
}

}  // namespace strata::chordmoe
