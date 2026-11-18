// include/strata/chordmoe/chord_router.hpp - Residency-Aware Window Union Set-Cover Router & Survival Prefetcher.
//
// Addresses four concrete failure modes verified in 2025-2026 baselines:
//   1. Residency blindness (S2-MoE): S2-MoE picks a window reuse set solely from router frequencies without
//      consulting VRAM residency, routing ~49% of "reused" slots to cold offloaded experts.
//   2. Anchor corruption (S2-MoE 99.3%, MoE-Infinity 97.4%, STRATA_ROUTE_RESIDENT): biasing row t = 0 distorts
//      the already-accepted anchor token's target distribution. ChordMoE locks row t = 0 bit-for-bit.
//   3. Per-token independence & wasted swaps (STRATA_ROUTE_RESIDENT): swapping a cold tail expert `e` on draft t > 0
//      when anchor t = 0 ALREADY requires loading `e` saves zero DDR5/PCIe bytes while degrading accuracy.
//      ChordMoE builds a shared window-union bitmap (`paid_bitmap`) in 3 waves:
//        - Wave 0: Lock anchor t = 0 and non-swappable head ranks 0..rank_lo-1; mark all their cold experts paid.
//        - Wave 1: High-survival prefix drafts (t = 1..2) keep any expert in (PrimaryVram U PeerVram U paid_bitmap)
//                  at zero distortion, swap genuinely unpaid cold tail experts within `eff_margin` subject to the
//                  exact sparse Total Variation gate, and publish any surviving cold experts into `paid_bitmap`.
//        - Wave 2: Tail drafts (t >= 3) reshape against the expanded (PrimaryVram U PeerVram U paid_bitmap),
//                  coalescing remaining tail misses onto already-paid experts.
//   4. Long-horizon drift (#4402): `ChordCreditLedger` enforces both a per-(layer, token) sparse TV cap and an
//      exponentially decayed multi-layer turn budget, guaranteeing ||y - y~||_2 <= 2 * B_max * tv_budget.
#pragma once

#include "strata/chordmoe/chord_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::chordmoe {

/// Classify expert `e` across primary VRAM (`primary_res[e] >= 0`) and optional helper/peer VRAM (`peer_res[e] >= 0`).
inline ExpertTier classify_expert_tier(int32_t e, const int32_t* primary_res, const int32_t* peer_res = nullptr) {
    if (primary_res != nullptr && primary_res[e] >= 0) return ExpertTier::PrimaryVram;
    if (peer_res != nullptr && peer_res[e] >= 0) return ExpertTier::PeerVram;
    return ExpertTier::ColdHost;
}

/// Exact sparse Total Variation distance between two Top-K routing distributions `(ids_a, w_a)` and `(ids_b, w_b)`:
///   TV(P, Q) = 0.5 * sum_{e in supp(P) U supp(Q)} |P(e) - Q(e)|  in [0, 1].
/// Unlike hard-truncated KL, sparse TV is finite when supports differ (`e -> f`), costs O(K^2) with K <= 10,
/// and deterministically bounds the expert-layer L2 shift: ||sum P(e) E_e(x) - sum Q(e) E_e(x)||_2 <= 2 * M * TV(P, Q).
inline float compute_sparse_tv(const int32_t* ids_a, const float* w_a,
                               const int32_t* ids_b, const float* w_b, int k) {
    float l1 = 0.0f;
    for (int i = 0; i < k; ++i) {
        const int32_t ea = ids_a[i];
        float wb = 0.0f;
        for (int j = 0; j < k; ++j) {
            if (ids_b[j] == ea) { wb = w_b[j]; break; }
        }
        l1 += std::fabs(w_a[i] - wb);
    }
    for (int j = 0; j < k; ++j) {
        const int32_t eb = ids_b[j];
        bool in_a = false;
        for (int i = 0; i < k; ++i) {
            if (ids_a[i] == eb) { in_a = true; break; }
        }
        if (!in_a) l1 += std::fabs(w_b[j]);
    }
    return 0.5f * l1;
}

/// Renormalize Top-K softmax weights from raw router logits `row_logits` over selected `sel_ids[0..k)`.
/// Matches Strata native_router's clamp (`6.103515625e-5f` = 2^-14) when `clamp_2m14` is true.
inline void renormalize_topk_weights(const float* row_logits, const int32_t* sel_ids, float* out_w,
                                     int k, bool clamp_2m14 = false) {
    float mx = -INFINITY;
    for (int r = 0; r < k; ++r) mx = std::max(mx, row_logits[sel_ids[r]]);
    float sum = 0.0f;
    for (int r = 0; r < k; ++r) {
        out_w[r] = std::exp(row_logits[sel_ids[r]] - mx);
        sum += out_w[r];
    }
    const float denom = clamp_2m14 ? std::max(sum, 6.103515625e-5f) : std::max(sum, 1e-30f);
    const float inv = 1.0f / denom;
    for (int r = 0; r < k; ++r) out_w[r] *= inv;
}

/// Sliding-window / multi-layer error credit ledger per window token.
/// Tracks `accum[t] <- residual_decay * accum[t] + tv_step`, refusing further approximations once `tv_budget_turn` is met.
class ChordCreditLedger {
public:
    explicit ChordCreditLedger(int max_tokens = kChordMaxWindow) : accum_((size_t) max_tokens, 0.0f) {}

    void reset() { std::fill(accum_.begin(), accum_.end(), 0.0f); }

    void step_layer_decay(float gamma_res) {
        for (float& v : accum_) v *= gamma_res;
    }

    bool can_spend(int t, float candidate_delta_tv, const ChordConfig& cfg) const {
        if (candidate_delta_tv > cfg.tv_budget_token + 1e-7f) return false;
        if (t >= 0 && (size_t) t < accum_.size()) {
            if (accum_[(size_t) t] + candidate_delta_tv > cfg.tv_budget_turn + 1e-7f) return false;
        }
        return true;
    }

    void commit_spend(int t, float delta_tv) {
        if (t >= 0 && (size_t) t < accum_.size()) accum_[(size_t) t] += delta_tv;
    }

    float accumulated(int t) const {
        return (t >= 0 && (size_t) t < accum_.size()) ? accum_[(size_t) t] : 0.0f;
    }

private:
    std::vector<float> accum_;
};

/// Populate default survival probabilities `p_alive[0..n_tok)` when explicit drafter acceptance probabilities are absent.
/// `p_alive[0] = 1.0f` (accepted anchor); `p_alive[t] = alpha^t` for chain drafts `t >= 1`.
inline void fill_default_survival(int n_tok, float alpha, float* p_alive) {
    if (n_tok <= 0) return;
    p_alive[0] = 1.0f;
    float cur = 1.0f;
    for (int t = 1; t < n_tok; ++t) {
        cur *= alpha;
        p_alive[t] = cur;
    }
}

/// Reshape one layer's routing `[n_tok, k]` across a speculative verify window under Mode 1 (`BoundedSetCover`).
///
/// Inputs:
///   - `logits`: `[n_tok, n_expert]` raw F32 router logits.
///   - `ids`:    `[n_tok, k]` in/out selected Top-K expert indices (sorted descending by logit from Top-K router).
///   - `weights`:`[n_tok, k]` in/out renormalized Top-K expert weights.
///   - `primary_res`: `[n_expert]` slot >= 0 if resident in GPU0 VRAM, < 0 otherwise.
///   - `peer_res`:    optional `[n_expert]` slot >= 0 if resident on any helper GPU, nullptr if single GPU.
///   - `p_alive`:     optional `[n_tok]` survival probabilities in [0, 1]; nullptr uses `cfg.default_alpha^t`.
///   - `ledger`:      optional multi-layer `ChordCreditLedger` enforcing cross-layer TV budget.
///   - `stats`:       optional telemetry accumulator.
inline void chord_reshape_window(const float* logits, int32_t* ids, float* weights,
                                 int n_tok, int n_expert, int k,
                                 const int32_t* primary_res,
                                 const ChordConfig& cfg,
                                 const int32_t* peer_res = nullptr,
                                 const float* p_alive = nullptr,
                                 ChordCreditLedger* ledger = nullptr,
                                 ChordWindowStats* stats = nullptr) {
    if (n_tok <= 0 || k <= 0 || n_expert <= 0 || n_expert > kChordMaxExperts || k > kChordMaxTopK) return;
    if (cfg.mode != ChordMode::BoundedSetCover) return;

    constexpr int kWords = kChordMaxExperts / 32;
    uint32_t paid_cold_bm[kWords] = {};
    uint32_t before_cold_bm[kWords] = {};
    uint32_t after_cold_bm[kWords] = {};

    auto bm_test = [](const uint32_t* bm, int e) -> bool {
        return ((bm[e >> 5] >> (e & 31)) & 1u) != 0;
    };
    auto bm_set = [](uint32_t* bm, int e) {
        bm[e >> 5] |= (1u << (e & 31));
    };
    auto bm_popcount = [&](const uint32_t* bm) -> uint64_t {
        uint64_t c = 0;
        const int nw = (n_expert + 31) / 32;
        for (int w = 0; w < nw; ++w) c += (uint64_t) __builtin_popcount(bm[w]);
        return c;
    };

    float local_surv[kChordMaxWindow];
    const int capped_tok = std::min(n_tok, kChordMaxWindow);
    if (p_alive == nullptr) {
        fill_default_survival(capped_tok, cfg.default_alpha, local_surv);
        p_alive = local_surv;
    }

    if (ledger != nullptr) ledger->step_layer_decay(cfg.residual_decay);

    const int r_lo = std::clamp(cfg.rank_lo, 0, k);
    const int r_hi = std::min((int) cfg.rank_hi, k - 1);

    // Census before reshaping + Wave 0: lock anchor t = 0 and non-swappable head ranks 0..r_lo-1.
    uint64_t cold_entries_before = 0;
    for (int t = 0; t < n_tok; ++t) {
        const bool is_anchor = (t == 0 && cfg.protect_anchor);
        for (int r = 0; r < k; ++r) {
            const int32_t e = ids[(size_t) t * k + r];
            if (e < 0 || e >= n_expert) continue;
            const ExpertTier tier = classify_expert_tier(e, primary_res, peer_res);
            if (tier == ExpertTier::ColdHost) {
                bm_set(before_cold_bm, e);
                ++cold_entries_before;
                if (is_anchor || r < r_lo || r > r_hi) {
                    bm_set(paid_cold_bm, e);
                }
            }
        }
    }

    // Helper to reshape token range [t_begin, t_end) against current (PrimaryVram U PeerVram U paid_cold_bm),
    // then publish any remaining cold experts into `paid_cold_bm` for later waves.
    auto process_wave = [&](int t_begin, int t_end) {
        for (int t = t_begin; t < t_end && t < n_tok; ++t) {
            if (t == 0 && cfg.protect_anchor) continue;

            const float* row_l = logits + (size_t) t * n_expert;
            int32_t* row_ids = ids + (size_t) t * k;
            float* row_w = weights + (size_t) t * k;

            int32_t orig_ids[kChordMaxTopK];
            float orig_w[kChordMaxTopK];
            int32_t cur_ids[kChordMaxTopK];
            float cur_w[kChordMaxTopK];
            for (int r = 0; r < k; ++r) {
                orig_ids[r] = cur_ids[r] = row_ids[r];
                orig_w[r] = cur_w[r] = row_w[r];
            }

            const float surv = (t < capped_tok) ? std::clamp(p_alive[t], 0.0f, 1.0f) : 0.1f;
            const float eff_margin = cfg.logit_margin * (1.0f - cfg.survival_strict * surv);
            float committed_tv = 0.0f;
            bool modified = false;

            for (int r = r_lo; r <= r_hi; ++r) {
                const int32_t e = cur_ids[r];
                if (e < 0 || e >= n_expert) continue;
                const ExpertTier tier_e = classify_expert_tier(e, primary_res, peer_res);
                if (tier_e != ExpertTier::ColdHost) continue;

                // Free-ride check: if cold expert `e` is ALREADY required by anchor t=0 or an earlier wave,
                // keeping `e` incurs ZERO extra offloaded expert weight loads!
                if (bm_test(paid_cold_bm, e)) {
                    if (stats != nullptr) ++stats->free_ride_kept;
                    continue;
                }

                // Scan for the highest-utility zero-marginal-miss candidate `f` not already in `cur_ids`.
                float best_score = -INFINITY;
                int32_t best_f = -1;
                ExpertTier best_tier = ExpertTier::ColdHost;

                for (int32_t f = 0; f < n_expert; ++f) {
                    const ExpertTier tier_f = classify_expert_tier(f, primary_res, peer_res);
                    const bool in_paid_union = (tier_f == ExpertTier::ColdHost) && bm_test(paid_cold_bm, f);
                    if (tier_f == ExpertTier::ColdHost && !in_paid_union) continue;

                    if (row_l[e] - row_l[f] > eff_margin) continue;

                    bool already_picked = false;
                    for (int q = 0; q < k; ++q) {
                        if (cur_ids[q] == f) { already_picked = true; break; }
                    }
                    if (already_picked) continue;

                    float penalty = 0.0f;
                    if (tier_f == ExpertTier::PeerVram) penalty = cfg.peer_tier_cost;
                    else if (in_paid_union) penalty = cfg.paid_union_cost;

                    const float score = row_l[f] - penalty;
                    if (score > best_score || (score == best_score && (best_f < 0 || f < best_f))) {
                        best_score = score;
                        best_f = f;
                        best_tier = tier_f;
                    }
                }

                if (best_f < 0) continue;

                // Trial swap + exact sparse TV check against the unbiased original Top-K distribution `orig_w`.
                int32_t trial_ids[kChordMaxTopK];
                float trial_w[kChordMaxTopK];
                for (int q = 0; q < k; ++q) trial_ids[q] = cur_ids[q];
                trial_ids[r] = best_f;
                renormalize_topk_weights(row_l, trial_ids, trial_w, k);

                const float new_tv = compute_sparse_tv(orig_ids, orig_w, trial_ids, trial_w, k);
                const float incremental_tv = std::max(0.0f, new_tv - committed_tv);
                const bool tv_ok = (new_tv <= cfg.tv_budget_token + 1e-7f) &&
                                   (ledger == nullptr || ledger->can_spend(t, incremental_tv, cfg));
                if (!tv_ok) {
                    if (stats != nullptr) ++stats->swaps_rejected_tv;
                    continue;
                }

                // Commit swap!
                cur_ids[r] = best_f;
                for (int q = 0; q < k; ++q) cur_w[q] = trial_w[q];
                if (ledger != nullptr) ledger->commit_spend(t, incremental_tv);
                committed_tv = new_tv;
                modified = true;

                if (stats != nullptr) {
                    if (best_tier == ExpertTier::PrimaryVram) ++stats->swaps_to_primary;
                    else if (best_tier == ExpertTier::PeerVram) ++stats->swaps_to_peer;
                    else ++stats->swaps_to_paid_union;
                }
            }

            if (modified) {
                for (int r = 0; r < k; ++r) {
                    row_ids[r] = cur_ids[r];
                    row_w[r] = cur_w[r];
                }
                if (stats != nullptr) {
                    if (t == 0) ++stats->anchor_swaps;
                    stats->max_token_tv = std::max(stats->max_token_tv, (double) committed_tv);
                }
            }

            // Publish any cold experts retained by token `t` into `paid_cold_bm` so subsequent tail tokens can coalesce!
            for (int r = 0; r < k; ++r) {
                const int32_t e = row_ids[r];
                if (e >= 0 && e < n_expert && classify_expert_tier(e, primary_res, peer_res) == ExpertTier::ColdHost) {
                    bm_set(paid_cold_bm, e);
                }
            }
        }
    };

    // Wave 1: High-survival prefix drafts (t = 1..2).
    const int wave1_start = cfg.protect_anchor ? 1 : 0;
    const int wave1_end = std::min(n_tok, 3);
    process_wave(wave1_start, wave1_end);

    // Wave 2: Lower-survival tail drafts (t = 3..n_tok-1) coalesce onto Wave 0 + Wave 1's union!
    if (wave1_end < n_tok) {
        process_wave(wave1_end, n_tok);
    }

    if (stats != nullptr) {
        uint64_t cold_entries_after = 0;
        for (int t = 0; t < n_tok; ++t) {
            for (int r = 0; r < k; ++r) {
                const int32_t e = ids[(size_t) t * k + r];
                if (e >= 0 && e < n_expert && classify_expert_tier(e, primary_res, peer_res) == ExpertTier::ColdHost) {
                    bm_set(after_cold_bm, e);
                    ++cold_entries_after;
                }
            }
        }
        ++stats->windows_seen;
        stats->unique_cold_before += bm_popcount(before_cold_bm);
        stats->unique_cold_after  += bm_popcount(after_cold_bm);
        stats->cold_entries_before += cold_entries_before;
        stats->cold_entries_after  += cold_entries_after;
    }
}

/// Candidate offloaded expert evaluated by the survival-weighted prefetch planner (Mode 0 & Mode 1).
struct PrefetchCandidate {
    int32_t expert_id   = -1;
    float   p_needed    = 0.0f; ///< Combined survival probability: 1 - prod_{t: e in S_t} (1 - p_alive[t]).
    int32_t first_token = 0;    ///< Earliest window token requesting this expert.
};

/// Mode 0 & Mode 1 Lossless Survival-Weighted Prefetch Filter.
///
/// Fixes two baseline bugs simultaneously:
///   1. Specter / MoE-Infinity equal-vote tail pollution: instead of weighting step-7 drafts equally with step-1,
///      computes exact union need probability `1 - prod(1 - p_alive[t])` and drops candidates below `cfg.prefetch_min_ev`.
///   2. ProMoE +215% duplicate-copy bug: deduplicates across the window AND filters out experts already resident in
///      `primary_res`, `peer_res`, or `in_flight_bm`.
inline std::vector<PrefetchCandidate> chord_score_prefetches(
    const int32_t* predicted_ids, int n_tok, int k, int n_expert,
    const int32_t* primary_res, const ChordConfig& cfg,
    const int32_t* peer_res = nullptr,
    const float* p_alive = nullptr,
    const uint32_t* in_flight_bm = nullptr,
    ChordWindowStats* stats = nullptr) {

    std::vector<PrefetchCandidate> out;
    if (n_tok <= 0 || k <= 0 || n_expert <= 0 || n_expert > kChordMaxExperts) return out;

    float local_surv[kChordMaxWindow];
    const int capped_tok = std::min(n_tok, kChordMaxWindow);
    if (p_alive == nullptr) {
        fill_default_survival(capped_tok, cfg.default_alpha, local_surv);
        p_alive = local_surv;
    }

    std::vector<float> prod_miss((size_t) n_expert, 1.0f);
    std::vector<int32_t> earliest((size_t) n_expert, -1);

    for (int t = 0; t < n_tok; ++t) {
        const float pa = (t < capped_tok) ? std::clamp(p_alive[t], 0.0f, 1.0f) : 0.05f;
        for (int r = 0; r < k; ++r) {
            const int32_t e = predicted_ids[(size_t) t * k + r];
            if (e < 0 || e >= n_expert) continue;
            if (classify_expert_tier(e, primary_res, peer_res) != ExpertTier::ColdHost) continue;
            if (in_flight_bm != nullptr && (((in_flight_bm[e >> 5] >> (e & 31)) & 1u) != 0)) {
                if (stats != nullptr) ++stats->prefetches_dropped;
                continue;
            }
            prod_miss[(size_t) e] *= (1.0f - pa);
            if (earliest[(size_t) e] < 0) earliest[(size_t) e] = t;
        }
    }

    for (int32_t e = 0; e < n_expert; ++e) {
        if (earliest[(size_t) e] < 0) continue;
        const float p_need = 1.0f - prod_miss[(size_t) e];
        if (p_need >= cfg.prefetch_min_ev) {
            out.push_back(PrefetchCandidate{e, p_need, earliest[(size_t) e]});
            if (stats != nullptr) ++stats->prefetches_kept;
        } else {
            if (stats != nullptr) ++stats->prefetches_dropped;
        }
    }

    std::sort(out.begin(), out.end(), [](const PrefetchCandidate& a, const PrefetchCandidate& b) {
        if (std::fabs(a.p_needed - b.p_needed) > 1e-6f) return a.p_needed > b.p_needed;
        if (a.first_token != b.first_token) return a.first_token < b.first_token;
        return a.expert_id < b.expert_id;
    });
    return out;
}

}  // namespace strata::chordmoe
