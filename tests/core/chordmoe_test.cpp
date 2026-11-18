// tests/core/chordmoe_test.cpp - Unit tests & head-to-head C++ microbenchmark for ChordMoE (no GPU needed).
#include "strata/chordmoe/chord_abort.hpp"
#include "strata/chordmoe/chord_router.hpp"
#include "strata/chordmoe/chord_sched.hpp"
#include "strata/chordmoe/chord_types.hpp"
#include "ggml_chord_moe.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <vector>

static int g_fails = 0;
#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_fails;                                                                   \
        }                                                                                \
    } while (0)

namespace {

using namespace strata::chordmoe;
using namespace strata::chordmoe::ggml_adapter;

// Generate realistic Zipf-prior + Gaussian perturbation logits `[n_tok, n_expert]`.
void make_zipf_window_logits(std::mt19937& rng, int n_tok, int n_expert, float zipf_s, float noise_sd,
                             std::vector<float>& logits) {
    logits.resize((size_t) n_tok * (size_t) n_expert);
    std::normal_distribution<float> norm(0.0f, noise_sd);
    for (int t = 0; t < n_tok; ++t) {
        for (int e = 0; e < n_expert; ++e) {
            const float prior = -zipf_s * std::log((float) (e + 4));
            logits[(size_t) t * (size_t) n_expert + (size_t) e] = prior + norm(rng);
        }
    }
}

uint64_t count_distinct_cold(const int32_t* ids, int n_tok, int k,
                             const int32_t* pri_res, const int32_t* peer_res = nullptr) {
    std::set<int32_t> cold;
    for (int i = 0; i < n_tok * k; ++i) {
        const int32_t e = ids[i];
        if (classify_expert_tier(e, pri_res, peer_res) == ExpertTier::ColdHost) {
            cold.insert(e);
        }
    }
    return (uint64_t) cold.size();
}

void test_default_off_noop() {
    ChordConfig cfg; // default Off
    CHECK(cfg.mode == ChordMode::Off);
    CHECK(!cfg.early_abort);
    CHECK(!cfg.split_offload);
    CHECK(!cfg.enabled());

    std::mt19937 rng(42);
    std::vector<float> logits;
    make_zipf_window_logits(rng, 6, 256, 0.85f, 1.1f, logits);
    std::vector<int32_t> ids(6 * 8), ids_ref(6 * 8);
    std::vector<float> w(6 * 8), w_ref(6 * 8);
    ggml_standard_topk_moe(logits.data(), 6, 256, 8, ids.data(), w.data());
    ids_ref = ids;
    w_ref = w;

    std::vector<int32_t> res(256, -1);
    for (int i = 0; i < 80; ++i) res[(size_t) i] = i;

    chord_reshape_window(logits.data(), ids.data(), w.data(), 6, 256, 8, res.data(), cfg);
    CHECK(std::memcmp(ids.data(), ids_ref.data(), ids.size() * sizeof(int32_t)) == 0);
    CHECK(std::memcmp(w.data(), w_ref.data(), w.size() * sizeof(float)) == 0);
}

void test_anchor_protection_and_tv_bounds() {
    struct ArchCase { int ne; int k; int vram_slots; int peer_slots; };
    const ArchCase archs[] = {
        {512, 10, 140, 60}, // Qwen3.8-Flash-Next (2 GPUs)
        {256, 8,  82,  0},  // Qwen3.6-35B-A3B / DeepSeek-V3
        {128, 8,  40,  16}, // Qwen3-235B-A22B
        {8,   2,  3,   0},  // Mixtral-8x22B
    };

    std::mt19937 rng(1337);
    for (const auto& ac : archs) {
        std::vector<int32_t> pri_res((size_t) ac.ne, -1);
        std::vector<int32_t> peer_res((size_t) ac.ne, -1);
        for (int i = 0; i < ac.vram_slots; ++i) pri_res[(size_t) i] = i;
        for (int i = ac.vram_slots; i < ac.vram_slots + ac.peer_slots; ++i) peer_res[(size_t) i] = i - ac.vram_slots;

        ChordConfig cfg;
        cfg.mode = ChordMode::BoundedSetCover;
        cfg.logit_margin = 1.35f;
        cfg.tv_budget_token = 0.14f;
        cfg.tv_budget_turn = 1.60f;
        cfg.rank_lo = std::max(1, ac.k / 2 - 1);
        cfg.rank_hi = ac.k - 1;
        cfg.protect_anchor = true;

        ChordCreditLedger ledger(8);
        ChordWindowStats stats{};
        const int T = 7;

        for (int layer = 0; layer < 48; ++layer) {
            std::vector<float> logits;
            make_zipf_window_logits(rng, T, ac.ne, 0.75f, 1.15f, logits);

            std::vector<int32_t> orig_ids((size_t) T * ac.k), ids((size_t) T * ac.k);
            std::vector<float> orig_w((size_t) T * ac.k), w((size_t) T * ac.k);
            ggml_standard_topk_moe(logits.data(), T, ac.ne, ac.k, orig_ids.data(), orig_w.data());
            ids = orig_ids;
            w = orig_w;

            chord_reshape_window(logits.data(), ids.data(), w.data(), T, ac.ne, ac.k,
                                 pri_res.data(), cfg,
                                 ac.peer_slots > 0 ? peer_res.data() : nullptr,
                                 nullptr, &ledger, &stats);

            // 1. Anchor t = 0 MUST be 100% bit-for-bit identical!
            CHECK(std::memcmp(ids.data(), orig_ids.data(), (size_t) ac.k * sizeof(int32_t)) == 0);
            CHECK(std::memcmp(w.data(), orig_w.data(), (size_t) ac.k * sizeof(float)) == 0);

            // 2. Every draft token's sparse TV MUST stay strictly within `tv_budget_token`,
            //    weights must sum to 1.0, and selected IDs per token must be distinct!
            for (int t = 1; t < T; ++t) {
                const float tv = compute_sparse_tv(orig_ids.data() + (size_t) t * ac.k,
                                                   orig_w.data() + (size_t) t * ac.k,
                                                   ids.data() + (size_t) t * ac.k,
                                                   w.data() + (size_t) t * ac.k,
                                                   ac.k);
                CHECK(tv <= cfg.tv_budget_token + 1e-5f);
                CHECK(ledger.accumulated(t) <= cfg.tv_budget_turn + 1e-5f);

                float wsum = 0.0f;
                std::set<int32_t> uniq;
                for (int r = 0; r < ac.k; ++r) {
                    uniq.insert(ids[(size_t) t * ac.k + r]);
                    wsum += w[(size_t) t * ac.k + r];
                }
                CHECK((int) uniq.size() == ac.k);
                CHECK(std::fabs(wsum - 1.0f) < 1e-5f);
            }
        }
        CHECK(stats.anchor_swaps == 0);
        CHECK(stats.unique_cold_after < stats.unique_cold_before);
        if (ac.peer_slots > 0) CHECK(stats.swaps_to_peer > 0);
    }
}

void test_free_ride_union_preservation() {
    // Anchor t=0 selects cold expert 200 at rank 0 (must be loaded anyway!).
    // Draft t=1 also selects cold expert 200 at tail rank 7, while resident expert 5 has a nearby logit.
    // Naive per-token swap (STRATA_ROUTE_RESIDENT) swaps 200 -> 5 on t=1, distorting t=1 for ZERO saved bytes.
    // ChordMoE sees 200 in `paid_cold_bm` and keeps 200 on t=1 with 0 distortion!
    const int NE = 256, K = 8, T = 2;
    std::vector<float> logits(T * NE, -10.0f);
    std::vector<int32_t> res(NE, -1);
    for (int e = 0; e < 64; ++e) res[(size_t) e] = e; // 0..63 resident, 64..255 cold

    // Anchor t=0: rank 0 is cold expert 200, ranks 1..7 are resident 0..6
    logits[200] = 9.0f;
    for (int r = 1; r < K; ++r) logits[r - 1] = 8.0f - 0.2f * (float) r;

    // Draft t=1: ranks 0..6 are resident 10..16, rank 7 is cold expert 200 (logit 3.0), resident 5 has logit 2.85
    for (int r = 0; r < K - 1; ++r) logits[NE + 10 + r] = 8.0f - 0.2f * (float) r;
    logits[NE + 200] = 3.0f;
    logits[NE + 5]   = 2.85f; // within margin 1.0, so naive per-token router would swap 200 -> 5!

    std::vector<int32_t> ids(T * K), orig_ids(T * K);
    std::vector<float> w(T * K), orig_w(T * K);
    ggml_standard_topk_moe(logits.data(), T, NE, K, ids.data(), w.data());
    orig_ids = ids;
    orig_w = w;

    ChordConfig cfg;
    cfg.mode = ChordMode::BoundedSetCover;
    cfg.logit_margin = 1.5f;
    cfg.tv_budget_token = 0.25f;
    cfg.rank_lo = 4;
    cfg.rank_hi = 7;

    ChordWindowStats stats{};
    chord_reshape_window(logits.data(), ids.data(), w.data(), T, NE, K, res.data(), cfg, nullptr, nullptr, nullptr, &stats);

    CHECK(stats.free_ride_kept == 1);
    CHECK(ids[K + 7] == 200); // Preserved exact cold expert 200 at zero extra miss cost!
    CHECK(compute_sparse_tv(orig_ids.data() + K, orig_w.data() + K, ids.data() + K, w.data() + K, K) == 0.0f);
}

void test_survival_prefetch_filter() {
    const int NE = 128, K = 4, T = 8;
    std::vector<int32_t> res(NE, -1);
    for (int e = 0; e < 32; ++e) res[(size_t) e] = e; // 0..31 resident

    // Construct predicted window IDs:
    // - t=0..1 request resident experts 0..2 and cold expert 50 (high survival -> keep!)
    // - t=7 (deep tail, survival 0.7^7 = 0.082 < 0.22) requests cold experts 90, 91, 92 (doomed tail -> drop!)
    // - cold expert 60 is already in `in_flight_bm` (ProMoE duplicate bug -> drop!)
    std::vector<int32_t> pred(T * K, 0);
    pred[1 * K + 0] = 50;
    pred[1 * K + 1] = 60;
    pred[7 * K + 0] = 90;
    pred[7 * K + 1] = 91;
    pred[7 * K + 2] = 92;

    uint32_t inflight[4] = {};
    inflight[60 >> 5] |= (1u << (60 & 31));

    ChordConfig cfg;
    cfg.mode = ChordMode::Lossless;
    cfg.default_alpha = 0.70f;
    cfg.prefetch_min_ev = 0.22f;

    ChordWindowStats stats{};
    auto cands = chord_score_prefetches(pred.data(), T, K, NE, res.data(), cfg, nullptr, nullptr, inflight, &stats);

    CHECK(cands.size() == 1);
    if (!cands.empty()) {
        CHECK(cands[0].expert_id == 50);
        CHECK(std::fabs(cands[0].p_needed - 0.70f) < 1e-5f);
    }
    CHECK(stats.prefetches_kept == 1);
    CHECK(stats.prefetches_dropped >= 4); // 60 (in-flight) + 90, 91, 92 (low survival)
}

void test_mid_window_early_abort() {
    constexpr int T = 8, K = 4, H = 64, NL = 48;
    struct FakePoolState {
        int64_t rows_computed_per_layer[NL] = {};
    };
    auto fake_pool = [](void* u, const float* x_f, const int32_t* ids,
                        int64_t n_tok, int64_t k, float* out, int64_t layer) {
        auto* st = static_cast<FakePoolState*>(u);
        st->rows_computed_per_layer[layer] += n_tok;
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t r = 0; r < k; ++r) {
                const float scale = (float) (ids[t * k + r] + 1) * 0.01f + (float) layer * 0.1f;
                for (int64_t d = 0; d < H; ++d) {
                    out[(t * k + r) * H + d] = x_f[t * H + d] * scale;
                }
            }
        }
    };

    std::vector<float> x(T * H, 1.5f);
    std::vector<int32_t> ids(T * K);
    for (int i = 0; i < T * K; ++i) ids[(size_t) i] = 40 + i; // distinct experts per (t, r)

    // Full reference run (no abort)
    FakePoolState ref_st{};
    std::vector<std::vector<float>> ref_out(NL, std::vector<float>(T * K * H, -999.0f));
    for (int l = 0; l < NL; ++l) {
        fake_pool(&ref_st, x.data(), ids.data(), T, K, ref_out[(size_t) l].data(), l);
    }

    // Aborted run: at layer 12, cap live prefix to `n_keep = 3` (`t = 0, 1, 2` kept; `t = 3..7` dead suffix!)
    FakePoolState abort_st{};
    ChordAbortController ctrl;
    ctrl.begin_window(T, H);
    std::vector<std::vector<float>> got_out(NL, std::vector<float>(T * K * H, -999.0f));

    for (int l = 0; l < NL; ++l) {
        if (l == 12) {
            std::thread sig([&ctrl] { ctrl.cap_live_prefix(3, 12); });
            sig.join();
        }
        ctrl.dispatch_subwindow(fake_pool, &abort_st, 0, x.data(), ids.data(), T, K, got_out[(size_t) l].data(), l);
    }

    CHECK(ctrl.live_rows() == 3);
    CHECK(ctrl.abort_layer() == 12);
    CHECK(ctrl.elided_row_layers() == (uint64_t) (NL - 12) * (T - 3)); // 36 * 5 = 180 row-layers skipped!

    for (int l = 0; l < NL; ++l) {
        // Committed prefix [0..3) MUST be 100% bit-for-bit identical across ALL 48 layers!
        CHECK(std::memcmp(got_out[(size_t) l].data(),
                          ref_out[(size_t) l].data(),
                          (size_t) (3 * K * H) * sizeof(float)) == 0);
        if (l >= 12) {
            CHECK(abort_st.rows_computed_per_layer[l] == 3);
            for (size_t idx = (size_t) (3 * K * H); idx < (size_t) (T * K * H); ++idx) {
                CHECK(got_out[(size_t) l][idx] == 0.0f);
            }
        }
    }

    std::vector<int32_t> pri_res(256, -1);
    auto live_exps = ctrl.filter_live_experts(ids.data(), T, K, pri_res.data());
    CHECK((int) live_exps.size() == 3 * K); // 12 live experts instead of 32!
}

void test_hetero_minmax_scheduler_vs_fiddler() {
    // 2x PCIe Gen4 GPUs (24 GB/s each) + 2x Dual-Socket Xeon/EPYC DDR5 NUMA nodes (90 GB/s local, 1.65x remote penalty)
    std::vector<HeteroQueueSpec> queues = {
        {ExecBackendKind::PrimaryPcieDma, 0, 24.0, 2.5, 0.15, 1.0},
        {ExecBackendKind::HelperGpuPcie,  1, 24.0, 3.0, 0.20, 1.0},
        {ExecBackendKind::CpuNumaSocket,  0, 85.0, 1.5, 1.80, 1.65},
        {ExecBackendKind::CpuNumaSocket,  1, 85.0, 1.5, 1.80, 1.65},
    };

    std::vector<MissedExpertTask> tasks;
    for (int i = 0; i < 28; ++i) {
        MissedExpertTask t;
        t.expert_id  = 100 + i;
        t.token_rows = (i < 6) ? 4 : 1;      // 6 multi-token shared misses + 22 single-token tail misses
        t.blob_bytes = 2621440;              // ~2.62 MB per expert (Q4 MoE expert)
        t.home_numa  = i % 2;                // Striped across NUMA 0 and NUMA 1
        tasks.push_back(t);
    }

    const auto chord_plan   = ChordHeteroScheduler::partition_layer_misses(tasks, queues);
    const auto fiddler_plan = ChordHeteroScheduler::fiddler_sum_objective_baseline(tasks, queues);

    CHECK(chord_plan.assignments.size() == tasks.size());
    // Regime A (striped across both 85 GB/s NUMA sockets + two 24 GB/s PCIe links: theoretical lower bound 170/218 = 0.78x):
    CHECK(chord_plan.makespan_us < fiddler_plan.makespan_us * 0.85);

    // Regime B (2x PCIe Gen5 x16 @ 52 GB/s + 2x DDR5 sockets @ 55 GB/s, tasks skewed onto NUMA 0):
    // Fiddler piles every task onto NUMA 0 (`min delta`), wasting 3 of 4 queues (~3.3x slower makespan)!
    std::vector<HeteroQueueSpec> sym_queues = {
        {ExecBackendKind::PrimaryPcieDma, 0, 52.0, 2.0, 0.15, 1.0},
        {ExecBackendKind::HelperGpuPcie,  1, 52.0, 2.5, 0.20, 1.0},
        {ExecBackendKind::CpuNumaSocket,  0, 55.0, 1.5, 1.20, 1.65},
        {ExecBackendKind::CpuNumaSocket,  1, 55.0, 1.5, 1.20, 1.65},
    };
    for (auto& t : tasks) t.home_numa = 0;
    const auto chord_sym   = ChordHeteroScheduler::partition_layer_misses(tasks, sym_queues);
    const auto fiddler_sym = ChordHeteroScheduler::fiddler_sum_objective_baseline(tasks, sym_queues);
    std::printf("chordmoe_test [Hetero Makespan 28 misses]:\n"
                "  Regime A (2xGen4 + 2xNUMA striped): Fiddler %.1f us -> ChordMoE %.1f us\n"
                "  Regime B (2xGen5 + 2xNUMA hotspot): Fiddler %.1f us -> ChordMoE %.1f us\n",
                fiddler_plan.makespan_us, chord_plan.makespan_us,
                fiddler_sym.makespan_us, chord_sym.makespan_us);
    CHECK(chord_sym.makespan_us < fiddler_sym.makespan_us * 0.40);
}

void test_ggml_adapter_vs_s2moe_head_to_head() {
    // 48-layer controlled C++ comparison on 256-expert Top-8 MoE (32% static VRAM residency, W=7 window).
    constexpr int NE = 256, K = 8, T = 7, NL = 48;
    std::vector<int32_t> vram_res(NE, -1);
    for (int e = 0; e < 82; ++e) vram_res[(size_t) e] = e; // Top 32% hottest experts resident in VRAM

    ChordConfig cfg;
    cfg.mode = ChordMode::BoundedSetCover;
    cfg.logit_margin = 1.35f;
    cfg.tv_budget_token = 0.15f;
    cfg.tv_budget_turn = 2.50f;
    cfg.rank_lo = 3;
    cfg.rank_hi = 7;

    std::mt19937 rng(202611);
    uint64_t base_cold = 0, s2_cold = 0, chord_cold = 0;
    uint64_t s2_anchor_corrupted = 0, chord_anchor_corrupted = 0;
    double chord_ns_total = 0.0;

    for (int l = 0; l < NL; ++l) {
        std::vector<float> logits;
        make_zipf_window_logits(rng, T, NE, 0.72f, 1.10f, logits);

        std::vector<int32_t> base_ids(T * K), s2_ids(T * K), ch_ids(T * K);
        std::vector<float> base_w(T * K), s2_w(T * K), ch_w(T * K);

        ggml_standard_topk_moe(logits.data(), T, NE, K, base_ids.data(), base_w.data());
        s2moe_two_pass_router(logits.data(), T, NE, K, /*beta=*/0.15f, /*m_factor=*/1.5f, s2_ids.data(), s2_w.data());

        const auto t0 = std::chrono::steady_clock::now();
        ggml_chord_moe_forward_window(logits.data(), T, NE, K, vram_res.data(), cfg, ch_ids.data(), ch_w.data());
        const auto t1 = std::chrono::steady_clock::now();
        chord_ns_total += (double) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        base_cold  += count_distinct_cold(base_ids.data(), T, K, vram_res.data());
        s2_cold    += count_distinct_cold(s2_ids.data(), T, K, vram_res.data());
        chord_cold += count_distinct_cold(ch_ids.data(), T, K, vram_res.data());

        if (std::memcmp(s2_ids.data(), base_ids.data(), K * sizeof(int32_t)) != 0) ++s2_anchor_corrupted;
        if (std::memcmp(ch_ids.data(), base_ids.data(), K * sizeof(int32_t)) != 0) ++chord_anchor_corrupted;
    }

    std::printf("chordmoe_test [48L 256x8 T=7 32%% VRAM]:\n"
                "  Baseline cold misses/layer : %.2f\n"
                "  S2-MoE   cold misses/layer : %.2f (anchor corrupted in %llu/%d layers)\n"
                "  ChordMoE cold misses/layer : %.2f (anchor corrupted in %llu/%d layers, %.2f us/layer CPU)\n",
                (double) base_cold / NL,
                (double) s2_cold / NL, (unsigned long long) s2_anchor_corrupted, NL,
                (double) chord_cold / NL, (unsigned long long) chord_anchor_corrupted, NL,
                (chord_ns_total / NL) / 1000.0);

    CHECK(chord_anchor_corrupted == 0);
    CHECK(s2_anchor_corrupted > 0);
    CHECK(chord_cold < base_cold);
    CHECK(chord_cold < s2_cold);
}

}  // namespace

int main() {
    test_default_off_noop();
    test_anchor_protection_and_tv_bounds();
    test_free_ride_union_preservation();
    test_survival_prefetch_filter();
    test_mid_window_early_abort();
    test_hetero_minmax_scheduler_vs_fiddler();
    test_ggml_adapter_vs_s2moe_head_to_head();

    if (g_fails == 0) {
        std::printf("chordmoe_test: all checks passed\n");
        return 0;
    }
    return 1;
}
