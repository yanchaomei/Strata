// contrib/llama_cpp_chordmoe/ggml_chord_moe.hpp - Drop-in ChordMoE adapter for llama.cpp / S2-MoE (ggml).
//
// PURPOSE (Cross-Engine Portability & Controlled Algorithmic Ablation):
//   In `llama.cpp` / `S2-MoE` (`src/llama-graph.cpp:1108-1136`), `probs` is computed from unbiased `logits_base`
//   while `logits_sel = ggml_moe_reuse_two_pass_runtime(...)` is fed ONLY into `ggml_top_k` to select the Top-K
//   expert indices (before gathering and renormalizing unbiased `probs`).
//
//   `ggml_chord_moe_transform_selection_logits` plugs directly into `ggml_compute_forward_moe_reuse_two_pass`
//   without modifying a single model graph file in `llama.cpp`:
//     1. Row `b = 0` (anchor token): copied bit-for-bit (`dst_row = src_row`), guaranteeing 0% anchor corruption;
//     2. Rows `b = 1 .. n_tok - 1`: runs `chord_reshape_window` against the layer's VRAM residency table and lifts
//        the certified `target_ids[b, 0..top_k)` by a monotone offset (`+1e4f`) in `dst` so downstream `ggml_top_k`
//        selects the exact `ChordMoE` expert set while downstream `probs` applies the exact unbiased weights!
#pragma once

#include "strata/chordmoe/chord_abort.hpp"
#include "strata/chordmoe/chord_router.hpp"
#include "strata/chordmoe/chord_sched.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <set>
#include <vector>

namespace strata::chordmoe::ggml_adapter {

/// Standard unbiased Top-K softmax router matching `ggml_op_topk_moe` / `llama.cpp` semantics.
inline void ggml_standard_topk_moe(const float* logits, int n_tok, int n_expert, int k,
                                   int32_t* out_ids, float* out_weights, bool norm_topk = true) {
    std::vector<int32_t> idx((size_t) n_expert);
    std::vector<float> probs((size_t) n_expert);
    for (int t = 0; t < n_tok; ++t) {
        const float* row = logits + (size_t) t * n_expert;
        float mx = -INFINITY;
        for (int e = 0; e < n_expert; ++e) mx = std::max(mx, row[e]);
        float sum = 0.0f;
        for (int e = 0; e < n_expert; ++e) {
            probs[(size_t) e] = std::exp(row[e] - mx);
            sum += probs[(size_t) e];
        }
        const float inv = 1.0f / std::max(sum, 1e-30f);
        for (int e = 0; e < n_expert; ++e) {
            probs[(size_t) e] *= inv;
            idx[(size_t) e] = e;
        }
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) {
            if (probs[(size_t) a] != probs[(size_t) b]) return probs[(size_t) a] > probs[(size_t) b];
            return a < b;
        });
        float sel_sum = 0.0f;
        for (int r = 0; r < k; ++r) {
            out_ids[(size_t) t * k + r] = idx[(size_t) r];
            out_weights[(size_t) t * k + r] = probs[(size_t) idx[(size_t) r]];
            sel_sum += out_weights[(size_t) t * k + r];
        }
        if (norm_topk) {
            const float r_inv = 1.0f / std::max(sel_sum, 1e-30f);
            for (int r = 0; r < k; ++r) out_weights[(size_t) t * k + r] *= r_inv;
        }
    }
}

/// Exact C++ reproduction of S2-MoE's `ggml_compute_forward_moe_reuse_two_pass` (`ggml-cpu/ops.cpp:9941-10050`),
/// followed by `llama-graph.cpp`'s `ggml_top_k(logits_sel)` + unbiased `probs` gather.
inline void s2moe_two_pass_router(const float* logits, int n_tok, int n_expert, int k,
                                  float lambda_param, int32_t cap, bool runtime_strength,
                                  int32_t* out_ids, float* out_weights) {
    ggml_standard_topk_moe(logits, n_tok, n_expert, k, out_ids, out_weights, true);
    float lambda = lambda_param;
    if (runtime_strength && k < n_expert && n_tok > 0) {
        float margin_sum = 0.0f;
        std::vector<float> row_copy((size_t) n_expert);
        for (int b = 0; b < n_tok; ++b) {
            const float* src_row = logits + (size_t) b * n_expert;
            std::copy(src_row, src_row + n_expert, row_copy.begin());
            std::nth_element(row_copy.begin(), row_copy.begin() + k, row_copy.end(), std::greater<float>());
            margin_sum += *std::max_element(src_row, src_row + n_expert) - row_copy[(size_t) k];
        }
        lambda = margin_sum / (float) n_tok;
    }
    if (n_tok <= 1 || n_expert <= 1 || lambda <= 0.0f || cap <= 0) return;

    std::vector<float> w_raw((size_t) n_tok, 0.0f);
    float max_w_raw = 0.0f;
    for (int b = 0; b < n_tok; ++b) {
        const float* row = logits + (size_t) b * n_expert;
        float mx = -INFINITY, s = 0.0f;
        for (int e = 0; e < n_expert; ++e) {
            mx = std::max(mx, row[e]);
            s += row[e];
        }
        w_raw[(size_t) b] = std::max(0.0f, mx - (s / (float) n_expert));
        max_w_raw = std::max(max_w_raw, w_raw[(size_t) b]);
    }
    if (max_w_raw <= 0.0f) return;

    std::vector<float> g((size_t) n_expert, 0.0f);
    for (int b = 0; b < n_tok; ++b) {
        const float w = w_raw[(size_t) b] / max_w_raw;
        const float* row = logits + (size_t) b * n_expert;
        for (int e = 0; e < n_expert; ++e) g[(size_t) e] += w * row[e];
    }

    const int k_global = std::max(k, std::min(n_expert, (int) cap));
    std::vector<int32_t> order((size_t) n_expert);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k_global, order.end(), [&](int32_t a, int32_t b) {
        if (g[(size_t) a] != g[(size_t) b]) return g[(size_t) a] > g[(size_t) b];
        return a < b;
    });
    std::vector<uint8_t> in_estar((size_t) n_expert, 0);
    for (int i = 0; i < k_global; ++i) in_estar[(size_t) order[(size_t) i]] = 1;

    std::vector<int32_t> idx((size_t) n_expert);
    for (int b = 0; b < n_tok; ++b) {
        const float* row = logits + (size_t) b * n_expert;
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t e1, int32_t e2) {
            const float s1 = row[e1] + (in_estar[(size_t) e1] ? lambda : 0.0f);
            const float s2 = row[e2] + (in_estar[(size_t) e2] ? lambda : 0.0f);
            if (s1 != s2) return s1 > s2;
            return e1 < e2;
        });

        for (int r = 0; r < k; ++r) out_ids[(size_t) b * k + r] = idx[(size_t) r];
        renormalize_topk_weights(row, out_ids + (size_t) b * k, out_weights + (size_t) b * k, k);
    }
}

/// Direct selection-logit transform `[n_expert, n_tok]` (`src_logits -> dst_logits`) for `ggml_compute_forward_moe_reuse_two_pass`.
/// Downstream `ggml_top_k(dst_logits, k)` + unbiased `probs` gather produces the exact `ChordMoE` routing!
inline void ggml_chord_moe_transform_selection_logits(const float* src_logits, float* dst_logits,
                                                      int n_tok, int n_expert, int k,
                                                      const int32_t* vram_res,
                                                      const ChordConfig& cfg,
                                                      const int32_t* peer_vram_res = nullptr,
                                                      const float* draft_survival = nullptr,
                                                      ChordCreditLedger* ledger = nullptr,
                                                      ChordWindowStats* stats = nullptr) {
    if (n_tok <= 0 || n_expert <= 0) return;
    std::memcpy(dst_logits, src_logits, (size_t) n_tok * (size_t) n_expert * sizeof(float));
    if (cfg.mode != ChordMode::BoundedSetCover || n_tok <= 1 || k <= 0 || k > n_expert || n_expert > kChordMaxExperts) {
        return;
    }

    std::vector<int32_t> ids((size_t) n_tok * (size_t) k);
    std::vector<float> weights((size_t) n_tok * (size_t) k);
    ggml_standard_topk_moe(src_logits, n_tok, n_expert, k, ids.data(), weights.data(), true);
    chord_reshape_window(src_logits, ids.data(), weights.data(), n_tok, n_expert, k,
                         vram_res, cfg, peer_vram_res, draft_survival, ledger, stats);

    // Row 0 stays bit-for-bit identical (`protect_anchor`). For rows b >= 1, lift the certified `ids[b, 0..k)`
    // by a uniform +1e4f offset above the row's maximum so `ggml_top_k(dst_logits, k)` selects the exact set
    // in descending unbiased logit order while `probs` (computed from `logits_base`) supplies exact weights!
    const int start_b = cfg.protect_anchor ? 1 : 0;
    for (int b = start_b; b < n_tok; ++b) {
        const float* srow = src_logits + (size_t) b * n_expert;
        float* drow       = dst_logits + (size_t) b * n_expert;
        float row_max = -INFINITY;
        for (int e = 0; e < n_expert; ++e) row_max = std::max(row_max, srow[e]);
        const float lift = std::max(1e4f, std::fabs(row_max) * 16.0f + 1024.0f);
        for (int r = 0; r < k; ++r) {
            const int32_t chosen_e = ids[(size_t) b * k + r];
            if (chosen_e >= 0 && chosen_e < n_expert) {
                drow[chosen_e] = srow[chosen_e] + lift;
            }
        }
    }
}

/// Convenience end-to-end helper returning `(out_ids, out_weights)` directly.
inline void ggml_chord_moe_forward_window(const float* logits, int n_tok, int n_expert, int k,
                                          const int32_t* vram_res,
                                          const ChordConfig& cfg,
                                          int32_t* out_ids, float* out_weights,
                                          const int32_t* peer_vram_res = nullptr,
                                          const float* draft_survival = nullptr,
                                          ChordCreditLedger* ledger = nullptr,
                                          ChordWindowStats* stats = nullptr) {
    ggml_standard_topk_moe(logits, n_tok, n_expert, k, out_ids, out_weights, true);
    if (cfg.mode == ChordMode::BoundedSetCover && n_tok > 1) {
        chord_reshape_window(logits, out_ids, out_weights, n_tok, n_expert, k,
                             vram_res, cfg, peer_vram_res, draft_survival, ledger, stats);
    }
}

/// Environment-controlled runtime hook for `ggml_compute_forward_moe_reuse_two_pass` in `llama.cpp` / `S2-MoE`:
///   - When `CHORDMOE_ENABLE=1`, intercepts `src_logits -> dst_logits` using `ChordMoE` Mode 1 set-cover routing
///     with static VRAM residency ratio `CHORDMOE_VRAM_RATIO` (default `0.32f`) and prints cumulative window stats at exit.
///   - Returns `true` if intercepted (`dst_logits` populated), `false` to fall through to original `S2-MoE`.
inline bool ggml_chord_moe_runtime_intercept(const float* src_logits, float* dst_logits,
                                             int n_tok, int n_expert, int top_k) {
    static const bool enabled = [] {
        const char* v = std::getenv("CHORDMOE_ENABLE");
        return v != nullptr && v[0] != '\0' && std::atoi(v) != 0;
    }();
    if (!enabled) return false;

    static const float vram_ratio = [] {
        const char* v = std::getenv("CHORDMOE_VRAM_RATIO");
        return (v != nullptr && v[0] != '\0') ? std::clamp((float) std::atof(v), 0.0f, 1.0f) : 0.32f;
    }();
    static const ChordConfig cfg = [top_k] {
        ChordConfig c = ChordConfig::from_env();
        c.mode = ChordMode::BoundedSetCover;
        if (std::getenv("STRATA_CHORD_RANKS") == nullptr) {
            c.rank_lo = std::max(1, top_k / 2 - 1);
            c.rank_hi = 31;
        }
        return c;
    }();

    struct GlobalChordTelemetry {
        std::mutex mu;
        ChordWindowStats st{};
        GlobalChordTelemetry() {
            std::atexit([] {
                // Accessed after main returns; single-threaded teardown
            });
        }
        ~GlobalChordTelemetry() {
            if (st.windows_seen > 0) {
                const double w = (double) st.windows_seen;
                std::fprintf(stderr,
                    "\n[ChordMoE-ggml] verify_layer_windows=%llu | anchor_swaps=%llu | "
                    "cold_misses/layer: %.2f -> %.2f (-%.1f%%) | free_rides=%llu | max_TV=%.3f\n",
                    (unsigned long long) st.windows_seen,
                    (unsigned long long) st.anchor_swaps,
                    (double) st.unique_cold_before / w,
                    (double) st.unique_cold_after / w,
                    st.unique_cold_before ? 100.0 * (double) (st.unique_cold_before - st.unique_cold_after) / (double) st.unique_cold_before : 0.0,
                    (unsigned long long) st.free_ride_kept,
                    st.max_token_tv);
            }
        }
    };
    static GlobalChordTelemetry telem;

    const int n_res = std::clamp((int) std::round(vram_ratio * (float) n_expert), 0, n_expert);
    std::vector<int32_t> vram_res((size_t) n_expert, -1);
    for (int e = 0; e < n_res; ++e) vram_res[(size_t) e] = e;

    ChordWindowStats local_st{};
    ggml_chord_moe_transform_selection_logits(src_logits, dst_logits, n_tok, n_expert, top_k,
                                              vram_res.data(), cfg, nullptr, nullptr, nullptr, &local_st);
    if (local_st.windows_seen > 0) {
        std::lock_guard<std::mutex> lk(telem.mu);
        telem.st.windows_seen       += local_st.windows_seen;
        telem.st.anchor_swaps       += local_st.anchor_swaps;
        telem.st.unique_cold_before += local_st.unique_cold_before;
        telem.st.unique_cold_after  += local_st.unique_cold_after;
        telem.st.free_ride_kept     += local_st.free_ride_kept;
        telem.st.swaps_rejected_tv  += local_st.swaps_rejected_tv;
        telem.st.max_token_tv        = std::max(telem.st.max_token_tv, local_st.max_token_tv);
    }
    return true;
}

/// Optional audit observer for stock `S2-MoE` (`S2MOE_AUDIT_STATS=1`) measuring the exact same metrics
/// (`anchor_corruptions`, `cold_before -> cold_after`, `max_TV`) after `S2-MoE` populates `dst_logits`.
inline void ggml_s2moe_audit_observe(const float* src_logits, const float* dst_logits,
                                     int n_tok, int n_expert, int top_k) {
    static const bool on = [] {
        const char* v = std::getenv("S2MOE_AUDIT_STATS");
        return v != nullptr && v[0] != '\0' && std::atoi(v) != 0;
    }();
    if (!on || n_tok <= 1 || n_expert <= 1 || top_k <= 0) return;

    static const float vram_ratio = [] {
        const char* v = std::getenv("CHORDMOE_VRAM_RATIO");
        return (v != nullptr && v[0] != '\0') ? std::clamp((float) std::atof(v), 0.0f, 1.0f) : 0.32f;
    }();
    struct S2Audit {
        std::mutex mu;
        uint64_t windows = 0, anchor_corrupt = 0, cold_before = 0, cold_after = 0;
        double max_tv = 0.0;
        ~S2Audit() {
            if (windows > 0) {
                std::fprintf(stderr,
                    "\n[S2-MoE-audit]  verify_layer_windows=%llu | anchor_corrupted=%llu/%llu (%.1f%%) | "
                    "cold_misses/layer: %.2f -> %.2f | max_TV=%.3f\n",
                    (unsigned long long) windows,
                    (unsigned long long) anchor_corrupt, (unsigned long long) windows,
                    100.0 * (double) anchor_corrupt / (double) windows,
                    (double) cold_before / (double) windows,
                    (double) cold_after / (double) windows,
                    max_tv);
            }
        }
    };
    static S2Audit audit;

    const int n_res = std::clamp((int) std::round(vram_ratio * (float) n_expert), 0, n_expert);
    std::vector<int32_t> base_ids((size_t) n_tok * top_k), s2_ids((size_t) n_tok * top_k);
    std::vector<float> base_w((size_t) n_tok * top_k), s2_w((size_t) n_tok * top_k);
    ggml_standard_topk_moe(src_logits, n_tok, n_expert, top_k, base_ids.data(), base_w.data(), true);

    std::vector<int32_t> idx((size_t) n_expert);
    for (int b = 0; b < n_tok; ++b) {
        const float* drow = dst_logits + (size_t) b * n_expert;
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int32_t a, int32_t c) {
            if (drow[a] != drow[c]) return drow[a] > drow[c];
            return a < c;
        });
        for (int r = 0; r < top_k; ++r) s2_ids[(size_t) b * top_k + r] = idx[(size_t) r];
        renormalize_topk_weights(src_logits + (size_t) b * n_expert, s2_ids.data() + (size_t) b * top_k,
                                 s2_w.data() + (size_t) b * top_k, top_k);
    }

    std::set<int32_t> cb, ca;
    for (int e : base_ids) if (e >= n_res) cb.insert(e);
    for (int e : s2_ids)   if (e >= n_res) ca.insert(e);
    std::set<int32_t> a0(base_ids.begin(), base_ids.begin() + top_k);
    std::set<int32_t> a1(s2_ids.begin(), s2_ids.begin() + top_k);
    double local_tv = 0.0;
    for (int b = 0; b < n_tok; ++b) {
        local_tv = std::max(local_tv, (double) compute_sparse_tv(
            base_ids.data() + b * top_k, base_w.data() + b * top_k,
            s2_ids.data() + b * top_k, s2_w.data() + b * top_k, top_k));
    }

    std::lock_guard<std::mutex> lk(audit.mu);
    ++audit.windows;
    if (a0 != a1) ++audit.anchor_corrupt;
    audit.cold_before += (uint64_t) cb.size();
    audit.cold_after  += (uint64_t) ca.size();
    audit.max_tv       = std::max(audit.max_tv, local_tv);
}

}  // namespace strata::chordmoe::ggml_adapter
