// contrib/llama_cpp_chordmoe/ggml_chord_moe.hpp - Drop-in ChordMoE adapter for llama.cpp / S2-MoE (ggml).
//
// PURPOSE (Cross-Engine Portability & Controlled Algorithmic Ablation):
//   Plugs `strata::chordmoe::chord_reshape_window` directly into `llama.cpp` / `S2-MoE`'s speculative MoE
//   routing tensor `[n_expert, n_tokens]` (column-major in ggml, i.e., `logits[t * n_expert + e]` in C memory)
//   for ANY standard GGUF MoE architecture:
//     - Mixtral-8x7B / 8x22B (`8x2`)
//     - Qwen3-30B-A3B / Qwen3-235B-A22B (`128x8`)
//     - DeepSeek-V2 / DeepSeek-V3 / GLM-4.5-Air (`160x6`, `256x8`)
//     - Qwen3.6-35B-A3B (`256x8`) & Qwen3.8-Flash-Next (`512x10`)
//
// Also provides an exact C++ port of S2-MoE's `ggml_compute_forward_moe_reuse_two_pass` (`ops.cpp:9941-10050`)
// so head-to-head controlled experiments run in identical C++ memory without framework confounds.
#pragma once

#include "strata/chordmoe/chord_abort.hpp"
#include "strata/chordmoe/chord_router.hpp"
#include "strata/chordmoe/chord_sched.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
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

/// Exact C++ reproduction of S2-MoE's `ggml_compute_forward_moe_reuse_two_pass` (`ggml-cpu/ops.cpp:9941-10050`).
/// Preserves S2-MoE's exact router-margin confidence proxy (`max - mean`), residency-blind preferred set `E*`
/// of size `M = ceil(m_factor * k)`, and biased Pass-2 re-selection across all rows `b = 0 .. B-1` (including anchor b=0).
inline void s2moe_two_pass_router(const float* logits, int n_tok, int n_expert, int k,
                                  float beta, float m_factor,
                                  int32_t* out_ids, float* out_weights) {
    // Pass 1: initial unbiased Top-K + router `max - mean` proxy `w_b` (`ops.cpp:9990-10004`).
    ggml_standard_topk_moe(logits, n_tok, n_expert, k, out_ids, out_weights, true);
    if (n_tok <= 1 || beta <= 0.0f) return;

    std::vector<float> wb((size_t) n_tok, 0.0f);
    for (int b = 0; b < n_tok; ++b) {
        const float* row = logits + (size_t) b * n_expert;
        float mx = -INFINITY, s = 0.0f;
        for (int e = 0; e < n_expert; ++e) {
            mx = std::max(mx, row[e]);
            s += row[e];
        }
        wb[(size_t) b] = mx - (s / (float) n_expert);
    }

    // Aggregate `S(e) = sum_{b: e in TopK(b)} w_b` (`ops.cpp:10010-10018`) -- NOTE: completely ignores VRAM residency!
    std::vector<float> score((size_t) n_expert, 0.0f);
    for (int b = 0; b < n_tok; ++b) {
        for (int r = 0; r < k; ++r) {
            const int32_t e = out_ids[(size_t) b * k + r];
            if (e >= 0 && e < n_expert) score[(size_t) e] += wb[(size_t) b];
        }
    }

    const int M = std::clamp((int) std::ceil(m_factor * (float) k), k, n_expert);
    std::vector<int32_t> order((size_t) n_expert);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + M, order.end(), [&](int32_t a, int32_t b) {
        if (score[(size_t) a] != score[(size_t) b]) return score[(size_t) a] > score[(size_t) b];
        return a < b;
    });
    std::vector<uint8_t> in_estar((size_t) n_expert, 0);
    for (int i = 0; i < M; ++i) in_estar[(size_t) order[(size_t) i]] = 1;

    // Pass 2: bias logits of `e in E*` by `+beta * std_b` for EVERY token `b = 0 .. n_tok - 1` (`ops.cpp:10035-10050`),
    // selecting with biased logits and renormalizing unbiased softmax weights.
    std::vector<int32_t> idx((size_t) n_expert);
    for (int b = 0; b < n_tok; ++b) {
        const float* row = logits + (size_t) b * n_expert;
        float mean = 0.0f;
        for (int e = 0; e < n_expert; ++e) mean += row[e];
        mean /= (float) n_expert;
        float var = 0.0f;
        for (int e = 0; e < n_expert; ++e) {
            const float d = row[e] - mean;
            var += d * d;
        }
        const float stdev = std::sqrt(var / (float) n_expert + 1e-6f);

        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t e1, int32_t e2) {
            const float s1 = row[e1] + (in_estar[(size_t) e1] ? beta * stdev : 0.0f);
            const float s2 = row[e2] + (in_estar[(size_t) e2] ? beta * stdev : 0.0f);
            if (s1 != s2) return s1 > s2;
            return e1 < e2;
        });

        for (int r = 0; r < k; ++r) out_ids[(size_t) b * k + r] = idx[(size_t) r];
        renormalize_topk_weights(row, out_ids + (size_t) b * k, out_weights + (size_t) b * k, k);
    }
}

/// Drop-in ChordMoE replacement for `ggml_compute_forward_moe_reuse_two_pass` in `llama.cpp` / `S2-MoE`.
/// Pass the ggml backend residency array `vram_res[0..n_expert)` (`>= 0` if expert slice is on GPU, `< 0` if on CPU).
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

}  // namespace strata::chordmoe::ggml_adapter
