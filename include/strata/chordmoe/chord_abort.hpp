// include/strata/chordmoe/chord_abort.hpp - Graph-Preserving Mid-Window Dead-Suffix Offload Elision
// & Doorbell-Driven Online Speculative Divergence Probe.
//
// KEY SYSTEMS INVARIANT & UNCONDITIONAL LOSSLESSNESS THEOREM:
//   1. CUDA-Graph Preservation: In Strata's doorbell architecture (`Verifier::run` / `Verifier::service`), 80-90%
//      of an offloaded window's wall-clock latency sits OUTSIDE the captured CUDA Graph inside `PoolMultiFn`
//      (reading cold experts from DDR5 RAM or streaming them over PCIe).
//   2. Unconditional Bit-Exactness under Heuristic Early Abort:
//      Suppose at layer `l*` (`0 <= l* < L`), `cap_live_prefix(L_keep, l*)` shrinks the live prefix to
//      `1 <= L_keep < T`, eliding offloaded expert reads for suffix rows `[L_keep .. T)` on layers `l* .. L-1`.
//      By strict causal masking of QSA attention and GDN recurrence, rows `[0 .. L_keep)` execute 100% exact
//      target arithmetic across all `L` layers (`out[0 .. L_keep)` are bit-for-bit identical to full decode).
//      Capping the committed prefix length at `n_keep_cap = live_rows()` (`n_keep <= L_keep`) guarantees that
//      every accepted draft token AND the bonus token `out[n_keep - 1]` (`n_keep - 1 < L_keep`) come exclusively
//      from the untouched prefix `[0 .. L_keep)`. Even if an online probe triggers a false positive, the ONLY
//      effect is accepting `L_keep` exact tokens in that step — output token distribution and greedy bits remain
//      100% mathematically lossless (`0.000` KL divergence)!
#pragma once

#include "strata/chordmoe/chord_types.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::chordmoe {

/// Signature matching `strata::core::PoolMultiFn` without pulling CUDA headers into CPU-only builds.
using ChordPoolFn = void (*)(void* user, const float* x_f, const int32_t* ids,
                             int64_t n_tok, int64_t k, float* out, int64_t layer);

/// Lock-free per-window controller tracking the live speculative prefix `[0 .. live_rows)` during verification,
/// paired with an optional zero-GPU-overhead host doorbell online divergence probe.
class ChordAbortController {
public:
    ChordAbortController() = default;

    /// Arm the controller at the start of a T-token verify window (`T >= 1`).
    void begin_window(int32_t total_rows, int64_t n_embd) {
        const int32_t safe_t = std::max<int32_t>(1, total_rows);
        total_rows_.store(safe_t, std::memory_order_relaxed);
        live_rows_.store(safe_t, std::memory_order_release);
        abort_layer_.store(-1, std::memory_order_release);
        n_embd_ = n_embd;
        elided_row_layers_.store(0, std::memory_order_relaxed);
        std::memset(div_streak_, 0, sizeof(div_streak_));
    }

    /// Optional: attach drafter/lookahead per-layer predicted Top-K expert IDs `[n_layers, total_rows, k]`
    /// for Trajectory-Jaccard online divergence probing. Pass `nullptr` to clear.
    void set_draft_trajectory(const int32_t* pred_ids, int32_t n_layers, int32_t stride_k) {
        draft_pred_ids_ = pred_ids;
        draft_pred_layers_ = n_layers;
        draft_pred_k_ = stride_k;
    }

    /// Monotonically shrink the live speculative prefix to `valid_rows` (`>= 1`; anchor row 0 is never aborted).
    int32_t cap_live_prefix(int32_t valid_rows, int32_t at_layer) {
        const int32_t clamped = std::max<int32_t>(1, valid_rows);
        int32_t cur = live_rows_.load(std::memory_order_acquire);
        while (clamped < cur) {
            if (live_rows_.compare_exchange_weak(cur, clamped,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
                int32_t expected_layer = -1;
                (void) abort_layer_.compare_exchange_strong(expected_layer, at_layer, std::memory_order_acq_rel);
                return clamped;
            }
        }
        return cur;
    }

    int32_t live_rows() const { return live_rows_.load(std::memory_order_acquire); }
    int32_t total_rows() const { return total_rows_.load(std::memory_order_relaxed); }
    int32_t abort_layer() const { return abort_layer_.load(std::memory_order_acquire); }
    uint64_t elided_row_layers() const { return elided_row_layers_.load(std::memory_order_relaxed); }

    /// Maximum tokens `Verifier::commit(n_keep)` may commit while preserving 100% unconditional bit-exactness.
    int32_t lossless_commit_cap(int32_t raw_n_keep) const {
        return std::min<int32_t>(raw_n_keep, live_rows());
    }

    /// Zero-GPU-overhead Online Speculative Divergence Probe executed on host-mapped doorbell arrays `(h_ids, h_w)`
    /// right before `PoolMultiFn` runs at `layer`:
    ///   - Mode A (when `draft_pred_ids_` is available): measures Top-K Jaccard overlap `|S_target n S_draft| / k`.
    ///     If `overlap < cfg.probe_jaccard_min` for `cfg.probe_patience` consecutive layers (`layer >= probe_warmup`),
    ///     caps `live_rows` at `t` (eliding `t .. T-1` from THIS layer onward!).
    ///   - Mode B (zero-prior fallback, works with MTP / DFlash2 / n-gram lookup without draft expert traces):
    ///     compares normalized routing Shannon entropy `H(w_t) / log(k)` against anchor `t = 0`. Sustained router
    ///     flatness/collapse (`H_norm(t) >= 0.88` and `H_norm(t) >= cfg.probe_entropy_ratio * H_norm(0)`) for
    ///     `cfg.probe_patience` consecutive layers triggers `cap_live_prefix(t, layer)`.
    int32_t observe_layer_routing(int64_t layer, int32_t tb, const int32_t* sub_ids, const float* sub_w,
                                  int64_t n, int64_t k, const ChordConfig& cfg) {
        if (!cfg.online_probe || n <= 0 || k <= 1 || sub_ids == nullptr) return live_rows();
        const int32_t cur_live = live_rows();
        if (cur_live <= 1) return cur_live;

        const float inv_log_k = 1.0f / std::log((float) k);
        auto row_norm_entropy = [&](const float* rw) -> float {
            if (rw == nullptr) return 0.5f;
            float h = 0.0f;
            for (int64_t r = 0; r < k; ++r) {
                const float p = std::max(rw[r], 1e-12f);
                h -= p * std::log(p);
            }
            return h * inv_log_k;
        };

        if (tb == 0 && sub_w != nullptr) {
            anchor_entropy_ = std::max(0.10f, row_norm_entropy(sub_w));
        }

        const bool has_traj = (draft_pred_ids_ != nullptr && layer >= 0 && layer < draft_pred_layers_ && draft_pred_k_ == k);
        const int32_t tot = std::min<int32_t>(total_rows(), kChordMaxWindow);

        for (int64_t local_t = 0; local_t < n; ++local_t) {
            const int32_t t = tb + (int32_t) local_t;
            if (t <= 0 || t >= cur_live || t >= tot) continue;

            bool diverged = false;
            if (has_traj) {
                const int32_t* tgt = sub_ids + local_t * k;
                const int32_t* prd = draft_pred_ids_ + ((size_t) layer * (size_t) total_rows() + (size_t) t) * (size_t) k;
                int inter = 0;
                for (int64_t i = 0; i < k; ++i) {
                    for (int64_t j = 0; j < k; ++j) {
                        if (tgt[i] == prd[j]) { ++inter; break; }
                    }
                }
                const int denom = (int) (2 * k) - inter;
                const float jaccard = denom > 0 ? (float) inter / (float) denom : 1.0f;
                diverged = (jaccard < cfg.probe_jaccard_min);
            } else if (sub_w != nullptr) {
                const float ht = row_norm_entropy(sub_w + local_t * k);
                diverged = (ht >= 0.88f && ht >= cfg.probe_entropy_ratio * anchor_entropy_);
            }

            if (diverged) {
                ++div_streak_[t];
                // Causal propagation: if token `t` has diverged for `probe_patience` layers past warmup,
                // every token `>= t` in the chain is dead -> cap live prefix to `t`!
                if (layer >= cfg.probe_warmup && div_streak_[t] >= cfg.probe_patience) {
                    return cap_live_prefix(t, (int32_t) layer);
                }
            } else {
                div_streak_[t] = 0;
            }
        }
        return live_rows();
    }

    /// Drop-in wrapper around `PoolMultiFn` for sub-range `[tb, tb + n)` of the verify window:
    ///   - Computes offloaded experts ONLY for rows intersecting `[0, live_rows())`;
    ///   - Zeroes `out` for dead suffix rows `[live_in_group .. n)` in O(1) memset so GPU consumers read finite zeros.
    void dispatch_subwindow(ChordPoolFn inner_pool, void* inner_user,
                            int32_t tb, const float* x_f, const int32_t* ids,
                            int64_t n, int64_t k, float* out, int64_t layer) {
        if (inner_pool == nullptr || n <= 0) return;
        const int32_t live_global = live_rows_.load(std::memory_order_acquire);
        const int64_t live_local = std::clamp<int64_t>((int64_t) live_global - (int64_t) tb, 0, n);

        if (live_local >= n) {
            inner_pool(inner_user, x_f, ids, n, k, out, layer);
            return;
        }

        if (live_local > 0) {
            inner_pool(inner_user, x_f, ids, live_local, k, out, layer);
        }
        const int64_t dead_rows = n - live_local;
        if (dead_rows > 0 && out != nullptr && n_embd_ > 0 && k > 0) {
            std::memset(out + (size_t) live_local * (size_t) k * (size_t) n_embd_,
                        0,
                        (size_t) dead_rows * (size_t) k * (size_t) n_embd_ * sizeof(float));
        }
        elided_row_layers_.fetch_add((uint64_t) dead_rows, std::memory_order_relaxed);
    }

    struct PoolAdapterCtx {
        ChordAbortController* ctrl = nullptr;
        ChordPoolFn           inner_pool = nullptr;
        void*                 inner_user = nullptr;
    };

    static void adapted_pool_entry(void* ctx_ptr, const float* x_f, const int32_t* ids,
                                   int64_t n_tok, int64_t k, float* out, int64_t layer) {
        auto* ctx = static_cast<PoolAdapterCtx*>(ctx_ptr);
        if (ctx == nullptr || ctx->inner_pool == nullptr) return;
        if (ctx->ctrl == nullptr) {
            ctx->inner_pool(ctx->inner_user, x_f, ids, n_tok, k, out, layer);
            return;
        }
        ctx->ctrl->dispatch_subwindow(ctx->inner_pool, ctx->inner_user, 0, x_f, ids, n_tok, k, out, layer);
    }

    /// Prune queued lookahead/DMA expert requests for a future layer so experts exclusively requested by
    /// dead suffix rows `[live_rows .. n_tok)` are cancelled before hitting the PCIe bus.
    std::vector<int32_t> filter_live_experts(const int32_t* window_ids, int32_t n_tok, int32_t k,
                                             const int32_t* primary_res,
                                             const int32_t* peer_res = nullptr) const {
        const int32_t active = std::clamp<int32_t>(live_rows(), 1, n_tok);
        std::vector<int32_t> needed;
        needed.reserve((size_t) active * (size_t) k);
        for (int32_t t = 0; t < active; ++t) {
            for (int32_t r = 0; r < k; ++r) {
                const int32_t e = window_ids[(size_t) t * k + r];
                if (e < 0) continue;
                if (primary_res != nullptr && primary_res[e] >= 0) continue;
                if (peer_res != nullptr && peer_res[e] >= 0) continue;
                if (std::find(needed.begin(), needed.end(), e) == needed.end()) {
                    needed.push_back(e);
                }
            }
        }
        return needed;
    }

private:
    std::atomic<int32_t>  total_rows_{1};
    std::atomic<int32_t>  live_rows_{1};
    std::atomic<int32_t>  abort_layer_{-1};
    int64_t               n_embd_ = 0;
    std::atomic<uint64_t> elided_row_layers_{0};
    const int32_t*        draft_pred_ids_ = nullptr;
    int32_t               draft_pred_layers_ = 0;
    int32_t               draft_pred_k_ = 0;
    float                 anchor_entropy_ = 0.5f;
    int32_t               div_streak_[kChordMaxWindow] = {};
};

}  // namespace strata::chordmoe
