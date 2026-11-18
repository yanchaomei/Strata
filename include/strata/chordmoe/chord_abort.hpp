// include/strata/chordmoe/chord_abort.hpp - Graph-Preserving Mid-Window Dead-Suffix Offload Elision.
//
// KEY SYSTEMS INVARIANT:
// In a captured CUDA Graph speculative verify window over positions [p0 .. p0 + T), destroying or halting the
// graph mid-pass would corrupt stream synchronization and force costly graph recaptures (the trap FineMoE hits).
// However, in Strata's doorbell architecture (`Verifier::run` / `Verifier::service`), 80-90% of an offloaded
// window's wall-clock latency sits OUTSIDE the captured graph inside `PoolMultiFn(user, x_f, ids, n_tok, k, out, l)`
// (reading hundreds of MBs of cold experts from DDR5 RAM or streaming them over PCIe).
//
// Because causal attention (QSA) and recurrent DeltaNet (GDN) never allow an earlier position t1 < n_live to read
// activations or K/V states from a future position t2 >= n_live, and `Verifier::commit(n_keep)` commits at most
// `n_keep <= n_live` tokens, eliding ALL offloaded CPU/PCIe expert work for rows `[n_live .. n_tok)` from layer
// `l*` onward (zeroing their `out` rows and raising the doorbell as normal):
//   1. Keeps the captured CUDA Graph 100% intact with zero recapture and zero deadlock;
//   2. Preserves 100% BIT-FOR-BIT mathematical exactness on all committed tokens `[0 .. n_keep)`!
#pragma once

#include "strata/chordmoe/chord_types.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::chordmoe {

/// Signature matching `strata::core::PoolMultiFn` without pulling CUDA headers into CPU-only builds.
using ChordPoolFn = void (*)(void* user, const float* x_f, const int32_t* ids,
                             int64_t n_tok, int64_t k, float* out, int64_t layer);

/// Lock-free per-window controller tracking the live speculative prefix `[0 .. live_rows)` during verification.
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
    }

    /// Monotonically shrink the live speculative prefix to `valid_rows` (`>= 1`; anchor row 0 is never aborted).
    /// Can be signaled from an early-exit probe head, an intermediate router-divergence detector, or host thread.
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

    /// Context struct allowing `ChordAbortController` to wrap any `(PoolMultiFn, user)` pair without changing ABI.
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
    /// dead suffix rows `[live_rows .. n_tok)` are cancelled before hitting the PCIe bus!
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
};

}  // namespace strata::chordmoe
