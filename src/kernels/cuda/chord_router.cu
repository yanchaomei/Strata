// src/kernels/cuda/chord_router.cu - 1-CTA-per-window warp-cooperative Residency-Aware Set-Cover Router.
//
// Unlike `STRATA_ROUTE_RESIDENT` (1 block per token, blind to cross-token sharing and peer GPUs, modifies t=0)
// and `STRATA_ROUTE_PRIOR` (unbounded global logit bonus causing #4402 long-session drift), `chord_route_window_k`
// launches ONE Cooperative Thread Block (`dim3(32, n_tok)`) for the entire T-token verify window, sharing a 64-byte
// `paid_cold_bm[16]` bitmap in L1 shared memory across 3 lightweight waves:
//   - Wave 0: Lock anchor token `t = 0` bit-for-bit and record all cold experts required by `t = 0` and head ranks
//             `0 .. r_lo - 1` into `paid_cold_bm`.
//   - Wave 1: High-survival prefix drafts (`t = 1..2`) skip swaps for any cold tail expert already in `paid_cold_bm`
//             (free ride: zero marginal DDR5/PCIe cost!), swap genuinely unpaid cold tail experts within `eff_margin`
//             subject to the exact register-resident sparse Total Variation gate, and publish remaining cold experts.
//   - Wave 2: Lower-survival tail drafts (`t >= 3`) coalesce onto `(PrimaryVram U PeerVram U paid_cold_bm)`.
#include "strata/kernels/chord_router_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {

template <int K>
__device__ __forceinline__ float device_sparse_tv(const int32_t (&ids_a)[K], const float (&w_a)[K],
                                                  const int32_t (&ids_b)[K], const float (&w_b)[K]) {
    float l1 = 0.0f;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        const int32_t ea = ids_a[i];
        float wb = 0.0f;
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            if (ids_b[j] == ea) wb = w_b[j];
        }
        l1 += fabsf(w_a[i] - wb);
    }
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        const int32_t eb = ids_b[j];
        bool in_a = false;
        #pragma unroll
        for (int i = 0; i < K; ++i) in_a |= (ids_a[i] == eb);
        if (!in_a) l1 += fabsf(w_b[j]);
    }
    return 0.5f * l1;
}

template <int NE, int K>
__global__ void chord_route_window_k(const float* __restrict__ logits,
                                     int32_t* __restrict__ ids,
                                     float* __restrict__ weights,
                                     const int32_t* __restrict__ primary_res,
                                     const int32_t* __restrict__ peer_res,
                                     int n_tok,
                                     float logit_margin,
                                     float tv_budget_token,
                                     float peer_cost,
                                     float paid_union_cost,
                                     float survival_strict,
                                     float default_alpha,
                                     int r_lo,
                                     int r_hi,
                                     int protect_anchor,
                                     unsigned long long* __restrict__ stats) {
    constexpr int kWords = NE / 32;
    __shared__ uint32_t paid_cold_bm[kWords];
    __shared__ uint32_t before_cold_bm[kWords];
    __shared__ uint32_t after_cold_bm[kWords];

    const int lane = (int) threadIdx.x;
    const int t    = (int) threadIdx.y;

    if (t == 0 && lane < kWords) {
        paid_cold_bm[lane]   = 0u;
        before_cold_bm[lane] = 0u;
        after_cold_bm[lane]  = 0u;
    }
    __syncthreads();

    if (t >= n_tok) return;

    const float* row_l = logits + (size_t) t * NE;
    int32_t* row_ids   = ids + (size_t) t * K;
    float* row_w       = weights + (size_t) t * K;

    int32_t orig_ids[K];
    float   orig_w[K];
    int32_t cur_ids[K];
    float   cur_w[K];
    #pragma unroll
    for (int r = 0; r < K; ++r) {
        orig_ids[r] = cur_ids[r] = row_ids[r];
        orig_w[r]   = cur_w[r]   = row_w[r];
    }

    auto is_cold = [&](int e) -> bool {
        if (primary_res[e] >= 0) return false;
        if (peer_res != nullptr && peer_res[e] >= 0) return false;
        return true;
    };

    // Wave 0: census initial cold union + mark anchor `t = 0` and non-swappable head ranks `0 .. r_lo - 1` as paid.
    const bool anchor_row = (t == 0 && protect_anchor != 0);
    int cold_before_row = 0;
    if (lane == 0) {
        for (int r = 0; r < K; ++r) {
            const int e = cur_ids[r];
            if (e >= 0 && e < NE && is_cold(e)) {
                ++cold_before_row;
                atomicOr(&before_cold_bm[e >> 5], 1u << (e & 31));
                if (anchor_row || r < r_lo || r > r_hi) {
                    atomicOr(&paid_cold_bm[e >> 5], 1u << (e & 31));
                }
            }
        }
    }
    __syncthreads();

    float surv = 1.0f;
    for (int j = 0; j < t; ++j) surv *= default_alpha;
    const float eff_margin = logit_margin * (1.0f - survival_strict * surv);

    int free_rides = 0;
    int swaps_done = 0;
    int tv_vetoes  = 0;

    auto run_token_reshape = [&]() {
        if (anchor_row) return;
        for (int r = r_lo; r <= r_hi && r < K; ++r) {
            const int e = cur_ids[r];
            if (!is_cold(e)) continue;
            if (((paid_cold_bm[e >> 5] >> (e & 31)) & 1u) != 0u) {
                if (lane == 0) ++free_rides;
                continue;
            }

            float best_score = -INFINITY;
            int   best_f     = -1;
            for (int f = lane; f < NE; f += 32) {
                const bool f_pri  = (primary_res[f] >= 0);
                const bool f_peer = (!f_pri && peer_res != nullptr && peer_res[f] >= 0);
                const bool f_paid = (!f_pri && !f_peer && (((paid_cold_bm[f >> 5] >> (f & 31)) & 1u) != 0u));
                if (!f_pri && !f_peer && !f_paid) continue;

                if (row_l[e] - row_l[f] > eff_margin) continue;

                bool used = false;
                #pragma unroll
                for (int q = 0; q < K; ++q) used |= (cur_ids[q] == f);
                if (used) continue;

                const float pen = f_pri ? 0.0f : (f_peer ? peer_cost : paid_union_cost);
                const float sc  = row_l[f] - pen;
                if (sc > best_score || (sc == best_score && (best_f < 0 || f < best_f))) {
                    best_score = sc;
                    best_f     = f;
                }
            }

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float osc = __shfl_down_sync(0xffffffffu, best_score, off);
                const int   of  = __shfl_down_sync(0xffffffffu, best_f, off);
                if (osc > best_score || (osc == best_score && of >= 0 && (best_f < 0 || of < best_f))) {
                    best_score = osc;
                    best_f     = of;
                }
            }

            int accept = 0;
            int chosen = cur_ids[r];
            float trial_w[K];
            if (lane == 0 && best_f >= 0) {
                int32_t trial_ids[K];
                #pragma unroll
                for (int q = 0; q < K; ++q) trial_ids[q] = cur_ids[q];
                trial_ids[r] = best_f;

                float mx = -INFINITY;
                #pragma unroll
                for (int q = 0; q < K; ++q) mx = fmaxf(mx, row_l[trial_ids[q]]);
                float sum = 0.0f;
                #pragma unroll
                for (int q = 0; q < K; ++q) {
                    trial_w[q] = expf(row_l[trial_ids[q]] - mx);
                    sum += trial_w[q];
                }
                const float inv = 1.0f / fmaxf(sum, 1e-30f);
                #pragma unroll
                for (int q = 0; q < K; ++q) trial_w[q] *= inv;

                const float tv = device_sparse_tv<K>(orig_ids, orig_w, trial_ids, trial_w);
                if (tv <= tv_budget_token + 1e-7f) {
                    accept = 1;
                    chosen = best_f;
                    ++swaps_done;
                } else {
                    ++tv_vetoes;
                }
            }

            accept = __shfl_sync(0xffffffffu, accept, 0);
            chosen = __shfl_sync(0xffffffffu, chosen, 0);
            if (accept) {
                cur_ids[r] = chosen;
                #pragma unroll
                for (int q = 0; q < K; ++q) {
                    cur_w[q] = __shfl_sync(0xffffffffu, trial_w[q], 0);
                }
            }
        }

        if (lane == 0) {
            for (int r = 0; r < K; ++r) {
                const int e = cur_ids[r];
                if (e >= 0 && e < NE && is_cold(e)) {
                    atomicOr(&paid_cold_bm[e >> 5], 1u << (e & 31));
                }
            }
        }
    };

    // Wave 1: High-survival prefix drafts (t = 1..2).
    if (t >= 1 && t <= 2) run_token_reshape();
    __syncthreads();

    // Wave 2: Lower-survival tail drafts (t >= 3) coalesce onto Wave 0 + Wave 1's union!
    if (t >= 3 || (t == 0 && !anchor_row)) run_token_reshape();
    __syncthreads();

    if (lane == 0) {
        int cold_after_row = 0;
        for (int r = 0; r < K; ++r) {
            row_ids[r] = cur_ids[r];
            row_w[r]   = cur_w[r];
            if (is_cold(cur_ids[r])) {
                ++cold_after_row;
                atomicOr(&after_cold_bm[cur_ids[r] >> 5], 1u << (cur_ids[r] & 31));
            }
        }
        if (stats != nullptr) {
            atomicAdd(stats + 2, (unsigned long long) cold_before_row);
            atomicAdd(stats + 3, (unsigned long long) cold_after_row);
            atomicAdd(stats + 4, (unsigned long long) free_rides);
            atomicAdd(stats + 5, (unsigned long long) swaps_done);
            atomicAdd(stats + 6, (unsigned long long) tv_vetoes);
        }
    }
    __syncthreads();

    if (stats != nullptr && t == 0 && lane < kWords) {
        const unsigned int ub = __popc(before_cold_bm[lane]);
        const unsigned int ua = __popc(after_cold_bm[lane]);
        if (ub) atomicAdd(stats + 0, (unsigned long long) ub);
        if (ua) atomicAdd(stats + 1, (unsigned long long) ua);
    }
}

}  // namespace

bool chord_router_supported(int n_expert, int k) {
    return (n_expert == 512 && k == 10) || (n_expert == 256 && k == 8);
}

void chord_route_window(const float* logits, int32_t* ids, float* weights,
                        const int32_t* primary_res, const int32_t* peer_res,
                        int n_tok, int n_expert, int k,
                        const strata::chordmoe::ChordConfig& cfg,
                        unsigned long long* stats, void* stream) {
    if (cfg.mode != strata::chordmoe::ChordMode::BoundedSetCover || n_tok <= 0) return;
    if (!stream || !logits || !ids || !weights || !primary_res || !chord_router_supported(n_expert, k)) {
        throw std::invalid_argument("chord_route_window: invalid buffers or unsupported (n_expert, k)");
    }
    // Single CTA spans up to 32 warps (`n_tok <= 32`, `32 * 32 = 1024` threads max per block).
    const int capped_tok = std::min(n_tok, 32);
    const int r_lo = std::clamp<int>(cfg.rank_lo, 0, k);
    const int r_hi = std::min<int>(cfg.rank_hi, k - 1);
    const dim3 block(32, (unsigned) capped_tok);

    if (n_expert == 512 && k == 10) {
        chord_route_window_k<512, 10><<<1, block, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, ids, weights, primary_res, peer_res, capped_tok,
            cfg.logit_margin, cfg.tv_budget_token, cfg.peer_tier_cost, cfg.paid_union_cost,
            cfg.survival_strict, cfg.default_alpha, r_lo, r_hi, cfg.protect_anchor ? 1 : 0, stats);
    } else {
        chord_route_window_k<256, 8><<<1, block, 0, static_cast<cudaStream_t>(stream)>>>(
            logits, ids, weights, primary_res, peer_res, capped_tok,
            cfg.logit_margin, cfg.tv_budget_token, cfg.peer_tier_cost, cfg.paid_union_cost,
            cfg.survival_strict, cfg.default_alpha, r_lo, r_hi, cfg.protect_anchor ? 1 : 0, stats);
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

}  // namespace strata::kernels
