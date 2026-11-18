// include/strata/kernels/chord_router_cuda.hpp - CUDA/HIP window-level residency-aware set-cover router.
//
// Strictly opt-in (`STRATA_CHORD_MODE=1`); default engine path never invokes this kernel.
#pragma once

#include "strata/chordmoe/chord_types.hpp"

#include <cstdint>

namespace strata::kernels {

/// True when the compiled CUDA/HIP kernel supports `(n_expert, k)` (`512x10` Flash-Next and `256x8` Qwen3.6-35B).
bool chord_router_supported(int n_expert, int k);

/// Reshape `[n_tok, k]` expert selections and weights in-place for one verify window on `stream`.
///   - `logits`:      device `[n_tok, n_expert]` raw F32 router logits
///   - `ids`:         device `[n_tok, k]` in/out Top-K expert IDs
///   - `weights`:     device `[n_tok, k]` in/out renormalized Top-K weights
///   - `primary_res`: device `[n_expert]` primary GPU0 residency table (`>= 0` resident, `< 0` cold)
///   - `peer_res`:    optional device `[n_expert]` helper GPU union residency (`>= 0` peer-resident, `nullptr` if none)
///   - `stats`:       optional device `uint64_t[8]` telemetry counters (`nullptr` disables counting)
void chord_route_window(const float* logits, int32_t* ids, float* weights,
                        const int32_t* primary_res, const int32_t* peer_res,
                        int n_tok, int n_expert, int k,
                        const strata::chordmoe::ChordConfig& cfg,
                        unsigned long long* stats, void* stream);

}  // namespace strata::kernels
