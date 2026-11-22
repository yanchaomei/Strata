// include/strata/chordmoe/chord_types.hpp - ChordMoE core configuration, residency tiers, and telemetry.
//
// Zero external dependencies (pure C++17/20), usable from Strata's CPU checks, Strata's CUDA/HIP engine,
// and the standalone llama.cpp / ggml portability adapter.
//
// Default policy (per AGENTS.md): ChordMoE is strictly OPT-IN (`ChordMode::Off` when `STRATA_CHORD_MODE` is
// unset or `< 0`), keeping the default engine path byte-for-byte identical to baseline Strata.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::chordmoe {

/// Operating mode of ChordMoE.
enum class ChordMode : int32_t {
    Off              = -1, ///< Disabled (default): zero overhead, bit-for-bit baseline behavior.
    Lossless         = 0,  ///< Mode 0: 100% mathematically bit-exact (survival-weighted prefetch + mid-window dead-suffix offload elision).
    BoundedSetCover  = 1,  ///< Mode 1: Error-bounded multi-tier window union set-cover routing + anchor t=0 protection + sparse TV gate.
};

/// Multi-tier memory residency classification for an expert (layer, e).
enum class ExpertTier : uint8_t {
    PrimaryVram = 0, ///< Resident in GPU0 primary ExpertCache (zero miss latency, zero PCIe weight transfer).
    PeerVram    = 1, ///< Resident in a helper GPU's RemoteExperts/PeerExperts cache (computed via portable host sum, no weight DMA).
    Shadow2Bit  = 2, ///< Resident as a 2-bit nibble / VNNI shadow representation (usable only for tiny tail weights under Mode 1).
    ColdHost    = 3, ///< Cold in host DDR5 RAM / NVMe (requires CPU pool GEMM or async PCIe H2D DMA).
};

inline constexpr int kChordMaxExperts = 1024;
inline constexpr int kChordMaxTopK    = 32;
inline constexpr int kChordMaxWindow  = 32;

/// Runtime configuration for ChordMoE, parsed once from environment or constructed explicitly in tests.
struct ChordConfig {
    ChordMode mode             = ChordMode::Off;
    float     logit_margin     = 1.25f;  ///< Max logit drop l[e] - l[f] permitted for a draft tail swap (Mode 1).
    float     tv_budget_token  = 0.12f;  ///< Hard per-(layer, token) sparse Total Variation ceiling: ||y - y~||_2 <= 2*M*tv.
    float     tv_budget_turn   = 2.40f;  ///< Cumulative exponentially-decayed TV budget across a full multi-layer forward pass.
    float     residual_decay   = 0.97f;  ///< Per-layer residual propagation decay factor gamma_res in the turn ledger.
    float     peer_tier_cost   = 0.08f;  ///< Small logit tie-breaker penalty for PeerVram vs PrimaryVram (both avoid DDR5/PCIe weight reads).
    float     paid_union_cost  = 0.04f;  ///< Small logit tie-breaker for coalescing onto an already-paid cold expert in the window union.
    float     survival_strict  = 0.35f;  ///< Tighten margin on high-survival prefix drafts: eff_margin = margin * (1 - survival_strict * p_alive).
    float     default_alpha    = 0.78f;  ///< Default per-step draft acceptance prior when explicit drafter q-probs are unavailable.
    float     prefetch_min_ev  = 0.22f;  ///< Minimum expected survival probability required before issuing an async prefetch (Mode 0 & 1).
    int32_t   rank_lo          = 4;      ///< First Top-K rank eligible for Mode 1 set-cover replacement (0-indexed; head ranks 0..rank_lo-1 stay exact).
    int32_t   rank_hi          = 31;     ///< Last Top-K rank eligible (clamped to k - 1).
    bool      protect_anchor   = true;   ///< Never alter token t = 0 (the verified/accepted anchor token).
    bool      enable_free_ride = true;   ///< Wave 1: keep cold tail experts already in paid_cold_bm at zero distortion.
    bool      enable_paid_coalesce = true; ///< Wave 2 Tier-3: allow coalescing onto cold experts already in paid_cold_bm.
    bool      early_abort      = false;  ///< Enable graph-preserving mid-window dead-suffix offload elision (`STRATA_CHORD_EARLY_ABORT=1`).
    bool      online_probe     = false;  ///< Enable zero-GPU-cost doorbell online divergence probe (`STRATA_CHORD_ONLINE_PROBE=1`).
    int32_t   probe_warmup     = 6;      ///< Minimum layer index before the online divergence probe may cap the suffix.
    int32_t   probe_patience   = 3;      ///< Consecutive divergent layers required before capping `live_rows` at draft `t`.
    float     probe_jaccard_min = 0.20f; ///< Trajectory Top-K Jaccard floor when draft expert predictions are provided.
    float     probe_entropy_ratio = 1.65f; ///< Draft-vs-anchor normalized router entropy spike ratio triggering streak increment.
    bool      split_offload    = false;  ///< Allow all-cards + multi-NUMA heterogeneous offload scheduling (`STRATA_CHORD_SPLIT_OFFLOAD=1`).

    static ChordConfig from_env() {
        ChordConfig cfg;
        if (const char* v = std::getenv("STRATA_CHORD_MODE")) {
            if (v[0] != '\0') {
                const int m = std::atoi(v);
                if (m == 0) cfg.mode = ChordMode::Lossless;
                else if (m >= 1) cfg.mode = ChordMode::BoundedSetCover;
            }
        }
        if (const char* v = std::getenv("STRATA_CHORD_MARGIN")) {
            if (v[0] != '\0') cfg.logit_margin = std::max(0.0f, (float) std::atof(v));
        }
        if (const char* v = std::getenv("STRATA_CHORD_TV_BUDGET")) {
            if (v[0] != '\0') cfg.tv_budget_token = std::clamp((float) std::atof(v), 0.0f, 1.0f);
        }
        if (const char* v = std::getenv("STRATA_CHORD_TURN_BUDGET")) {
            if (v[0] != '\0') cfg.tv_budget_turn = std::max(0.0f, (float) std::atof(v));
        }
        if (const char* v = std::getenv("STRATA_CHORD_RANKS")) {
            int lo = cfg.rank_lo, hi = cfg.rank_hi;
            if (std::sscanf(v, "%d-%d", &lo, &hi) == 2 && lo >= 0 && hi >= lo) {
                cfg.rank_lo = lo;
                cfg.rank_hi = hi;
            }
        }
        if (const char* v = std::getenv("STRATA_CHORD_PROTECT_ANCHOR")) {
            if (v[0] != '\0') cfg.protect_anchor = (std::atoi(v) != 0);
        }
        if (const char* v = std::getenv("STRATA_CHORD_EARLY_ABORT")) {
            if (v[0] != '\0') cfg.early_abort = (std::atoi(v) != 0);
        }
        if (const char* v = std::getenv("STRATA_CHORD_ONLINE_PROBE")) {
            if (v[0] != '\0') {
                cfg.online_probe = (std::atoi(v) != 0);
                if (cfg.online_probe) cfg.early_abort = true;
            }
        }
        if (const char* v = std::getenv("STRATA_CHORD_SPLIT_OFFLOAD")) {
            if (v[0] != '\0') cfg.split_offload = (std::atoi(v) != 0);
        }
        if (const char* v = std::getenv("STRATA_CHORD_PREFETCH_MIN")) {
            if (v[0] != '\0') cfg.prefetch_min_ev = std::clamp((float) std::atof(v), 0.0f, 1.0f);
        }
        return cfg;
    }

    bool enabled() const { return mode != ChordMode::Off || early_abort || split_offload; }
};

/// Per-window / cumulative telemetry for diagnostics and paper evaluation tables.
struct ChordWindowStats {
    uint64_t windows_seen          = 0;
    uint64_t anchor_swaps          = 0; ///< Must remain strictly 0 whenever protect_anchor == true.
    uint64_t unique_cold_before    = 0; ///< Distinct cold experts in window union before ChordMoE reshaping.
    uint64_t unique_cold_after     = 0; ///< Distinct cold experts in window union after ChordMoE reshaping.
    uint64_t cold_entries_before   = 0; ///< Total (token, rank) cold entries before reshaping.
    uint64_t cold_entries_after    = 0; ///< Total (token, rank) cold entries after reshaping.
    uint64_t free_ride_kept        = 0; ///< Cold tail entries kept untouched because already paid in window union (0 marginal cost).
    uint64_t swaps_to_primary      = 0; ///< Cold tail entries redirected to PrimaryVram.
    uint64_t swaps_to_peer         = 0; ///< Cold tail entries redirected to PeerVram (helper GPU).
    uint64_t swaps_to_paid_union   = 0; ///< Cold tail entries coalesced onto an already-activated cold expert in the window.
    uint64_t swaps_rejected_tv     = 0; ///< Candidate swaps vetoed by the per-token or per-turn sparse TV credit gate.
    uint64_t prefetches_kept       = 0; ///< Survival-weighted prefetches admitted.
    uint64_t prefetches_dropped    = 0; ///< Doomed-tail or duplicate prefetches suppressed.
    uint64_t dead_suffix_rows      = 0; ///< Speculative rows whose offloaded expert GEMMs were elided mid-window.
    double   max_token_tv          = 0.0;
};

}  // namespace strata::chordmoe
