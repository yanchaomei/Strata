# ChordMoE Portability Adapter (`llama.cpp` / `S²-MoE` & Server Evaluation Guide)

This directory contains the zero-dependency `ggml` / `llama.cpp` portability bridge ([ggml_chord_moe.hpp](ggml_chord_moe.hpp)) and patch ([s2moe_chordmoe.patch](s2moe_chordmoe.patch)) pairing with the core `ChordMoE` headers in `include/strata/chordmoe/`.

## 1. Environment Knobs (Opt-In; Default Off)

### Strata Main Engine & All-Cards Split Engine
| Variable | Default | Description |
| :--- | :--- | :--- |
| `STRATA_CHORD_MODE` | `-1` (Off) | `0`: 100% lossless prefetch & dead-suffix elision; `1`: error-bounded window union set-cover router |
| `STRATA_CHORD_MARGIN` | `1.25` | Max logit gap `l[e] - l[f]` permitted for tail expert coalescing (`Mode 1`) |
| `STRATA_CHORD_TV_BUDGET` | `0.12` | Hard per-(layer, token) sparse Total Variation ceiling guaranteeing `\|\|y - y~\|\|_2 <= 2 M * TV` |
| `STRATA_CHORD_TURN_BUDGET` | `2.40` | Exponentially-decayed cumulative TV budget across layers |
| `STRATA_CHORD_RANKS` | `4-31` | 0-indexed Top-K rank window eligible for Mode 1 reshaping (`0..lo-1` locked exact) |
| `STRATA_CHORD_EARLY_ABORT` | `0` | `1`: Elide offloaded CPU/PCIe expert reads for dead speculative suffix rows without recapturing CUDA Graphs |
| `STRATA_CHORD_ONLINE_PROBE` | `0` | `1`: Enable zero-GPU-cost doorbell divergence probe (Trajectory-Jaccard / Anchor-Entropy-Spike) |
| `STRATA_CHORD_SPLIT_OFFLOAD` | `0` | `1`: Unlock `FNX_EXPERT_RAM_GIB` hybrid VRAM + host-RAM offloading in `strata-split` on 2–4× 3090/4090/A6000 |

### `llama.cpp` / `S²-MoE` Adapter (`s2moe_chordmoe.patch`)
| Variable | Default | Description |
| :--- | :--- | :--- |
| `CHORDMOE_ENABLE` | `0` | `1`: Intercept `ggml_compute_forward_moe_reuse_two_pass` with `ChordMoE` selection-logit reshaping |
| `CHORDMOE_VRAM_RATIO` | `0.32` | Fraction of experts resident in VRAM tier for controlled iso-budget evaluation |
| `S2MOE_AUDIT_STATS` | `0` | `1`: Measure anchor corruption, cold misses/layer, and max TV on stock `S²-MoE` |

## 2. Linux Multi-GPU Server Quickstart

```bash
git clone -b feat/chordmoe-proto https://github.com/yanchaomei/Strata.git
cd Strata
chmod +x scripts/chordmoe_server_suite.sh

# 1. Build Strata CUDA + GPU parity tests + patched S2-MoE CUDA
./scripts/chordmoe_server_suite.sh build

# 2. Verify CPU unit tests + live NVIDIA GPU CTA parity & latency
./scripts/chordmoe_server_suite.sh parity

# 3. Run Strata end-to-end ablation & S2-MoE controlled GGUF benchmark
MODEL_GGUF=/path/to/Qwen3.8-or-Qwen3.6.gguf ./scripts/chordmoe_server_suite.sh bench-strata
S2_MODEL_GGUF=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf ./scripts/chordmoe_server_suite.sh bench-s2moe
```
