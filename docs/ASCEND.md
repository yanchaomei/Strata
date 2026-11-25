# Strata on Huawei Ascend NPU (CANN 9.1 / Ascend C)

`ascend/` is Strata's standalone CMake tree for Huawei Ascend inference cards (`Ascend 310P3` / `dav-m200`), built the
same way `metal/` and `sycl/` sit beside the CUDA and HIP trees without touching any default source path.

## Architecture

- **280-TOPS `INT8 FRACTAL_NZ` Grouped Cube MoE (`aclnnQuantGroupedMatmulDequantWeightNZ`)**: At startup, a two-pass
  loader allocates all dense matrices, `128K` KV/GDN states, and the output head first, then automatically fills up to
  `~98%` of every visible NPU's HBM with `INT8 FRACTAL_NZ` (`[E, K/32, N/16, 16, 32]`) routed expert weights (`E = 512`,
  `top_k = 10`, dynamic per-token activation quantization, FP32 per-channel scales). Any remaining tail layers fall back
  cleanly to 8-core vectorized planar `Q2_0`. Full `256`-token prompt chunks (`M = 2,560` routed rows) run in one
  grouped Cube pass per layer.
- **FP16 `FRACTAL_NZ` Dense Cube Projections (`aclnnMm`)**: Hyper-connection down/inject/up matrices, fused GDN input
  `[2560, 16480]` and `ssm_out`, fused QSA input `[2560, 13952]` and `attn_output`, MoE router + shared expert
  projections, layer-1 PLE `key`/`value`, and the `248,320`-row output head run on the DaVinci FP16 Cube Cores.
- **8 AI Cores (`ascend/src/kernels/ascend_ops.asc`, `dav-m200`)**: Custom Ascend C kernels implement hyper-connection
  RMSNorm (with fused FP16 residual cast) and `PIPE_V` gating, 4-way ILP 48-head GDN conv1d + L2 norm + `128x128`
  delta-rule recurrence, QSA RoPE + `INT8` KV quantization + `PIPE_V` block scoring + exact 4-pass radix-256 top-k
  (`2,051` budget) + 16-cell Flash-Block online softmax attention, 8-core `PIPE_V` MoE routing + SwiGLU + combine,
  planar `Q2_0` fallback GEMVs, and 8-core FP16 vocabulary argmax.
- **Multi-NPU Pipeline (`1..8` NPUs) & Host RAM Hot Tier**: Layers split evenly across all visible NPUs
  (`STRATA_ASCEND_SINGLE_NPU=1` or `STRATA_ASCEND_MAX_NPUS=N` overrides), passing `~50 KB` per token across stages via
  DMA-pinned host buffers (`aclrtMallocHost`) on NUMA-local CPUs. Pre-packed `FRACTAL_NZ` weights and FP32 token
  embeddings cache in `/dev/shm/chaomei_strata_nz` (`STRATA_ASCEND_CACHE_DIR=off` disables), alongside `--ple-io ram`
  for the `26.8 GiB` PLE n-gram table.

### `dav-m200` (`Ascend 310P3`) ISA Notes

Verified on `Ascend 310P3` with CANN `9.1.0` (`bisheng` clang 15.0.5):

1. `AscendC::Cast(half, float, ...)` requires `AscendC::RoundMode::CAST_NONE`; `CAST_RINT` is a silent no-op on
   `dav-m200`.
2. `AscendC::ShiftRight` on `uint16_t` is a silent no-op on `dav-m200`. Planar `Q2_0` unpacking masks the four 2-bit
   planes with `{0x0303, 0x0C0C, 0x3030, 0xC0C0}`, casts `uint8_t -> half -> float`, adds `{-1, -4, -16, -64}`, and
   pre-scales deinterleaved activations by `{1, 1/4, 1/16, 1/64}` before `BlockReduceSum`.
3. `aclnnQuantGroupedMatmulDequantWeightNZ` expects `ACL_FORMAT_FRACTAL_NZ` of logical `[E, N, K]` (`transposeWeight =
   true`), whose physical storage layout `[E, K/32, N/16, 16, 32]` matches CPU `view(E, N/16, 16, K/32, 32).permute(0,
   3, 1, 2, 4)` byte-for-byte.

## Building

Inside a CANN `9.1.0` container (for example `quay.io/ascend/vllm-ascend:v0.23.0-310p`):

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S ascend -B build-ascend -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build-ascend ascend_parity_test strata
./build-ascend/ascend_parity_test
```

## Measured Numbers (`3x Huawei Atlas 300V Pro / Ascend 310P3`, Kunpeng 920)

Measured on Huawei TaiShan 200 2280 (`2x Kunpeng 920 5220`, 64 cores @ 2.6 GHz, 376 GiB DDR4-2933) with three `48 GB`
`Atlas 300V Pro` cards (`Ascend 310P3`, driver `26.0.rc1`, CANN `9.1.0`) running
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`Q2_0`, `--max-context 131072`, `--ple-io ram`, no MTP drafter):

- **Operator parity (`ascend_parity_test` on `Ascend 310P3` vs FP32 CPU reference)**:
  - `f32 <-> f16`: `max_abs = 0.00e+00`
  - `aclnnMm FRACTAL_NZ`: `max_abs = 9.75e-04`, `rmse = 1.75e-04`
  - `gr_norm / post_down / post_up`: `xn = 2.86e-06`, `out = 5.96e-07`
  - `gdn_step` (verify + fold): `yg = 2.44e-04`, `state = 1.12e-08`
  - `qsa` (prep + `PIPE_V` radix-256 select + Flash-Block `INT8` online attention): `xq = 1.22e-04`, `radix_2051 = EXACT`
  - `moe` (`route + q2_gu + q2_down + combine`): `max_abs = 5.85e-07`, `rmse = 1.44e-07`
  - `moe_int8_cube` (`T = 16, M = 160` vs FP32 `Q2_0` reference): `cos_sim = 0.999997`, `rmse = 6.57e-03`
- **HBM & Host RAM footprint (`max_context = 131072`, `45/48` layers in `INT8 FRACTAL_NZ` Cube)**:
  - NPU 0 (layers `0..15` + PLE, `15` INT8 Cube + `1` `Q2_0`): `42,338 / 44,213 MB` (`95.8%`)
  - NPU 1 (layers `16..31`, `15` INT8 Cube + `1` `Q2_0`): `42,260 / 44,213 MB` (`95.6%`)
  - NPU 2 (layers `32..47` + output head, `15` INT8 Cube + `1` `Q2_0`): `43,506 / 44,213 MB` (`98.4%`)
  - Host DDR4 RAM: `210 / 376 GiB` active (`119 GiB` `/dev/shm` `FRACTAL_NZ` hot cache + `26.8 GiB` locked PLE table +
    `32 GiB` conversation checkpoint budget, `0 B` swap)
- **End-to-end serving (`serve/server.py`, warm engine)**:
  - Warm `/dev/shm` reload (`128.1 GB` across 3 NPUs + `128K` states): `65.2 s` (`9.1 s` dense Pass 1)
  - Short-context decode (`T = 1`, greedy, sampled, and SSE streaming): **`8.5 - 8.9 tok/s`** (`112.8 - 117.2 ms/token`,
    up from `4.2 tok/s` on 2-NPU `Q2_0` and `0.47 tok/s` initial bring-up)
  - Long-context decode (`2,266 - 4,254` tokens, sparse QSA `2,051` budget): **`7.0 - 7.1 tok/s`** (`140.4 - 143.5 ms/token`)
  - Warm prompt prefill (`93 - 4,227` tokens, `256`-token chunks): **`28.7 - 39.5 tok/s`** (up from `5.1 - 6.0 tok/s`),
    with multi-turn checkpoint reuse (`2,265` cached tokens)
