# Strata on Huawei Ascend NPU (CANN 9.1 / Ascend C)

`ascend/` is Strata's standalone CMake tree for Huawei Ascend inference cards (`Ascend 310P3` / `dav-m200`), built the
same way `metal/` and `sycl/` sit beside the CUDA and HIP trees without touching any default source path.

## Architecture

- **280-TOPS `INT8 FRACTAL_NZ` Cube Engine (`aclnnQuantGroupedMatmulDequantWeightNZ`)**: A two-pass loader packs large
  dense projections (fused GDN input `[2560, 16480]` and `ssm_out`, fused QSA input `[2560, 13952]` and `attn_output`,
  and the column-chunked `248,320`-row LM head) plus `100%` (`48/48`) of the `E = 512` routed MoE expert layers onto
  three `48 GB` `Ascend 310P3` cards in `INT8 FRACTAL_NZ` (`[E, K/32, N/16, 16, 32]`, dynamic per-token activation
  quantization, FP32 per-channel scales), leaving room on NPU 1 for the `946 MiB` native `DFlash2` block drafter. Small
  hyper-connection, router, shared-expert, and layer-1 PLE projections run in FP16 `FRACTAL_NZ` (`aclnnMm`). Any tail
  MoE layers on smaller HBM configs fall back automatically to 8-core vectorized planar `Q2_0`.
- **Native `DFlash2` Block Speculative Decoding (`Dflash2Drafter`)**: Five tapped target hidden states (`layers 5, 15,
  25, 35, 45`) feed a 5-layer bidirectional dynamic-convolution transformer (`2,048` sliding KV ring + 8-row block
  Flash-Block online softmax attention) and a column-chunked `106,299`-token `INT8 FRACTAL_NZ` Cube draft head on NPU 1,
  paired with `Q4_0` selector codebooks in Host DDR4 RAM for exact IEEE FP32 left-to-right candidate selection and
  multi-turn row-log lineage rebuilds. Non-drafter NPUs overlap asynchronous GDN state folding (`commit`) in parallel
  with `DFlash2` drafting on NPU 1.
- **8 AI Cores (`ascend/src/kernels/ascend_ops.asc`, `dav-m200`)**: Custom Ascend C kernels implement hyper-connection
  RMSNorm (with fused FP16 residual cast) and 4-stream vectorized `PIPE_V` gating, 16-way `Axpy` ILP 48-head GDN
  conv1d + L2 norm + `128x128` delta-rule recurrence, QSA RoPE + `INT8` KV quantization + shared-KV multi-token
  strided DMA + batched `PIPE_V` `Exp` Flash-Block online softmax attention (+ 4-pass radix-256 top-k past `2,051`
  cells), 16-bucket max-prefiltered `PIPE_V` MoE routing + SwiGLU + combine, and 64-lane `PIPE_V` max-prefiltered FP16
  vocabulary argmax / top-16.
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
   3, 1, 2, 4)` byte-for-byte. On `dav-m200`, `N` per call must stay `<= 65,536` (`N * sizeof(half) <= 131,072`); wider
   vocabulary heads (`248,320` or `106,299`) split into `<= 62,080`-column chunks (`Int8ChunkedMat`).

## Building

Inside a CANN `9.1.0` container (for example `quay.io/ascend/vllm-ascend:v0.23.0-310p`):

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S ascend -B build-ascend -DCMAKE_BUILD_TYPE=Release
cmake --build build-ascend --target ascend_parity_test strata -j32
./build-ascend/ascend_parity_test
```

## Measured Numbers (`3x Huawei Atlas 300V Pro / Ascend 310P3`, Kunpeng 920)

Measured on Huawei TaiShan 200 2280 (`2x Kunpeng 920 5220`, 64 cores @ 2.6 GHz, 376 GiB DDR4-2933) with three `48 GB`
`Atlas 300V Pro` cards (`Ascend 310P3`, driver `26.0.rc1`, CANN `9.1.0`) running
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`Q2_0`, `--max-context 131072`, `--ple-io ram`) and
`qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack` (`--draft dflash2 --draft-fallback none --spec 6 --draft-cap 6
--draft-conf 0.55`):

- **Operator parity (`ascend_parity_test`, `9/9` suites on `Ascend 310P3` vs FP32 CPU reference)**:
  - `f32 <-> f16`: `max_abs = 0.00e+00`
  - `aclnnMm FRACTAL_NZ`: `max_abs = 9.75e-04`, `rmse = 1.75e-04`
  - `gr_norm / post_down / post_up`: `xn = 2.86e-06`, `out = 5.96e-07`
  - `gdn_step` (verify + fold, 16-way `Axpy` ILP): `yg = 2.44e-04`, `state = 1.12e-08`
  - `qsa` (prep + `PIPE_V` radix-256 select + batched `PIPE_V` Flash-Block `INT8` attention): `xq = 1.22e-04`, `radix_2051 = EXACT`
  - `moe` (`route + q2_gu + q2_down + combine`): `max_abs = 5.85e-07`, `rmse = 1.44e-07`
  - `argmax_f16` (`[4, 248320]`): `EXACT`
  - `moe_int8_cube` (`T = 16, M = 160` vs FP32 `Q2_0` reference): `cos_sim = 0.999997`, `rmse = 6.57e-03`
  - `dflash2` (`tap + dyn_conv + attn + swiglu + topk16`): `tap = 1.19e-07`, `conv = 5.96e-08`, `attn = 1.53e-05`, `topk16 = EXACT`
- **HBM & Host RAM footprint (`max_context = 131072`, `48/48` layers in `INT8 FRACTAL_NZ` Cube + `DFlash2`)**:
  - NPU 0 (layers `0..15` + PLE, `16` INT8 Cube): `41.71 / 43.18 GiB` (`96.6%`)
  - NPU 1 (layers `16..31`, `16` INT8 Cube + `946 MiB` `DFlash2` + `106,299`-row INT8 draft head): `42.57 / 43.18 GiB` (`98.6%`)
  - NPU 2 (layers `32..47` + chunked INT8 LM head, `16` INT8 Cube): `42.26 / 43.18 GiB` (`97.9%`)
  - Host DDR4 RAM: `/dev/shm` `FRACTAL_NZ` hot cache + `26.8 GiB` PLE table + `Q4_0` selector codebooks + `32 GiB`
    conversation checkpoint budget (`0 B` swap)
- **End-to-end single-stream serving (`serve/server.py`, warm engine, SSE streaming)**:
  - Warm `/dev/shm` reload (`48/48` INT8 Cube + `128K` states + `DFlash2`): `65.3 s` (`7.9 s` dense Pass 1, `0.6 s` `DFlash2`)
  - Non-speculative single-token decode (`T = 1`, `48/48` INT8 Cube): **`11.69 - 11.76 tok/s`** (`85.1 - 85.6 ms/token`,
    up from `8.5 - 8.9 tok/s` prior FP16-dense/`45/48` build and `0.47 tok/s` initial bring-up)
  - Verify window latency (`T = 2..6`): **`84 - 112 ms/window`** (`DFlash2` proposal: `11.4 - 16.0 ms/round`)
  - Native `DFlash2` speculative decode (`--draft dflash2 --spec 6 --draft-conf 0.55`):
    - C++20 code generation (`512` tokens, greedy): **`28.93 tok/s`** (`34.57 ms/token`, `26.39 tok/s` end-to-end incl. TTFT)
    - Python 3.12 `asyncio` crawler (`512` tokens, greedy): **`26.24 tok/s`** (`38.11 ms/token`, `24.07 tok/s` end-to-end)
    - Mathematical derivation (`Cooley-Tukey FFT`, `448` tokens, greedy): **`21.58 - 22.00 tok/s`** (`45.5 - 46.3 ms/token`)
    - Chinese systems technical prose (`448` tokens, greedy & `temperature = 0.6`): **`16.60 - 16.70 tok/s`** (`59.9 - 60.2 ms/token`)
  - Warm prompt prefill (`256`-token chunks): **`33.1 - 44.3 tok/s`** (`158 ms` TTFT on multi-turn checkpoint hit)
