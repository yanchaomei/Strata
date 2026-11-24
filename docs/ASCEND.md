# Strata on Huawei Ascend NPU (CANN 9.1 / Ascend C)

`ascend/` is Strata's standalone CMake tree for Huawei Ascend inference cards (`Ascend 310P3` / `dav-m200`), built the
same way `metal/` and `sycl/` sit beside the CUDA and HIP trees without touching any default source path.

## Architecture

- **Cube Cores (`aclnnMm` + `FRACTAL_NZ`)**: Every dense projection in `Qwen3.8-Flash-Next` (hyper-connection down,
  inject, and up matrices, fused GDN input `[2560, 16480]` and `ssm_out`, fused QSA input `[2560, 13952]` and
  `attn_output`, MoE router + shared expert projections, layer-1 PLE `key`/`value`, and the `248,320`-row output head)
  is dequantized once at startup with a `32x32` L1-tiled transpose and converted in-place to `ACL_FORMAT_FRACTAL_NZ` via
  `aclnnTransMatmulWeight`.
- **8 AI Cores (`ascend/src/kernels/ascend_ops.asc`, `dav-m200`)**: Custom Ascend C kernels implement FP32/FP16 casts,
  hyper-connection RMSNorm and vectorized `PIPE_V` gating, fused 48-head GDN conv1d + L2 norm + `128x128` delta-rule
  recurrence (both verify and fold passes), QSA RoPE + `INT8` KV quantization + exact 4-pass radix-256 block top-k
  (`2,051` budget) + online softmax attention, MoE top-10 routing, planar `Q2_0` bit-plane unpacking + `PIPE_V`
  `BlockReduceSum` expert GEMVs, and 8-core FP16 vocabulary argmax.
- **Automatic 2-NPU Pipeline Split**: When two or more Ascend NPUs are visible, NPU 0 holds PLE + layers `0..23` and
  NPU 1 holds layers `24..47` + `output_hc` + `output.weight` (`STRATA_ASCEND_SINGLE_NPU=1` forces 1 NPU). Only the
  residual stream (`~50 KB` per token) crosses host staging at layer 24.

### `dav-m200` (`Ascend 310P3`) ISA Notes

Verified on `Ascend 310P3` with CANN `9.1.0` (`bisheng` clang 15.0.5):

1. `AscendC::Cast(half, float, ...)` requires `AscendC::RoundMode::CAST_NONE`; `CAST_RINT` is a silent no-op on
   `dav-m200`.
2. `AscendC::ShiftRight` on `uint16_t` is a silent no-op on `dav-m200`. Planar `Q2_0` unpacking therefore masks the
   four 2-bit planes directly with `{0x0303, 0x0C0C, 0x3030, 0xC0C0}`, casts `uint8_t -> half -> float`, adds
   `{-1, -4, -16, -64}`, and pre-scales deinterleaved activations by `{1, 1/4, 1/16, 1/64}` before `BlockReduceSum`.
3. Direct `uint32_t -> float` scalar casts fail AICore codegen; cast through `int32_t` first.

## Building

Inside a CANN `9.1.0` container (for example `quay.io/ascend/vllm-ascend:v0.23.0-310p`):

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S ascend -B build-ascend -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build-ascend ascend_parity_test strata
./build-ascend/ascend_parity_test
```

## Measured Numbers (`2x Huawei Atlas 300V Pro / Ascend 310P3`, Kunpeng 920)

Measured on Huawei TaiShan 200 2280 (`2x Kunpeng 920 5220`, 64 cores @ 2.6 GHz, 768 GB DDR4-2933) with two `48 GB`
`Atlas 300V Pro` (`Ascend 310P3`, driver `26.0.rc1`, CANN `9.1.0`) running
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`Q2_0`, `32K` context, `PLE=on`, no MTP drafter):

- **Operator parity (`ascend_parity_test` on `Ascend 310P3` vs FP32 CPU reference)**:
  - `f32 <-> f16`: `max_abs = 0.00e+00`
  - `aclnnMm FRACTAL_NZ`: `max_abs = 9.75e-04`, `rmse = 1.75e-04`
  - `gr_norm / post_down / post_up`: `xn = 2.86e-06`, `out = 7.15e-07`
  - `gdn_step` (verify + fold): `yg = 4.88e-04`, `state = 1.12e-08`
  - `qsa` (prep + radix-256 select + `INT8` online attention): `xq = 1.22e-04`, `radix_2051 = EXACT`
  - `moe` (`route + q2_gu + q2_down + combine`): `max_abs = 5.85e-07`, `rmse = 1.44e-07`
- **HBM footprint (`max_context = 32768`)**:
  - NPU 0 (layers `0..23` + PLE): `21.66 / 43.18 GiB` (`21.52 GiB` free)
  - NPU 1 (layers `24..47` + output head): `22.81 / 43.18 GiB` (`20.37 GiB` free)
- **End-to-end serving (`serve/server.py`, warm engine)**:
  - Model load + `FRACTAL_NZ` formatting (48 layers + `248,320` head): `71.2 s`
  - Warm decode (`T = 1`, greedy & Philox sampled): **`4.2 tok/s`** (`~238 ms/token`)
  - Warm prompt prefill (`34-792` tokens): **`5.1 - 6.0 tok/s`**, with multi-turn conversation checkpoint reuse
    (`cached_tokens` restored in `< 10 ms`)
