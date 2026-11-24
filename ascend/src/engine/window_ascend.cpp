// ascend/src/engine/window_ascend.cpp - Multi-NPU Huawei Ascend (CANN 9.1 / dav-m200 + Cube FRACTAL_NZ) engine for Strata.
//
// Implements strata::metal_engine::* over 1 or 2 Ascend 310P3 NPUs:
//   - Dense projections (HC down/inject/up, GDN fused qkv/z/alpha/beta + ssm_out, QSA fused q/k/v/iq/ik + attn_out,
//     MoE router + shared expert gate/up/sg + down, PLE key/value, and 248320-row LM head) are dequantized once at
//     init into FP16 FRACTAL_NZ weights on the DaVinci Cube Cores (aclnnMm).
//   - All 48 x 512 routed Q2_0 experts are split into 32B-aligned planar (qs, d) buffers in NPU HBM and executed on
//     8 AI Cores via custom Ascend C kernels (ascend_ops.asc).
//   - When >= 2 NPUs are visible (default on .100 with davinci4 + davinci6), layers 0..23 + PLE live on NPU 0 (~19.6 GiB)
//     and layers 24..47 + output_hc + LM head live on NPU 1 (~20.8 GiB), transferring only ~50 KB per token at layer 24.

#include "ascend_ops.h"
#include "geo.h"
#include "gguf_lite.h"
#include "ref_quant.h"

#include "acl/acl.h"
#include "aclnnop/aclnn_mm.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"

#define STRATA_GGUF_MAIN_DISABLED 1
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/conversation_cache.hpp"
#include "strata/core/dflash2.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/sampler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::gpu {
void ensure_acl_initialized();
void* host_view(const void* device_pointer);
}  // namespace strata::gpu

namespace strata::metal_engine {

namespace {

using namespace strata_metal::geo;

constexpr int MAXT = 8;
constexpr int PROMPT_CHUNK_DEFAULT = 64;
constexpr int MAX_ROWS = 64;  // max rows per NPU pass (supports T=1..8 decode/verify and C=1..64 prompt chunks)

constexpr int W_HC_DN_COLS = 336;   // 320 + 4 + 12 pad (multiple of 16)
constexpr int GDN_IN_COLS  = 16480; // 10240 + 6144 + 48 + 48
constexpr int QSA_IN_COLS  = 13952; // 12288 + 512 + 512 + 512 + 128
constexpr int MOE_IN_COLS  = 1808;  // 512 + 640 + 640 + 16
constexpr int QIDS_STRIDE  = 2056;  // >= BUD(2051), multiple of 8 ints (32B)

double now_ms() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

bool check_acl(aclError e, const char* what, std::string& err) {
    if (e == ACL_SUCCESS) return true;
    err = std::string("ascend window: ") + what + " failed (aclError=" + std::to_string((int)e) + ")";
    return false;
}

struct DevMem {
    int dev = 0;
    void* ptr = nullptr;
    size_t bytes = 0;

    DevMem() = default;
    DevMem(const DevMem&) = delete;
    DevMem& operator=(const DevMem&) = delete;
    DevMem(DevMem&& o) noexcept : dev(o.dev), ptr(o.ptr), bytes(o.bytes) {
        o.ptr = nullptr;
        o.bytes = 0;
    }
    DevMem& operator=(DevMem&& o) noexcept {
        if (this != &o) {
            reset();
            dev = o.dev;
            ptr = o.ptr;
            bytes = o.bytes;
            o.ptr = nullptr;
            o.bytes = 0;
        }
        return *this;
    }
    ~DevMem() { reset(); }

    void reset() {
        if (ptr) {
            (void)aclrtSetDevice(dev);
            (void)aclrtFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }

    bool alloc(int d, size_t sz, std::string& err, bool zero = true) {
        reset();
        dev = d;
        bytes = ((sz + 63) & ~size_t(63)) + 64;
        if (!check_acl(aclrtSetDevice(dev), "aclrtSetDevice", err)) return false;
        if (!check_acl(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc", err)) return false;
        if (zero && !check_acl(aclrtMemset(ptr, bytes, 0, bytes), "aclrtMemset", err)) return false;
        return true;
    }

    bool upload(const void* host, size_t sz, std::string& err) {
        if (sz == 0) return true;
        if (!check_acl(aclrtSetDevice(dev), "aclrtSetDevice", err)) return false;
        return check_acl(aclrtMemcpy(ptr, bytes, host, sz, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D", err);
    }

    bool download(void* host, size_t sz, std::string& err) const {
        if (sz == 0) return true;
        if (!check_acl(aclrtSetDevice(dev), "aclrtSetDevice", err)) return false;
        return check_acl(aclrtMemcpy(host, sz, ptr, sz, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H", err);
    }

    bool zero_fill(std::string& err) {
        if (!ptr || bytes == 0) return true;
        if (!check_acl(aclrtSetDevice(dev), "aclrtSetDevice", err)) return false;
        return check_acl(aclrtMemset(ptr, bytes, 0, bytes), "aclrtMemset", err);
    }
};

// Static FP16 FRACTAL_NZ weight matrix [K, N] on one NPU
struct NzMat {
    DevMem mem;
    int64_t K = 0;
    int64_t N_cols = 0;
    aclTensor* desc = nullptr;

    NzMat() = default;
    NzMat(const NzMat&) = delete;
    NzMat& operator=(const NzMat&) = delete;
    NzMat(NzMat&& o) noexcept : mem(std::move(o.mem)), K(o.K), N_cols(o.N_cols), desc(o.desc) { o.desc = nullptr; }
    NzMat& operator=(NzMat&& o) noexcept {
        if (this != &o) {
            if (desc) aclDestroyTensor(desc);
            mem = std::move(o.mem);
            K = o.K;
            N_cols = o.N_cols;
            desc = o.desc;
            o.desc = nullptr;
        }
        return *this;
    }
    ~NzMat() {
        if (desc) aclDestroyTensor(desc);
    }

    bool upload_and_format(int dev, aclrtStream stream, void* ws_ptr, size_t ws_cap,
                           const uint16_t* host_kn, int64_t k, int64_t n, std::string& err) {
        K = k;
        N_cols = n;
        const size_t sz = (size_t)K * (size_t)N_cols * sizeof(uint16_t);
        if (!mem.alloc(dev, sz, err, false)) return false;
        if (!mem.upload(host_kn, sz, err)) return false;

        int64_t shape[2] = {K, N_cols};
        int64_t stride[2] = {N_cols, 1};
        aclTensor* nd = aclCreateTensor(shape, 2, ACL_FLOAT16, stride, 0, ACL_FORMAT_ND, shape, 2, mem.ptr);
        uint64_t ws_need = 0;
        aclOpExecutor* exec = nullptr;
        if (!check_acl(aclnnTransMatmulWeightGetWorkspaceSize(nd, &ws_need, &exec),
                       "aclnnTransMatmulWeightGetWorkspaceSize", err)) {
            aclDestroyTensor(nd);
            return false;
        }
        DevMem tmp_ws;
        void* use_ws = ws_ptr;
        if (ws_need > ws_cap) {
            if (!tmp_ws.alloc(dev, (size_t)ws_need, err, false)) {
                aclDestroyTensor(nd);
                return false;
            }
            use_ws = tmp_ws.ptr;
        }
        if (!check_acl(aclnnTransMatmulWeight(use_ws, ws_need, exec, stream), "aclnnTransMatmulWeight", err) ||
            !check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(TransMatmulWeight)", err)) {
            aclDestroyTensor(nd);
            return false;
        }
        aclDestroyTensor(nd);
        desc = aclCreateTensor(shape, 2, ACL_FLOAT16, stride, 0, ACL_FORMAT_FRACTAL_NZ, shape, 2, mem.ptr);
        return desc != nullptr;
    }
};

struct LayerWeights {
    int dev = 0;
    bool is_qsa = false;

    // Hyper-connection attn & ffn
    DevMem hc_attn_norm;   // [4 * 2560] f32
    NzMat  W_hc_attn_dn;   // [10240, 336] f16 NZ
    NzMat  W_hc_attn_up;   // [320, 10240] f16 NZ

    DevMem hc_ffn_norm;    // [4 * 2560] f32
    NzMat  W_hc_ffn_dn;    // [10240, 336] f16 NZ
    NzMat  W_hc_ffn_up;    // [320, 10240] f16 NZ

    // GDN weights (when !is_qsa)
    NzMat  W_gdn_in;       // [2560, 16480] f16 NZ
    NzMat  W_ssm_out;      // [6144, 2560] f16 NZ
    DevMem ssm_conv_w;     // [10240 * 4] f32
    DevMem ssm_dt;         // [64] f32 (first 48 used)
    DevMem ssm_a;          // [64] f32 (first 48 used)
    DevMem ssm_norm;       // [128] f32

    // QSA weights (when is_qsa)
    NzMat  W_qsa_in;       // [2560, 13952] f16 NZ
    NzMat  W_attn_out;     // [6144, 2560] f16 NZ
    DevMem q_norm;         // [256] f32
    DevMem k_norm;         // [256] f32
    DevMem iq_norm;        // [128] f32
    DevMem ik_norm;        // [128] f32

    // MoE router + shared expert + planar Q2_0 routed experts
    NzMat  W_moe_in;       // [2560, 1808] f16 NZ
    NzMat  W_shexp_down;   // [640, 2560] f16 NZ
    DevMem gu_qs;          // [nexp, 2, 640, 640] u8
    DevMem gu_d;           // [nexp, 2, 640, 40] f16
    DevMem down_qs;        // [nexp, 2560, 160] u8
    DevMem down_d;         // [nexp, 2560, 10] f16
};

struct GdnState {
    int dev = 0;
    DevMem state;          // [48, 128, 128] f32
    DevMem hist[2];        // [10240, 4] f32 ping-pong
    DevMem qkv_save;       // [MAXT, 10240] f32
    DevMem z_save;         // [MAXT, 6144] f32
    DevMem ab_save;        // [MAXT, 96] f32
};

struct QsaLayerState {
    int dev = 0;
    DevMem kq;             // [max_cells + 16, 2, 256] i8
    DevMem vq;             // [max_cells + 16, 2, 256] i8
    DevMem ksc;            // [max_cells + 16, 16] f16
    DevMem vsc;            // [max_cells + 16, 16] f16
    DevMem tail;           // [4, 128] f32
    DevMem dead;           // [128] f32
    DevMem pooled;         // [max_blocks, 128] f32
};

// Scratch & stream per active NPU
struct NpuCtx {
    int dev = 0;
    aclrtStream stream = nullptr;
    DevMem ws;             // 96 MB Cube workspace
    std::vector<aclTensor*> temp_tensors;

    DevMem rope_cs;        // [(max_cells + 32) * 64] f32
    DevMem R;              // [MAX_ROWS, 4, 2560] f32
    DevMem bo_f32;         // [MAX_ROWS, 2560] f32
    DevMem bo_f16;         // [MAX_ROWS, 2560] f16
    DevMem inj[2];         // [MAX_ROWS * 4 + 16] f32
    DevMem xn_f32;         // [MAX_ROWS, 4, 2560] f32
    DevMem xn_f16;         // [MAX_ROWS, 10240] f16
    DevMem hc_dn_f16;      // [MAX_ROWS, 336] f16
    DevMem lo_f16;         // [MAX_ROWS, 320] f16
    DevMem gate_f16;       // [MAX_ROWS, 10240] f16
    DevMem xa_f16;         // [MAX_ROWS, 2560] f16
    DevMem xa_f32;         // [MAX_ROWS, 2560] f32
    DevMem xf_f16;         // [MAX_ROWS, 2560] f16
    DevMem xf_f32;         // [MAX_ROWS, 2560] f32

    // GDN scratch
    DevMem gdn_in_f16;     // [MAX_ROWS, 16480] f16
    DevMem yg_f16;         // [MAX_ROWS, 6144] f16

    // QSA scratch
    DevMem qsa_in_f16;     // [MAX_ROWS, 13952] f16
    DevMem qcur_f32;       // [MAX_ROWS, 24, 256] f32
    DevMem qgate_f32;      // [MAX_ROWS, 24, 256] f32
    DevMem qidx_f32;       // [MAX_ROWS, 4, 128] f32
    DevMem scores_f32;     // [MAX_ROWS, max_blocks] f32
    DevMem qids_i32;       // [MAX_ROWS, 2056] i32
    DevMem xq_f16;         // [MAX_ROWS, 6144] f16

    // MoE scratch (operates on windows of <= 8 rows)
    DevMem moe_in_f16;     // [MAX_ROWS, 1808] f16
    DevMem ids_i32;        // [16 * 16] i32
    DevMem wts_f32;        // [16 * 16] f32
    DevMem plan_i32;       // [P_INTS] i32
    DevMem sh_h_f16;       // [MAX_ROWS, 640] f16
    DevMem sg_f32;         // [16] f32
    DevMem h_f32;          // [128, 640] f32
    DevMem ye_f32;         // [128, 2560] f32
    DevMem ysh_f16;        // [MAX_ROWS, 2560] f16

    void clear_temp_tensors() {
        for (aclTensor* t : temp_tensors) {
            if (t) aclDestroyTensor(t);
        }
        temp_tensors.clear();
    }

    bool sync(std::string& err) {
        if (!check_acl(aclrtSetDevice(dev), "aclrtSetDevice", err)) return false;
        bool ok = check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream", err);
        clear_temp_tensors();
        return ok;
    }

    bool mm(const void* a_ptr, const NzMat& B, void* c_ptr, int64_t M, std::string& err) {
        if (!check_acl(aclrtSetDevice(dev), "aclrtSetDevice", err)) return false;
        int64_t aShape[2] = {M, B.K}, aStride[2] = {B.K, 1};
        int64_t cShape[2] = {M, B.N_cols}, cStride[2] = {B.N_cols, 1};
        aclTensor* tA = aclCreateTensor(aShape, 2, ACL_FLOAT16, aStride, 0, ACL_FORMAT_ND, aShape, 2, const_cast<void*>(a_ptr));
        aclTensor* tC = aclCreateTensor(cShape, 2, ACL_FLOAT16, cStride, 0, ACL_FORMAT_ND, cShape, 2, c_ptr);
        temp_tensors.push_back(tA);
        temp_tensors.push_back(tC);
        uint64_t ws_need = 0;
        aclOpExecutor* exec = nullptr;
        if (!check_acl(aclnnMmGetWorkspaceSize(tA, B.desc, tC, 0, &ws_need, &exec), "aclnnMmGetWorkspaceSize", err)) return false;
        if (ws_need > ws.bytes) {
            if (!check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(grow ws)", err)) return false;
            if (!ws.alloc(dev, std::max<size_t>((size_t)ws_need, ws.bytes * 2), err, false)) return false;
        }
        return check_acl(aclnnMm(ws.ptr, ws_need, exec, stream), "aclnnMm", err);
    }

    ~NpuCtx() {
        clear_temp_tensors();
        if (stream) {
            (void)aclrtSetDevice(dev);
            (void)aclrtDestroyStream(stream);
            stream = nullptr;
        }
    }
};

// Philox 4x32-10 matching src/kernels/cuda/sampler.cu
float philox_uniform_host(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t)counter, c1 = (uint32_t)(counter >> 32);
    uint32_t c2 = (uint32_t)seed,    c3 = (uint32_t)(seed >> 32);
    for (int i = 0; i < 10; ++i) {
        uint64_t p0 = (uint64_t)0x9E3779B9u * (uint64_t)c0;
        uint64_t p1 = (uint64_t)0xBB67AE85u * (uint64_t)c2;
        uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
        uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
        uint32_t n0 = hi1 ^ c1 ^ (uint32_t)i;
        uint32_t n1 = lo1;
        uint32_t n2 = hi0 ^ c3;
        uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return (float)(c0 >> 8) * (1.0f / 16777216.0f);
}

void gnorm_cpu(const float* x, const float* w, float* y, int n_embd, int hc, float eps) {
    for (int c = 0; c < hc; ++c) {
        const float* xc = x + (size_t)c * n_embd;
        double acc = 0.0;
        for (int d = 0; d < n_embd; ++d) {
            const float sq = xc[d] * xc[d];
            acc += (double)sq;
        }
        const float mean = (float)(acc / (double)n_embd);
        const float scale = 1.0f / std::sqrt(mean + eps);
        for (int d = 0; d < n_embd; ++d) y[(size_t)c * n_embd + d] = xc[d] * scale * w[(size_t)c * n_embd + d];
    }
}

struct MatSlice {
    const gguf::Tensor* t = nullptr;
    int64_t col_offset = 0;
    int64_t rows = 0;  // N_i in GGUF [K, N_i]
};

// Parallel dequantize + transpose of one or more GGUF matrices sharing input width K into row-major [K, N_total] FP16
void build_fused_kn_f16(const uint8_t* base_map, int64_t K, int64_t N_total,
                        const std::vector<MatSlice>& slices, std::vector<uint16_t>& out_kn,
                        unsigned nth, uint16_t pad_fill = 0) {
    out_kn.assign((size_t)K * (size_t)N_total, pad_fill);
    for (const MatSlice& sl : slices) {
        if (!sl.t || sl.rows <= 0) continue;
        const size_t rb = ref::row_bytes(sl.t->type, (int)K);
        const uint8_t* src_base = base_map + sl.t->offset;
        const int64_t R = sl.rows;
        const int64_t col0 = sl.col_offset;
        const int64_t n_tiles = (R + 31) / 32;
        const unsigned workers = std::max(1u, std::min<unsigned>(nth, (unsigned)n_tiles));
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (unsigned tid = 0; tid < workers; ++tid) {
            threads.emplace_back([&, tid] {
                std::vector<float> row_f32((size_t)K);
                std::vector<uint16_t> buf_f16((size_t)32 * (size_t)K);
                uint16_t blk[32][32];
                const int64_t t0 = (n_tiles * tid) / workers;
                const int64_t t1 = (n_tiles * (tid + 1)) / workers;
                for (int64_t ti = t0; ti < t1; ++ti) {
                    const int64_t r_blk = ti * 32;
                    const int64_t br_cnt = std::min<int64_t>(32, R - r_blk);
                    for (int64_t b = 0; b < br_cnt; ++b) {
                        ref::dequant(sl.t->type, src_base + (size_t)(r_blk + b) * rb, row_f32.data(), (size_t)K);
                        uint16_t* dst_b = buf_f16.data() + (size_t)b * (size_t)K;
                        for (int64_t k = 0; k < K; ++k) dst_b[k] = ref::f32_to_f16(row_f32[(size_t)k]);
                    }
                    for (int64_t k0 = 0; k0 < K; k0 += 32) {
                        const int64_t kc = std::min<int64_t>(32, K - k0);
                        for (int64_t b = 0; b < br_cnt; ++b) {
                            const uint16_t* src_r = buf_f16.data() + (size_t)b * (size_t)K + (size_t)k0;
                            for (int64_t dk = 0; dk < kc; ++dk) blk[dk][b] = src_r[dk];
                        }
                        for (int64_t dk = 0; dk < kc; ++dk) {
                            std::memcpy(out_kn.data() + (size_t)(k0 + dk) * (size_t)N_total + (size_t)(col0 + r_blk),
                                        blk[dk], (size_t)br_cnt * sizeof(uint16_t));
                        }
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
    }
}

void split_q2_experts_gu(const uint8_t* gate_base, const uint8_t* up_base, int nexp,
                         std::vector<uint8_t>& out_qs, std::vector<uint16_t>& out_d, unsigned nth) {
    constexpr size_t gu_row_bytes = (N / 64) * 18;  // 720 B
    out_qs.resize((size_t)nexp * 2 * NFF * 640);
    out_d.resize((size_t)nexp * 2 * NFF * 40);
    std::vector<std::thread> threads;
    for (unsigned tid = 0; tid < nth; ++tid) {
        threads.emplace_back([&, tid] {
            for (int e = (int)tid; e < nexp; e += (int)nth) {
                for (int mat = 0; mat < 2; ++mat) {
                    const uint8_t* src = (mat == 0 ? gate_base : up_base) + (size_t)e * NFF * gu_row_bytes;
                    for (int r = 0; r < NFF; ++r) {
                        const uint8_t* row = src + (size_t)r * gu_row_bytes;
                        uint8_t* dqs = out_qs.data() + (((size_t)e * 2 + mat) * NFF + r) * 640;
                        uint16_t* dd = out_d.data()  + (((size_t)e * 2 + mat) * NFF + r) * 40;
                        for (int b = 0; b < 40; ++b) {
                            std::memcpy(&dd[b], row + b * 18, 2);
                            std::memcpy(dqs + b * 16, row + b * 18 + 2, 16);
                        }
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();
}

void split_q2_experts_down(const uint8_t* down_base, int nexp,
                           std::vector<uint8_t>& out_qs, std::vector<uint16_t>& out_d, unsigned nth) {
    constexpr size_t dn_row_bytes = (NFF / 64) * 18;  // 180 B
    out_qs.resize((size_t)nexp * N * 160);
    out_d.resize((size_t)nexp * N * 10);
    std::vector<std::thread> threads;
    for (unsigned tid = 0; tid < nth; ++tid) {
        threads.emplace_back([&, tid] {
            for (int e = (int)tid; e < nexp; e += (int)nth) {
                const uint8_t* src = down_base + (size_t)e * N * dn_row_bytes;
                for (int r = 0; r < N; ++r) {
                    const uint8_t* row = src + (size_t)r * dn_row_bytes;
                    uint8_t* dqs = out_qs.data() + ((size_t)e * N + r) * 160;
                    uint16_t* dd = out_d.data()  + ((size_t)e * N + r) * 10;
                    for (int b = 0; b < 10; ++b) {
                        std::memcpy(&dd[b], row + b * 18, 2);
                        std::memcpy(dqs + b * 16, row + b * 18 + 2, 16);
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();
}

}  // namespace

class WindowAscend {
public:
    bool ready() const { return ready_; }
    int64_t vocab() const { return n_vocab_; }
    void set_max_t(int t) { max_t_ = std::max(1, std::min(t, MAXT)); }

    bool init(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, const std::string& shard1,
              int max_t, int64_t max_cells, std::string& err) {
        strata::gpu::ensure_acl_initialized();
        ss_ = &ss;
        main_ss_ = &ss;
        max_t_ = std::max(1, std::min(max_t, MAXT));
        eps_ = 1e-6f;

        const bool experts_ok = g.n_expert == NEXP || (NEXP == 512 && g.n_expert == 256);
        if (!strata::geometry_unsupported(g).empty() || !experts_ok || g.n_expert_used != TOPK) {
            err = "ascend window: geometry differs from Qwen3.8-Flash-Next";
            return false;
        }
        nexp_ = (int)g.n_expert;

        const double t0 = now_ms();
        if (!f_.open(shard1, err)) return false;

        uint32_t dev_cnt = 0;
        if (!check_acl(aclrtGetDeviceCount(&dev_cnt), "aclrtGetDeviceCount", err) || dev_cnt == 0) {
            if (err.empty()) err = "ascend window: no Ascend NPU devices visible";
            return false;
        }
        const bool force_single = std::getenv("STRATA_ASCEND_SINGLE_NPU") && std::string(std::getenv("STRATA_ASCEND_SINGLE_NPU")) == "1";
        n_npu_ = (!force_single && dev_cnt >= 2) ? 2 : 1;
        split_layer_ = (n_npu_ == 2) ? (NL / 2) : NL;

        max_cells_ = std::max<int64_t>(4096, ((max_cells + 63) / 64) * 64 + 64);
        max_blocks_ = ((max_cells_ / 4 + 16 + 7) / 8) * 8;  // multiple of 8 floats (32B)

        const gguf::Tensor* t_out = f_.find("output.weight");
        if (!t_out || t_out->shape.size() < 2) {
            err = "ascend window: missing output.weight in " + shard1;
            return false;
        }
        n_vocab_ = (int64_t)t_out->shape[1];
        vocab_pad_ = ((n_vocab_ + 127) / 128) * 128;

        // Precompute RoPE table on host
        const double rope_base = f_.meta_num.count("qwen4exp.rope.freq_base") ? f_.meta_num["qwen4exp.rope.freq_base"] : 1e7;
        std::vector<float> h_rope((size_t)(max_cells_ + 32) * 64);
        for (int64_t p = 0; p < max_cells_ + 32; ++p) {
            for (int i = 0; i < 32; ++i) {
                const double th = (double)p * std::pow(rope_base, -2.0 * (double)i / 64.0);
                h_rope[(size_t)(p * 32 + i) * 2 + 0] = (float)std::cos(th);
                h_rope[(size_t)(p * 32 + i) * 2 + 1] = (float)std::sin(th);
            }
        }

        // Initialize per-NPU streams & scratch buffers
        npu_.resize((size_t)n_npu_);
        for (int d = 0; d < n_npu_; ++d) {
            npu_[d] = std::make_unique<NpuCtx>();
            NpuCtx& c = *npu_[d];
            c.dev = d;
            if (!check_acl(aclrtSetDevice(d), "aclrtSetDevice", err) ||
                !check_acl(aclrtCreateStream(&c.stream), "aclrtCreateStream", err)) return false;
            if (!c.ws.alloc(d, 96ULL << 20, err, false) ||
                !c.rope_cs.alloc(d, h_rope.size() * 4, err, false) ||
                !c.rope_cs.upload(h_rope.data(), h_rope.size() * 4, err) ||
                !c.R.alloc(d, (size_t)MAX_ROWS * HCD * 4, err) ||
                !c.bo_f32.alloc(d, (size_t)MAX_ROWS * N * 4, err) ||
                !c.bo_f16.alloc(d, (size_t)MAX_ROWS * N * 2, err) ||
                !c.inj[0].alloc(d, (size_t)(MAX_ROWS * HC + 16) * 4, err) ||
                !c.inj[1].alloc(d, (size_t)(MAX_ROWS * HC + 16) * 4, err) ||
                !c.xn_f32.alloc(d, (size_t)MAX_ROWS * HCD * 4, err) ||
                !c.xn_f16.alloc(d, (size_t)MAX_ROWS * HCD * 2, err) ||
                !c.hc_dn_f16.alloc(d, (size_t)MAX_ROWS * W_HC_DN_COLS * 2, err) ||
                !c.lo_f16.alloc(d, (size_t)MAX_ROWS * LR * 2, err) ||
                !c.gate_f16.alloc(d, (size_t)MAX_ROWS * HCD * 2, err) ||
                !c.xa_f16.alloc(d, (size_t)MAX_ROWS * N * 2, err) ||
                !c.xa_f32.alloc(d, (size_t)MAX_ROWS * N * 4, err) ||
                !c.xf_f16.alloc(d, (size_t)MAX_ROWS * N * 2, err) ||
                !c.xf_f32.alloc(d, (size_t)MAX_ROWS * N * 4, err) ||
                !c.gdn_in_f16.alloc(d, (size_t)MAX_ROWS * GDN_IN_COLS * 2, err) ||
                !c.yg_f16.alloc(d, (size_t)MAX_ROWS * ZV * 2, err) ||
                !c.qsa_in_f16.alloc(d, (size_t)MAX_ROWS * QSA_IN_COLS * 2, err) ||
                !c.qcur_f32.alloc(d, (size_t)MAX_ROWS * QOUT * 4, err) ||
                !c.qgate_f32.alloc(d, (size_t)MAX_ROWS * QOUT * 4, err) ||
                !c.qidx_f32.alloc(d, (size_t)MAX_ROWS * IQR * 4, err) ||
                !c.scores_f32.alloc(d, (size_t)MAX_ROWS * max_blocks_ * 4, err) ||
                !c.qids_i32.alloc(d, (size_t)MAX_ROWS * QIDS_STRIDE * 4, err) ||
                !c.xq_f16.alloc(d, (size_t)MAX_ROWS * QOUT * 2, err) ||
                !c.moe_in_f16.alloc(d, (size_t)MAX_ROWS * MOE_IN_COLS * 2, err) ||
                !c.ids_i32.alloc(d, 256 * 4, err) ||
                !c.wts_f32.alloc(d, 256 * 4, err) ||
                !c.plan_i32.alloc(d, strata::ascend_ops::P_INTS * 4, err) ||
                !c.sh_h_f16.alloc(d, (size_t)MAX_ROWS * NFF * 2, err) ||
                !c.sg_f32.alloc(d, 64 * 4, err) ||
                !c.h_f32.alloc(d, 128 * NFF * 4, err) ||
                !c.ye_f32.alloc(d, 128 * N * 4, err) ||
                !c.ysh_f16.alloc(d, (size_t)MAX_ROWS * N * 2, err)) {
                return false;
            }
        }

        const unsigned nth = std::max(1u, std::min(24u, std::thread::hardware_concurrency()));
        const uint8_t* map_base = (const uint8_t*)f_.map;

        auto load_f32_vec = [&](const std::string& name, size_t pad_to, std::vector<float>& out) -> bool {
            const gguf::Tensor* t = f_.find(name);
            if (!t) { err = "ascend window: missing tensor " + name; return false; }
            uint64_t ne = 1;
            for (uint64_t s : t->shape) ne *= s;
            out.assign(std::max<size_t>(pad_to, (size_t)ne), 0.0f);
            ref::dequant(t->type, map_base + t->offset, out.data(), (size_t)ne);
            return true;
        };

        auto upload_f32_dev = [&](int dev, const std::string& name, size_t pad_to, DevMem& dm) -> bool {
            std::vector<float> tmp;
            if (!load_f32_vec(name, pad_to, tmp)) return false;
            return dm.alloc(dev, tmp.size() * 4, err, false) && dm.upload(tmp.data(), tmp.size() * 4, err);
        };

        // PLE setup on NPU 0
        ple_on_ = ss.ple.ready() && f_.find("blk.1.ple_key.weight") != nullptr;
        if (ple_on_) {
            NpuCtx& c0 = *npu_[0];
            if (!load_f32_vec("blk.1.ple_norm_key.weight", HCD, h_ple_nkey_) ||
                !load_f32_vec("blk.1.ple_norm_query.weight", HCD, h_ple_nqry_) ||
                !load_f32_vec("blk.1.ple_norm_conv.weight", HCD, h_ple_nconv_) ||
                !load_f32_vec("blk.1.ple_conv1d.weight", (size_t)strata::kernels::PLE_CONV_KERNEL * HCD, h_ple_conv_w_)) {
                return false;
            }
            // Match Metal/CUDA PLE conv1d half rounding
            for (float& v : h_ple_conv_w_) v = ref::f16_to_f32(ref::f32_to_f16(v));

            std::vector<uint16_t> h_kn;
            build_fused_kn_f16(map_base, N, HCD + N,
                               {{f_.find("blk.1.ple_key.weight"), 0, HCD},
                                {f_.find("blk.1.ple_value.weight"), HCD, N}},
                               h_kn, nth);
            if (!W_ple_kv_.upload_and_format(0, c0.stream, c0.ws.ptr, c0.ws.bytes, h_kn.data(), N, HCD + N, err) ||
                !d_ple_emb_f32_.alloc(0, (size_t)MAX_ROWS * N * 4, err) ||
                !d_ple_emb_f16_.alloc(0, (size_t)MAX_ROWS * N * 2, err) ||
                !d_ple_kv_f16_.alloc(0, (size_t)MAX_ROWS * (HCD + N) * 2, err)) {
                return false;
            }
        }
        ple_hist_.assign((size_t)strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM, 0.0f);
        last_norm_.assign((size_t)MAX_ROWS * strata::kernels::NG_HC_DIM, 0.0f);

        // Allocate & upload all 48 layers + states
        layers_.resize(NL);
        gdn_states_.resize(NGDN);
        qsa_states_.resize(NQSA);

        std::fprintf(stderr, "strata ascend: loading %d layers onto %d NPU(s) (split=%d, %u CPU threads)...\n",
                     NL, n_npu_, split_layer_, nth);
        std::fflush(stderr);

        std::vector<uint16_t> h_kn;
        std::vector<uint8_t> h_qs;
        std::vector<uint16_t> h_d;

        for (int l = 0; l < NL; ++l) {
            const int dev = (l < split_layer_) ? 0 : 1;
            NpuCtx& nc = *npu_[dev];
            LayerWeights& lw = layers_[(size_t)l];
            lw.dev = dev;
            lw.is_qsa = (l % QI == QI - 1);
            const std::string b = "blk." + std::to_string(l) + ".";

            // 1) HC attn + HC ffn
            if (!upload_f32_dev(dev, b + "hc_attn_norm.weight", HCD, lw.hc_attn_norm) ||
                !upload_f32_dev(dev, b + "hc_ffn_norm.weight", HCD, lw.hc_ffn_norm)) return false;

            build_fused_kn_f16(map_base, HCD, W_HC_DN_COLS,
                               {{f_.find(b + "hc_attn_down.weight"), 0, LR},
                                {f_.find(b + "hc_attn_inject.weight"), LR, HC}},
                               h_kn, nth);
            if (!lw.W_hc_attn_dn.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), HCD, W_HC_DN_COLS, err)) return false;

            build_fused_kn_f16(map_base, LR, HCD, {{f_.find(b + "hc_attn_up.weight"), 0, HCD}}, h_kn, nth);
            if (!lw.W_hc_attn_up.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), LR, HCD, err)) return false;

            build_fused_kn_f16(map_base, HCD, W_HC_DN_COLS,
                               {{f_.find(b + "hc_ffn_down.weight"), 0, LR},
                                {f_.find(b + "hc_ffn_inject.weight"), LR, HC}},
                               h_kn, nth);
            if (!lw.W_hc_ffn_dn.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), HCD, W_HC_DN_COLS, err)) return false;

            build_fused_kn_f16(map_base, LR, HCD, {{f_.find(b + "hc_ffn_up.weight"), 0, HCD}}, h_kn, nth);
            if (!lw.W_hc_ffn_up.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), LR, HCD, err)) return false;

            // 2) Mixer: GDN or QSA
            if (!lw.is_qsa) {
                const int gi = l - l / QI;
                build_fused_kn_f16(map_base, N, GDN_IN_COLS,
                                   {{f_.find(b + "attn_qkv.weight"),  0,             CONV},
                                    {f_.find(b + "attn_gate.weight"), CONV,          ZV},
                                    {f_.find(b + "ssm_alpha.weight"), CONV + ZV,     HV},
                                    {f_.find(b + "ssm_beta.weight"),  CONV + ZV + HV, HV}},
                                   h_kn, nth);
                if (!lw.W_gdn_in.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), N, GDN_IN_COLS, err)) return false;

                build_fused_kn_f16(map_base, ZV, N, {{f_.find(b + "ssm_out.weight"), 0, N}}, h_kn, nth);
                if (!lw.W_ssm_out.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), ZV, N, err)) return false;

                if (!upload_f32_dev(dev, b + "ssm_conv1d.weight", CONV * 4, lw.ssm_conv_w) ||
                    !upload_f32_dev(dev, b + "ssm_dt.bias", 64, lw.ssm_dt) ||
                    !upload_f32_dev(dev, b + "ssm_a", 64, lw.ssm_a) ||
                    !upload_f32_dev(dev, b + "ssm_norm.weight", SS, lw.ssm_norm)) return false;

                GdnState& gs = gdn_states_[(size_t)gi];
                gs.dev = dev;
                if (!gs.state.alloc(dev, (size_t)HV * SS * SS * 4, err) ||
                    !gs.hist[0].alloc(dev, (size_t)CONV * 4 * 4, err) ||
                    !gs.hist[1].alloc(dev, (size_t)CONV * 4 * 4, err) ||
                    !gs.qkv_save.alloc(dev, (size_t)MAXT * CONV * 4, err) ||
                    !gs.z_save.alloc(dev, (size_t)MAXT * ZV * 4, err) ||
                    !gs.ab_save.alloc(dev, (size_t)MAXT * 96 * 4, err)) return false;
            } else {
                const int qi = l / QI;
                build_fused_kn_f16(map_base, N, QSA_IN_COLS,
                                   {{f_.find(b + "attn_q.weight"),         0,                     QFULL},
                                    {f_.find(b + "attn_k.weight"),         QFULL,                 KV},
                                    {f_.find(b + "attn_v.weight"),         QFULL + KV,            KV},
                                    {f_.find(b + "indexer.q_proj.weight"), QFULL + 2 * KV,        IQR},
                                    {f_.find(b + "indexer.k_proj.weight"), QFULL + 2 * KV + IQR,  ID}},
                                   h_kn, nth);
                if (!lw.W_qsa_in.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), N, QSA_IN_COLS, err)) return false;

                build_fused_kn_f16(map_base, QOUT, N, {{f_.find(b + "attn_output.weight"), 0, N}}, h_kn, nth);
                if (!lw.W_attn_out.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), QOUT, N, err)) return false;

                if (!upload_f32_dev(dev, b + "attn_q_norm.weight", QD, lw.q_norm) ||
                    !upload_f32_dev(dev, b + "attn_k_norm.weight", QD, lw.k_norm) ||
                    !upload_f32_dev(dev, b + "indexer.q_norm.weight", ID, lw.iq_norm) ||
                    !upload_f32_dev(dev, b + "indexer.k_norm.weight", ID, lw.ik_norm)) return false;

                QsaLayerState& qs = qsa_states_[(size_t)qi];
                qs.dev = dev;
                if (!qs.kq.alloc(dev, (size_t)(max_cells_ + 16) * QKV * QD, err) ||
                    !qs.vq.alloc(dev, (size_t)(max_cells_ + 16) * QKV * QD, err) ||
                    !qs.ksc.alloc(dev, (size_t)(max_cells_ + 16) * 16 * 2, err) ||
                    !qs.vsc.alloc(dev, (size_t)(max_cells_ + 16) * 16 * 2, err) ||
                    !qs.tail.alloc(dev, 4 * ID * 4, err) ||
                    !qs.dead.alloc(dev, ID * 4, err) ||
                    !qs.pooled.alloc(dev, (size_t)max_blocks_ * ID * 4, err)) return false;
            }

            // 3) MoE router + shared expert + Q2_0 routed experts
            build_fused_kn_f16(map_base, N, MOE_IN_COLS,
                               {{f_.find(b + "ffn_gate_inp.weight"),       0,             nexp_},
                                {f_.find(b + "ffn_gate_shexp.weight"),     512,           NFF},
                                {f_.find(b + "ffn_up_shexp.weight"),       512 + NFF,     NFF},
                                {f_.find(b + "ffn_gate_inp_shexp.weight"), 512 + 2 * NFF, 1}},
                               h_kn, nth);
            if (!lw.W_moe_in.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), N, MOE_IN_COLS, err)) return false;

            build_fused_kn_f16(map_base, NFF, N, {{f_.find(b + "ffn_down_shexp.weight"), 0, N}}, h_kn, nth);
            if (!lw.W_shexp_down.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), NFF, N, err)) return false;

            const gguf::Tensor* tg = f_.find(b + "ffn_gate_exps.weight");
            const gguf::Tensor* tu = f_.find(b + "ffn_up_exps.weight");
            const gguf::Tensor* td = f_.find(b + "ffn_down_exps.weight");
            if (!tg || !tu || !td || tg->type != ref::Q2_0 || tu->type != ref::Q2_0 || td->type != ref::Q2_0) {
                err = "ascend window: routed experts in layer " + std::to_string(l) + " are not Q2_0";
                return false;
            }

            split_q2_experts_gu(map_base + tg->offset, map_base + tu->offset, nexp_, h_qs, h_d, nth);
            if (!lw.gu_qs.alloc(dev, h_qs.size(), err, false) || !lw.gu_qs.upload(h_qs.data(), h_qs.size(), err) ||
                !lw.gu_d.alloc(dev, h_d.size() * 2, err, false) || !lw.gu_d.upload(h_d.data(), h_d.size() * 2, err)) return false;

            split_q2_experts_down(map_base + td->offset, nexp_, h_qs, h_d, nth);
            if (!lw.down_qs.alloc(dev, h_qs.size(), err, false) || !lw.down_qs.upload(h_qs.data(), h_qs.size(), err) ||
                !lw.down_d.alloc(dev, h_d.size() * 2, err, false) || !lw.down_d.upload(h_d.data(), h_d.size() * 2, err)) return false;

            if ((l + 1) % 8 == 0 || l + 1 == NL) {
                std::fprintf(stderr, "strata ascend: loaded %d / %d layers (%.1f s)\n", l + 1, NL, (now_ms() - t0) / 1000.0);
                std::fflush(stderr);
            }
        }

        // Output HC + LM Head on final NPU
        const int last_dev = n_npu_ - 1;
        NpuCtx& lc = *npu_[last_dev];
        if (!upload_f32_dev(last_dev, "output_hc_norm.weight", HCD, out_hc_norm_)) return false;
        build_fused_kn_f16(map_base, HCD, W_HC_DN_COLS, {{f_.find("output_hc_down.weight"), 0, LR}}, h_kn, nth);
        if (!W_out_hc_dn_.upload_and_format(last_dev, lc.stream, lc.ws.ptr, lc.ws.bytes, h_kn.data(), HCD, W_HC_DN_COLS, err)) return false;

        build_fused_kn_f16(map_base, LR, HCD, {{f_.find("output_hc_up.weight"), 0, HCD}}, h_kn, nth);
        if (!W_out_hc_up_.upload_and_format(last_dev, lc.stream, lc.ws.ptr, lc.ws.bytes, h_kn.data(), LR, HCD, err)) return false;

        // Pad any columns past n_vocab_ with -65504 (0xFBFF in FP16) so padded columns never win argmax
        build_fused_kn_f16(map_base, N, vocab_pad_, {{t_out, 0, n_vocab_}}, h_kn, nth, 0xFBFFu);
        if (!W_lm_head_.upload_and_format(last_dev, lc.stream, lc.ws.ptr, lc.ws.bytes, h_kn.data(), N, vocab_pad_, err) ||
            !d_lm_f16_.alloc(last_dev, (size_t)MAXT * vocab_pad_ * 2, err) ||
            !d_lm_f32_.alloc(last_dev, (size_t)MAXT * vocab_pad_ * 4, err) ||
            !d_arg_pv_.alloc(last_dev, (size_t)MAXT * 64 * 4, err) ||
            !d_arg_pi_.alloc(last_dev, (size_t)MAXT * 64 * 4, err) ||
            !d_tok_out_.alloc(last_dev, 8 * 4, err)) {
            return false;
        }

        h_R_.resize((size_t)MAX_ROWS * HCD);
        h_bo_.resize((size_t)MAX_ROWS * N);
        h_inj_.resize((size_t)(MAX_ROWS * HC + 16));
        h_ple_emb_.resize((size_t)MAX_ROWS * N);
        h_ple_kv_f16_.resize((size_t)MAX_ROWS * (HCD + N));
        h_logits_f32_.resize((size_t)MAXT * vocab_pad_);

        for (int d = 0; d < n_npu_; ++d) {
            size_t free_b = 0, total_b = 0;
            (void)aclrtSetDevice(d);
            (void)aclrtGetMemInfo(ACL_HBM_MEM, &free_b, &total_b);
            std::fprintf(stderr, "strata ascend: NPU %d (%s) HBM used %.2f / %.2f GiB (%.2f GiB free)\n",
                         d, aclrtGetSocName() ? aclrtGetSocName() : "Ascend",
                         (double)(total_b - free_b) / (1024.0 * 1024.0 * 1024.0),
                         (double)total_b / (1024.0 * 1024.0 * 1024.0),
                         (double)free_b / (1024.0 * 1024.0 * 1024.0));
        }
        std::fprintf(stderr, "strata ascend: Qwen3.8-Flash-Next window ready in %.1f s (max_cells=%lld, PLE=%s)\n",
                     (now_ms() - t0) / 1000.0, (long long)max_cells_, ple_on_ ? "on" : "off");
        std::fflush(stderr);
        ready_ = true;
        return true;
    }

    bool fold_pending(std::string& err) {
        if (pend_fold_ <= 0) return true;
        const int nf = pend_fold_;
        const int src_hp = 1 - hp_;  // hp_ already flipped when commit() scheduled pend_fold_
        for (int l = 0; l < NL; ++l) {
            if (l % QI == QI - 1) continue;
            const int gi = l - l / QI;
            const LayerWeights& lw = layers_[(size_t)l];
            GdnState& gs = gdn_states_[(size_t)gi];
            NpuCtx& nc = *npu_[lw.dev];
            if (!check_acl(aclrtSetDevice(lw.dev), "aclrtSetDevice", err)) return false;
            strata::ascend_ops::launch_gdn_step(
                nc.stream, gs.state.ptr, gs.hist[src_hp].ptr, gs.hist[1 - src_hp].ptr,
                nc.gdn_in_f16.ptr, gs.qkv_save.ptr, gs.z_save.ptr, gs.ab_save.ptr,
                lw.ssm_conv_w.ptr, lw.ssm_dt.ptr, lw.ssm_a.ptr, lw.ssm_norm.ptr,
                nc.yg_f16.ptr, (uint32_t)nf, (uint32_t)nf, /*from_saved=*/1u, eps_);
        }
        for (int d = 0; d < n_npu_; ++d) {
            if (!npu_[d]->sync(err)) return false;
        }
        pend_fold_ = 0;
        return true;
    }

    bool run_forward(int Tn, const int32_t* tokens, int64_t pos0, const float* const* embd_rows,
                     bool head, bool immediate_commit, int32_t* out, std::string& err) {
        const int row_cap = head ? MAXT : MAX_ROWS;
        if (Tn < 1 || Tn > row_cap || (head && Tn > max_t_)) {
            err = "ascend window: window size out of range";
            return false;
        }
        if (pos0 + Tn + 1 >= max_cells_) {
            err = "ascend window: position exceeds max_cells";
            return false;
        }
        if (!fold_pending(err)) return false;

        // 1) Token embeddings on host -> broadcast across 4 HC streams -> upload to NPU 0
        const gguf::Tensor* te = f_.find("token_embd.weight");
        if (!te) { err = "ascend window: missing token_embd.weight"; return false; }
        const size_t rb = ref::row_bytes(te->type, N);
        const uint8_t* map_base = (const uint8_t*)f_.map;
        for (int t = 0; t < Tn; ++t) {
            float* r = h_R_.data() + (size_t)t * HCD;
            if (embd_rows && embd_rows[t]) {
                std::memcpy(r, embd_rows[t], (size_t)N * 4);
            } else {
                ref::dequant(te->type, map_base + te->offset + (size_t)tokens[t] * rb, r, N);
            }
            for (int c = 1; c < HC; ++c) std::memcpy(r + (size_t)c * N, r, (size_t)N * 4);
        }

        NpuCtx& c0 = *npu_[0];
        if (!c0.R.upload(h_R_.data(), (size_t)Tn * HCD * 4, err)) return false;

        // 2) Gather PLE n-gram rows on host & launch W_ple_kv Cube matmul on NPU 0
        if (ple_on_) {
            std::vector<uint32_t> ple_rows((size_t)Tn * strata::kernels::PLE_N_HEADS);
            int32_t prev[2] = {ss_->ple_prev[0], ss_->ple_prev[1]};
            for (int t = 0; t < Tn; ++t) {
                strata::kernels::ngram_rows(&tokens[t], prev, 1, main_ss_->ple.consts,
                                            ple_rows.data() + (size_t)t * strata::kernels::PLE_N_HEADS);
                prev[0] = prev[1];
                prev[1] = tokens[t];
            }
            if (!main_ss_->ple.table->gather_batch(ple_rows.data(), (size_t)Tn, h_ple_emb_.data(), err)) return false;
            if (!d_ple_emb_f32_.upload(h_ple_emb_.data(), (size_t)Tn * N * 4, err)) return false;
            if (!check_acl(aclrtSetDevice(0), "aclrtSetDevice(0)", err)) return false;
            strata::ascend_ops::launch_f32_to_f16(c0.stream, d_ple_emb_f32_.ptr, d_ple_emb_f16_.ptr, (uint32_t)(Tn * N));
            if (!c0.mm(d_ple_emb_f16_.ptr, W_ple_kv_, d_ple_kv_f16_.ptr, Tn, err)) return false;
        }

        int cur_inj = 0;
        const uint32_t gdn_commit = (immediate_commit || Tn == 1) ? (uint32_t)Tn : 0u;

        for (int l = 0; l < NL; ++l) {
            // Cross-NPU activation handoff at split_layer_
            if (n_npu_ > 1 && l == split_layer_) {
                if (!c0.sync(err)) return false;
                const size_t r_bytes = (size_t)Tn * HCD * 4;
                const size_t b_bytes = (size_t)Tn * N * 4;
                const size_t i_bytes = (size_t)(((Tn * HC + 7) / 8) * 8) * 4;
                if (!c0.R.download(h_R_.data(), r_bytes, err) ||
                    !c0.bo_f32.download(h_bo_.data(), b_bytes, err) ||
                    !c0.inj[cur_inj].download(h_inj_.data(), i_bytes, err)) return false;
                NpuCtx& c1 = *npu_[1];
                if (!c1.R.upload(h_R_.data(), r_bytes, err) ||
                    !c1.bo_f32.upload(h_bo_.data(), b_bytes, err) ||
                    !c1.inj[cur_inj].upload(h_inj_.data(), i_bytes, err)) return false;
            }

            const LayerWeights& lw = layers_[(size_t)l];
            NpuCtx& nc = *npu_[lw.dev];
            if (!check_acl(aclrtSetDevice(lw.dev), "aclrtSetDevice", err)) return false;

            // ---- Mixer Hyper-Connection Read ----
            const bool apply_attn = (l > 0) && !(l == 1 && ple_on_);
            strata::ascend_ops::launch_gr_norm(
                nc.stream, nc.R.ptr, nc.bo_f32.ptr, nc.inj[cur_inj].ptr, lw.hc_attn_norm.ptr,
                nc.xn_f32.ptr, nc.xn_f16.ptr, (uint32_t)Tn, apply_attn ? 1u : 0u, eps_);
            if (!nc.mm(nc.xn_f16.ptr, lw.W_hc_attn_dn, nc.hc_dn_f16.ptr, Tn, err)) return false;
            strata::ascend_ops::launch_gr_post_down(
                nc.stream, nc.hc_dn_f16.ptr, nc.lo_f16.ptr, nc.inj[1 - cur_inj].ptr, (uint32_t)Tn);
            if (!nc.mm(nc.lo_f16.ptr, lw.W_hc_attn_up, nc.gate_f16.ptr, Tn, err)) return false;
            strata::ascend_ops::launch_gr_post_up(
                nc.stream, nc.xn_f32.ptr, nc.gate_f16.ptr, nc.xa_f16.ptr, nc.xa_f32.ptr, (uint32_t)Tn);
            cur_inj = 1 - cur_inj;

            // ---- Mixer: GDN or QSA ----
            if (!lw.is_qsa) {
                const int gi = l - l / QI;
                GdnState& gs = gdn_states_[(size_t)gi];
                if (!nc.mm(nc.xa_f16.ptr, lw.W_gdn_in, nc.gdn_in_f16.ptr, Tn, err)) return false;
                strata::ascend_ops::launch_gdn_step(
                    nc.stream, gs.state.ptr, gs.hist[hp_].ptr, gs.hist[1 - hp_].ptr,
                    nc.gdn_in_f16.ptr,
                    gdn_commit == 0 ? gs.qkv_save.ptr : nullptr,
                    gdn_commit == 0 ? gs.z_save.ptr   : nullptr,
                    gdn_commit == 0 ? gs.ab_save.ptr  : nullptr,
                    lw.ssm_conv_w.ptr, lw.ssm_dt.ptr, lw.ssm_a.ptr, lw.ssm_norm.ptr,
                    nc.yg_f16.ptr, (uint32_t)Tn, gdn_commit, /*from_saved=*/0u, eps_);
                if (!nc.mm(nc.yg_f16.ptr, lw.W_ssm_out, nc.bo_f16.ptr, Tn, err)) return false;
                strata::ascend_ops::launch_f16_to_f32(nc.stream, nc.bo_f16.ptr, nc.bo_f32.ptr, (uint32_t)(Tn * N));
            } else {
                const int qi = l / QI;
                QsaLayerState& qs = qsa_states_[(size_t)qi];
                if (!nc.mm(nc.xa_f16.ptr, lw.W_qsa_in, nc.qsa_in_f16.ptr, Tn, err)) return false;
                strata::ascend_ops::launch_qsa_prep(
                    nc.stream, nc.qsa_in_f16.ptr,
                    lw.q_norm.ptr, lw.k_norm.ptr, lw.iq_norm.ptr, lw.ik_norm.ptr,
                    nc.rope_cs.ptr,
                    nc.qcur_f32.ptr, nc.qgate_f32.ptr, nc.qidx_f32.ptr,
                    qs.kq.ptr, qs.vq.ptr, qs.ksc.ptr, qs.vsc.ptr,
                    qs.tail.ptr, qs.dead.ptr, qs.pooled.ptr,
                    (uint32_t)pos0, 0u, (uint32_t)Tn, eps_);
                strata::ascend_ops::launch_qsa_select(
                    nc.stream, nc.qidx_f32.ptr, qs.pooled.ptr, qs.dead.ptr,
                    nc.scores_f32.ptr, nc.qids_i32.ptr,
                    (uint32_t)pos0, (uint32_t)max_blocks_, (uint32_t)Tn);
                strata::ascend_ops::launch_qsa_attn_merge(
                    nc.stream, nc.qcur_f32.ptr, nc.qgate_f32.ptr,
                    qs.kq.ptr, qs.vq.ptr, qs.ksc.ptr, qs.vsc.ptr,
                    nc.qids_i32.ptr, nc.xq_f16.ptr,
                    (uint32_t)pos0, (uint32_t)Tn);
                if (!nc.mm(nc.xq_f16.ptr, lw.W_attn_out, nc.bo_f16.ptr, Tn, err)) return false;
                strata::ascend_ops::launch_f16_to_f32(nc.stream, nc.bo_f16.ptr, nc.bo_f32.ptr, (uint32_t)(Tn * N));
            }

            // ---- FFN Hyper-Connection Read ----
            strata::ascend_ops::launch_gr_norm(
                nc.stream, nc.R.ptr, nc.bo_f32.ptr, nc.inj[cur_inj].ptr, lw.hc_ffn_norm.ptr,
                nc.xn_f32.ptr, nc.xn_f16.ptr, (uint32_t)Tn, /*apply=*/1u, eps_);
            if (!nc.mm(nc.xn_f16.ptr, lw.W_hc_ffn_dn, nc.hc_dn_f16.ptr, Tn, err)) return false;
            strata::ascend_ops::launch_gr_post_down(
                nc.stream, nc.hc_dn_f16.ptr, nc.lo_f16.ptr, nc.inj[1 - cur_inj].ptr, (uint32_t)Tn);
            if (!nc.mm(nc.lo_f16.ptr, lw.W_hc_ffn_up, nc.gate_f16.ptr, Tn, err)) return false;
            strata::ascend_ops::launch_gr_post_up(
                nc.stream, nc.xn_f32.ptr, nc.gate_f16.ptr, nc.xf_f16.ptr, nc.xf_f32.ptr, (uint32_t)Tn);
            cur_inj = 1 - cur_inj;

            // ---- MoE Sublayer (batched dense Cube + 8-row Q2_0 expert windows) ----
            if (!nc.mm(nc.xf_f16.ptr, lw.W_moe_in, nc.moe_in_f16.ptr, Tn, err)) return false;
            for (int t0 = 0; t0 < Tn; t0 += MAXT) {
                const int sub = std::min(MAXT, Tn - t0);
                uint8_t* p_moe_in = (uint8_t*)nc.moe_in_f16.ptr + (size_t)t0 * MOE_IN_COLS * 2;
                uint8_t* p_sh_h   = (uint8_t*)nc.sh_h_f16.ptr   + (size_t)t0 * NFF * 2;
                uint8_t* p_xf     = (uint8_t*)nc.xf_f32.ptr     + (size_t)t0 * N * 4;
                uint8_t* p_ysh    = (uint8_t*)nc.ysh_f16.ptr    + (size_t)t0 * N * 2;
                uint8_t* p_bo     = (uint8_t*)nc.bo_f32.ptr     + (size_t)t0 * N * 4;

                strata::ascend_ops::launch_moe_route_shexp(
                    nc.stream, p_moe_in, nc.ids_i32.ptr, nc.wts_f32.ptr, nc.plan_i32.ptr,
                    p_sh_h, nc.sg_f32.ptr, (uint32_t)sub, (uint32_t)nexp_);
                if (!nc.mm(p_sh_h, lw.W_shexp_down, p_ysh, sub, err)) return false;
                strata::ascend_ops::launch_moe_q2_gu(
                    nc.stream, lw.gu_qs.ptr, lw.gu_d.ptr, p_xf, nc.plan_i32.ptr, nc.h_f32.ptr, (uint32_t)sub);
                strata::ascend_ops::launch_moe_q2_down(
                    nc.stream, lw.down_qs.ptr, lw.down_d.ptr, nc.h_f32.ptr, nc.plan_i32.ptr, nc.ye_f32.ptr);
                strata::ascend_ops::launch_moe_combine(
                    nc.stream, nc.ye_f32.ptr, nc.plan_i32.ptr, p_ysh, nc.sg_f32.ptr, p_bo, (uint32_t)sub);
            }

            // ---- PLE Block after Layer 0 ----
            if (l == 0 && ple_on_) {
                // Apply Layer 0's pending FFN write to R on NPU 0 before PLE
                strata::ascend_ops::launch_gr_norm(
                    c0.stream, c0.R.ptr, c0.bo_f32.ptr, c0.inj[cur_inj].ptr, layers_[1].hc_attn_norm.ptr,
                    c0.xn_f32.ptr, c0.xn_f16.ptr, (uint32_t)Tn, /*apply=*/1u, eps_);
                if (!c0.sync(err)) return false;
                if (!c0.R.download(h_R_.data(), (size_t)Tn * HCD * 4, err) ||
                    !d_ple_kv_f16_.download(h_ple_kv_f16_.data(), (size_t)Tn * (HCD + N) * 2, err)) return false;
                run_ple_cpu(Tn);
                if (!c0.R.upload(h_R_.data(), (size_t)Tn * HCD * 4, err)) return false;
            }
        }

        // 3) Final Hyper-Connection + LM Head on final NPU
        const int last_dev = n_npu_ - 1;
        NpuCtx& lc = *npu_[last_dev];
        if (!check_acl(aclrtSetDevice(last_dev), "aclrtSetDevice(last)", err)) return false;
        strata::ascend_ops::launch_gr_norm(
            lc.stream, lc.R.ptr, lc.bo_f32.ptr, lc.inj[cur_inj].ptr, out_hc_norm_.ptr,
            lc.xn_f32.ptr, lc.xn_f16.ptr, (uint32_t)Tn, /*apply=*/1u, eps_);

        if (head) {
            if (!lc.mm(lc.xn_f16.ptr, W_out_hc_dn_, lc.hc_dn_f16.ptr, Tn, err)) return false;
            strata::ascend_ops::launch_gr_post_down(
                lc.stream, lc.hc_dn_f16.ptr, lc.lo_f16.ptr, lc.inj[1 - cur_inj].ptr, (uint32_t)Tn);
            if (!lc.mm(lc.lo_f16.ptr, W_out_hc_up_, lc.gate_f16.ptr, Tn, err)) return false;
            strata::ascend_ops::launch_gr_post_up(
                lc.stream, lc.xn_f32.ptr, lc.gate_f16.ptr, lc.xa_f16.ptr, lc.xa_f32.ptr, (uint32_t)Tn);

            if (!lc.mm(lc.xa_f16.ptr, W_lm_head_, d_lm_f16_.ptr, Tn, err)) return false;
            strata::ascend_ops::launch_argmax_f16(
                lc.stream, d_lm_f16_.ptr, d_arg_pv_.ptr, d_arg_pi_.ptr, d_tok_out_.ptr,
                (uint32_t)Tn, (uint32_t)vocab_pad_);
        }

        for (int d = 0; d < n_npu_; ++d) {
            if (!npu_[d]->sync(err)) return false;
        }

        if (head) {
            int32_t tok_buf[8] = {};
            if (!d_tok_out_.download(tok_buf, 8 * 4, err)) return false;
            for (int t = 0; t < Tn; ++t) {
                tok_out_[t] = tok_buf[t];
                if (out) out[t] = tok_buf[t];
            }
        } else if (out) {
            for (int t = 0; t < Tn; ++t) out[t] = -1;
        }

        if (gdn_commit > 0) hp_ = 1 - hp_;
        logits_copied_mask_ = 0;
        last_T_ = Tn;
        last_head_ = head;
        for (int t = 0; t < std::min(Tn, MAX_ROWS); ++t) last_tokens_[t] = tokens[t];

        if (immediate_commit) {
            return advance_ple_and_tokens(Tn);
        }
        return true;
    }

    bool commit(int n_keep, std::string& err) {
        if (n_keep < 1 || n_keep > last_T_) {
            err = "ascend window: commit count out of range";
            return false;
        }
        if (last_T_ > 1) {
            pend_fold_ = n_keep;
            hp_ = 1 - hp_;
        }
        return advance_ple_and_tokens(n_keep);
    }

    void set_sampling(const strata::kernels::SamplerParams* sp, const int32_t* hist, int hist_len) {
        samp_on_ = sp != nullptr;
        if (sp) samp_ = *sp;
        samp_hist_ = hist;
        samp_hist_len_ = hist_len;
    }

    bool copy_logits(int t, float* host) {
        if (t < 0 || t >= last_T_ || !last_head_) return false;
        std::string err;
        if (!ensure_host_logits(last_T_, err)) return false;
        std::memcpy(host, h_logits_f32_.data() + (size_t)t * vocab_pad_, (size_t)n_vocab_ * 4);
        return true;
    }

    bool sample_rows(int Tn, int32_t* out, std::string& err) {
        if (!samp_on_ || Tn < 1 || Tn > MAXT) return true;
        if (!sample_run(0, Tn, samp_, samp_hist_, samp_hist_len_, err)) return false;
        for (int t = 0; t < Tn; ++t) out[t] = tok_out_[t];
        return true;
    }

    bool sample_slot_row(int row, const strata::kernels::SamplerParams& sp, const int32_t* hist, int hist_len,
                         int32_t* tok, std::string& err) {
        if (row < 0 || row >= MAXT) return true;
        if (!sample_run(row, 1, sp, hist, hist_len, err)) return false;
        *tok = tok_out_[row];
        return true;
    }

    bool prompt(const int64_t* tokens, int64_t n, int64_t pos0, const float* const* embd_rows,
                const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err) {
        static const int chunk_sz = [] {
            const char* e = std::getenv("STRATA_ASCEND_PROMPT_CHUNK");
            int v = e ? std::atoi(e) : PROMPT_CHUNK_DEFAULT;
            return std::max(1, std::min(v, MAX_ROWS));
        }();
        std::vector<int32_t> tk((size_t)chunk_sz);
        for (int64_t c = 0; c < n; c += chunk_sz) {
            const int k = (int)std::min<int64_t>(chunk_sz, n - c);
            for (int i = 0; i < k; ++i) tk[(size_t)i] = (int32_t)tokens[c + i];
            const float* const* er = embd_rows ? (embd_rows + c) : nullptr;
            if (!run_forward(k, tk.data(), pos0 + c, er, /*head=*/false, /*immediate_commit=*/true, nullptr, err)) return false;
            if (on_chunk && !on_chunk(nullptr, k, pos0 + c, err)) return false;
        }
        return true;
    }

    void reset() {
        pend_fold_ = 0;
        last_T_ = 0;
        std::string err;
        for (auto& gs : gdn_states_) {
            (void)gs.state.zero_fill(err);
            (void)gs.hist[0].zero_fill(err);
            (void)gs.hist[1].zero_fill(err);
        }
        for (auto& qs : qsa_states_) {
            (void)qs.tail.zero_fill(err);
            (void)qs.dead.zero_fill(err);
        }
        std::fill(ple_hist_.begin(), ple_hist_.end(), 0.0f);
    }

    bool checkpoint_save(strata::core::ConversationCheckpoint& c, std::string& err) {
        if (!fold_pending(err)) return false;
        const size_t gs_b = (size_t)HV * SS * SS * 4;
        const size_t hs_b = (size_t)CONV * 4 * 4;
        c.gdn.resize(gdn_states_.size() * (gs_b + hs_b));
        for (size_t i = 0; i < gdn_states_.size(); ++i) {
            uint8_t* dst = c.gdn.data() + i * (gs_b + hs_b);
            if (!gdn_states_[i].state.download(dst, gs_b, err) ||
                !gdn_states_[i].hist[hp_].download(dst + gs_b, hs_b, err)) return false;
        }
        c.ple.resize(ple_hist_.size() * 4);
        std::memcpy(c.ple.data(), ple_hist_.data(), c.ple.size());
        const size_t ts_b = 4 * ID * 4, ds_b = ID * 4;
        c.tails.resize(qsa_states_.size() * ts_b);
        c.dead.resize(qsa_states_.size() * ds_b);
        for (size_t j = 0; j < qsa_states_.size(); ++j) {
            if (!qsa_states_[j].tail.download(c.tails.data() + j * ts_b, ts_b, err) ||
                !qsa_states_[j].dead.download(c.dead.data() + j * ds_b, ds_b, err)) return false;
        }
        c.block_pos.assign(qsa_states_.size() * 4, 0);
        return true;
    }

    bool checkpoint_restore(const strata::core::ConversationCheckpoint& c, std::string& err) {
        const size_t gs_b = (size_t)HV * SS * SS * 4;
        const size_t hs_b = (size_t)CONV * 4 * 4;
        const size_t ts_b = 4 * ID * 4, ds_b = ID * 4;
        if (c.gdn.size() != gdn_states_.size() * (gs_b + hs_b) ||
            c.ple.size() != ple_hist_.size() * 4 ||
            c.tails.size() != qsa_states_.size() * ts_b ||
            c.dead.size() != qsa_states_.size() * ds_b) {
            err = "ascend window: checkpoint size mismatch";
            return false;
        }
        pend_fold_ = 0;
        last_T_ = 0;
        for (size_t i = 0; i < gdn_states_.size(); ++i) {
            const uint8_t* src = c.gdn.data() + i * (gs_b + hs_b);
            if (!gdn_states_[i].state.upload(src, gs_b, err) ||
                !gdn_states_[i].hist[hp_].upload(src + gs_b, hs_b, err)) return false;
        }
        std::memcpy(ple_hist_.data(), c.ple.data(), c.ple.size());
        for (size_t j = 0; j < qsa_states_.size(); ++j) {
            if (!qsa_states_[j].tail.upload(c.tails.data() + j * ts_b, ts_b, err) ||
                !qsa_states_[j].dead.upload(c.dead.data() + j * ds_b, ds_b, err)) return false;
        }
        return true;
    }

private:
    void run_ple_cpu(int Tn) {
        using namespace strata::kernels;
        const int hc_dim = NG_HC_DIM;
        std::vector<float> pkey(hc_dim), pval(N), key(hc_dim), query(hc_dim), gated(hc_dim), norm(hc_dim);
        std::vector<float> hist = ple_hist_;
        const float inv_sqrt_n = 1.0f / std::sqrt((float)N);

        for (int t = 0; t < Tn; ++t) {
            const uint16_t* kv_row = h_ple_kv_f16_.data() + (size_t)t * (hc_dim + N);
            for (int i = 0; i < hc_dim; ++i) pkey[i] = ref::f16_to_f32(kv_row[i]);
            for (int i = 0; i < N; ++i)      pval[i] = ref::f16_to_f32(kv_row[hc_dim + i]);

            float* hidden = h_R_.data() + (size_t)t * hc_dim;
            gnorm_cpu(pkey.data(), h_ple_nkey_.data(), key.data(), N, HC, NG_RMS_EPS);
            gnorm_cpu(hidden, h_ple_nqry_.data(), query.data(), N, HC, NG_RMS_EPS);

            float gate[HC];
            for (int c = 0; c < HC; ++c) {
                double acc = 0.0;
                for (int d = 0; d < N; ++d) acc += (double)(key[(size_t)c * N + d] * query[(size_t)c * N + d]);
                const float s = (float)acc * inv_sqrt_n;
                const float mag = std::sqrt(std::max(std::fabs(s), 1e-6f));
                const float sgn = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
                gate[c] = 1.0f / (1.0f + std::exp(-(sgn * mag)));
            }
            for (int i = 0; i < hc_dim; ++i) gated[i] = pval[i % N] * gate[i / N];
            gnorm_cpu(gated.data(), h_ple_nconv_.data(), norm.data(), N, HC, NG_RMS_EPS);

            for (int c = 0; c < hc_dim; ++c) {
                float acc = 0.0f;
                for (int k = 0; k < PLE_CONV_KERNEL; ++k) {
                    const int row = NG_HIST - (PLE_CONV_KERNEL - 1 - k) * NGRAM_SIZE;
                    const float v = (row == NG_HIST) ? norm[c] : hist[(size_t)c * NG_HIST + (size_t)row];
                    acc += h_ple_conv_w_[k + PLE_CONV_KERNEL * c] * v;
                }
                const float conv = acc / (1.0f + std::exp(-acc));
                hidden[c] = hidden[c] + gated[c] + conv;
            }
            std::memcpy(last_norm_.data() + (size_t)t * hc_dim, norm.data(), (size_t)hc_dim * 4);
            for (int c = 0; c < hc_dim; ++c) {
                float* col = hist.data() + (size_t)c * NG_HIST;
                for (int r = 0; r + 1 < NG_HIST; ++r) col[r] = col[r + 1];
                col[NG_HIST - 1] = norm[c];
            }
        }
    }

    bool advance_ple_and_tokens(int n_keep) {
        using namespace strata::kernels;
        for (int t = 0; t < n_keep; ++t) {
            if (ple_on_) {
                const float* norm = last_norm_.data() + (size_t)t * NG_HC_DIM;
                for (int c = 0; c < NG_HC_DIM; ++c) {
                    float* col = ple_hist_.data() + (size_t)c * NG_HIST;
                    for (int r = 0; r + 1 < NG_HIST; ++r) col[r] = col[r + 1];
                    col[NG_HIST - 1] = norm[c];
                }
            }
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
        return true;
    }

    bool ensure_host_logits(int Tn, std::string& err) {
        if (logits_copied_mask_ == ((1u << Tn) - 1u)) return true;
        const int last_dev = n_npu_ - 1;
        NpuCtx& lc = *npu_[last_dev];
        if (!check_acl(aclrtSetDevice(last_dev), "aclrtSetDevice(logits)", err)) return false;
        strata::ascend_ops::launch_f16_to_f32(lc.stream, d_lm_f16_.ptr, d_lm_f32_.ptr, (uint32_t)(Tn * vocab_pad_));
        if (!lc.sync(err)) return false;
        if (!d_lm_f32_.download(h_logits_f32_.data(), (size_t)Tn * vocab_pad_ * 4, err)) return false;
        logits_copied_mask_ = (1u << Tn) - 1u;
        return true;
    }

    // Exact llama.cpp / Strata sampler chain: logit_bias -> penalties -> top_k -> top_p -> min_p -> temperature -> Philox draw
    bool sample_run(int row0, int n, const strata::kernels::SamplerParams& sp, const int32_t* hist_in, int hl, std::string& err) {
        if (!ensure_host_logits(row0 + n, err)) return false;
        const bool greedy = sp.greedy || sp.temperature <= 0.0f;
        const float* bias_host = sp.logit_bias ? (const float*)strata::gpu::host_view(sp.logit_bias) : nullptr;
        const int32_t* hist_host = (hist_in && hl > 0 && sp.penalty_last_n > 0)
                                       ? (const int32_t*)strata::gpu::host_view(hist_in)
                                       : nullptr;
        const int pen_n = hist_host ? std::min(sp.penalty_last_n, hl) : 0;

        std::vector<float> row((size_t)n_vocab_);
        std::vector<int> pen_cnt;
        if (pen_n > 0) pen_cnt.assign((size_t)n_vocab_, 0);

        for (int r = 0; r < n; ++r) {
            const float* src = h_logits_f32_.data() + (size_t)(row0 + r) * vocab_pad_;
            std::memcpy(row.data(), src, (size_t)n_vocab_ * 4);

            if (bias_host) {
                for (int64_t i = 0; i < n_vocab_; ++i) row[(size_t)i] += bias_host[i];
            }

            if (pen_n > 0) {
                const int32_t* hrow = hist_host + (size_t)r * hl + (hl - pen_n);
                for (int j = 0; j < pen_n; ++j) {
                    int32_t tid = hrow[j];
                    if (tid >= 0 && tid < n_vocab_) ++pen_cnt[(size_t)tid];
                }
                for (int j = 0; j < pen_n; ++j) {
                    int32_t tid = hrow[j];
                    if (tid >= 0 && tid < n_vocab_ && pen_cnt[(size_t)tid] > 0) {
                        int c = pen_cnt[(size_t)tid];
                        pen_cnt[(size_t)tid] = 0;
                        float v = row[(size_t)tid];
                        v = (v <= 0.0f) ? (v * sp.penalty_repeat) : (v / sp.penalty_repeat);
                        v -= (float)c * sp.penalty_freq + sp.penalty_present;
                        row[(size_t)tid] = v;
                    }
                }
            }

            if (greedy) {
                int32_t best_i = 0;
                float best_v = row[0];
                for (int64_t i = 1; i < n_vocab_; ++i) {
                    if (row[(size_t)i] > best_v) {
                        best_v = row[(size_t)i];
                        best_i = (int32_t)i;
                    }
                }
                tok_out_[row0 + r] = best_i;
                continue;
            }

            const int K = (sp.top_k > 0 && sp.top_k < 64) ? sp.top_k : 64;
            std::vector<std::pair<float, int32_t>> cand((size_t)n_vocab_);
            for (int64_t i = 0; i < n_vocab_; ++i) cand[(size_t)i] = {row[(size_t)i], (int32_t)i};
            const size_t kk = std::min<size_t>((size_t)K, cand.size());
            std::partial_sort(cand.begin(), cand.begin() + kk, cand.end(),
                              [](const auto& a, const auto& b) {
                                  return a.first > b.first || (a.first == b.first && a.second < b.second);
                              });
            cand.resize(kk);

            // Softmax pre-temperature for top_p & min_p filters
            float mx = cand[0].first;
            std::vector<float> p(kk);
            float psum = 0.0f;
            for (size_t i = 0; i < kk; ++i) {
                p[i] = std::exp(cand[i].first - mx);
                psum += p[i];
            }
            for (size_t i = 0; i < kk; ++i) p[i] /= psum;

            size_t keep = kk;
            const size_t min_keep = (size_t)std::max(1, sp.min_keep);
            if (sp.top_p < 1.0f) {
                float cum = 0.0f;
                for (size_t i = 0; i < keep; ++i) {
                    cum += p[i];
                    if (cum >= sp.top_p && i + 1 >= min_keep) {
                        keep = i + 1;
                        break;
                    }
                }
            }
            if (sp.min_p > 0.0f && keep > min_keep) {
                const float thr = sp.min_p * p[0];
                for (size_t i = min_keep; i < keep; ++i) {
                    if (p[i] < thr) {
                        keep = i;
                        break;
                    }
                }
            }

            // Temperature scaling + Philox draw
            const float inv_t = 1.0f / sp.temperature;
            float mx_t = cand[0].first * inv_t;
            float tsum = 0.0f;
            for (size_t i = 0; i < keep; ++i) {
                p[i] = std::exp(cand[i].first * inv_t - mx_t);
                tsum += p[i];
            }
            const float u = philox_uniform_host(sp.seed, sp.counter + (uint64_t)r) * tsum;
            float acc = 0.0f;
            int32_t picked = cand[keep - 1].second;
            for (size_t i = 0; i < keep; ++i) {
                acc += p[i];
                if (u < acc) {
                    picked = cand[i].second;
                    break;
                }
            }
            tok_out_[row0 + r] = picked;
        }
        return true;
    }

    bool ready_ = false;
    strata::core::SessionState* ss_ = nullptr;
    strata::core::SessionState* main_ss_ = nullptr;
    gguf::File f_;

    int n_npu_ = 1;
    int split_layer_ = NL;
    int max_t_ = 1;
    int nexp_ = NEXP;
    int64_t max_cells_ = 0;
    int64_t max_blocks_ = 0;
    int64_t n_vocab_ = 0;
    int64_t vocab_pad_ = 0;
    float eps_ = 1e-6f;

    std::vector<std::unique_ptr<NpuCtx>> npu_;
    std::vector<LayerWeights> layers_;
    std::vector<GdnState> gdn_states_;
    std::vector<QsaLayerState> qsa_states_;

    bool ple_on_ = false;
    std::vector<float> h_ple_nkey_, h_ple_nqry_, h_ple_nconv_, h_ple_conv_w_;
    std::vector<float> ple_hist_, last_norm_;
    NzMat W_ple_kv_;
    DevMem d_ple_emb_f32_, d_ple_emb_f16_, d_ple_kv_f16_;

    DevMem out_hc_norm_;
    NzMat W_out_hc_dn_, W_out_hc_up_, W_lm_head_;
    DevMem d_lm_f16_, d_lm_f32_, d_arg_pv_, d_arg_pi_, d_tok_out_;

    std::vector<float> h_R_, h_bo_, h_inj_, h_ple_emb_, h_logits_f32_;
    std::vector<uint16_t> h_ple_kv_f16_;

    int hp_ = 0;
    int pend_fold_ = 0;
    int last_T_ = 0;
    bool last_head_ = false;
    uint32_t logits_copied_mask_ = 0;
    int32_t last_tokens_[MAX_ROWS] = {};
    int32_t tok_out_[MAXT] = {};

    bool samp_on_ = false;
    strata::kernels::SamplerParams samp_{};
    const int32_t* samp_hist_ = nullptr;
    int samp_hist_len_ = 0;
};

// ============================================================================
// Public strata::metal_engine Entry Points
// ============================================================================
namespace {
std::string g_shard1;
int g_slot_cap = 0;
WindowAscend* g_the_window = nullptr;
const int32_t* g_mrope_dev = nullptr;
}  // namespace

void set_model_shard(const char* shard1) { g_shard1 = shard1 ? shard1 : ""; }
void set_batch_slots(int n) { g_slot_cap = n > 0 ? std::min(n, 8) : 0; }

std::string draft_vocab_default() {
    const char* e = std::getenv("STRATA_METAL_DRAFT_VOCAB");
    if (e != nullptr && e[0] != 0 && std::string(e) != "full") return e;
    return std::string(STRATA_ASCEND_DATA_DEFAULT) + "/draft_vocab.bin";
}

bool draft_vocab_whole_head() {
    const char* e = std::getenv("STRATA_METAL_DRAFT_VOCAB");
    return e != nullptr && std::string(e) == "full";
}

void* window_create() {
    if (g_the_window == nullptr) g_the_window = new WindowAscend();
    return g_the_window;
}

void window_destroy(void* w) {
    if (w != nullptr && w == g_the_window) {
        delete g_the_window;
        g_the_window = nullptr;
    }
}

bool window_init(void* w, const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int max_t, int64_t max_cells,
                 int64_t* n_vocab, std::string& err) {
    auto* x = (WindowAscend*)w;
    if (!x->ready()) {
        if (g_shard1.empty()) {
            err = "ascend window: model shard is not set (--native SHARD1)";
            return false;
        }
        if (!x->init(g, ss, g_shard1, max_t, max_cells, err)) return false;
    }
    x->set_max_t(max_t);
    *n_vocab = x->vocab();
    return true;
}

bool window_ensure(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int64_t max_cells, std::string& err) {
    int64_t nv = 0;
    return window_init(window_create(), g, ss, MAXT, max_cells, &nv, err);
}

void window_set_sampling(void* w, const strata::kernels::SamplerParams* sp, const int32_t* hist, int hist_len) {
    ((WindowAscend*)w)->set_sampling(sp, hist, hist_len);
}

bool window_sample_slot_row(void* w, int row, const strata::kernels::SamplerParams& sp, const int32_t* hist, int hist_len,
                            int32_t* tok, std::string& err) {
    return ((WindowAscend*)w)->sample_slot_row(row, sp, hist, hist_len, tok, err);
}

bool window_run(void* w, int T, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err) {
    static const bool logpos = std::getenv("STRATA_LOGPOS") != nullptr;
    auto* win = (WindowAscend*)w;
    if (!win->run_forward(T, tokens, pos0, nullptr, head || logpos, /*immediate_commit=*/false, out, err)) return false;
    return !head || win->sample_rows(T, out, err);
}

bool window_commit(void* w, int n_keep, std::string& err) {
    return ((WindowAscend*)w)->commit(n_keep, err);
}

bool window_copy_logits(void* w, int t, float* host) {
    return ((WindowAscend*)w)->copy_logits(t, host);
}

void window_reset(const strata::core::SessionState*) {
    if (g_the_window && g_the_window->ready()) g_the_window->reset();
}

bool window_checkpoint_save(strata::core::ConversationCheckpoint& c, const strata::core::SessionState*, std::string& err) {
    if (!g_the_window || !g_the_window->ready()) {
        err = "ascend window: not initialized";
        return false;
    }
    return g_the_window->checkpoint_save(c, err);
}

bool window_checkpoint_restore(const strata::core::ConversationCheckpoint& c, const strata::core::SessionState*, std::string& err) {
    if (!g_the_window || !g_the_window->ready()) {
        err = "ascend window: not initialized";
        return false;
    }
    return g_the_window->checkpoint_restore(c, err);
}

bool window_init_slots(void*, const std::vector<strata::core::SessionState*>& slots, std::string& err) {
    if (!slots.empty()) {
        err = "ascend window: --batch slots not enabled yet (use single-sequence or suffix speculation)";
        return false;
    }
    return true;
}

bool window_run_slots(void*, const int*, int, const int32_t*, const int64_t*, int32_t*, std::string& err) {
    err = "ascend window: --batch slots not enabled yet";
    return false;
}

bool window_slot_copy_main(bool, int, int64_t, bool, std::string& err) {
    err = "ascend window: --batch slots not enabled yet";
    return false;
}

bool window_taps(void*, const std::vector<int>&, const float**, std::string& err) {
    err = "ascend window: DFlash2 taps are not supported on Ascend";
    return false;
}

bool prompt_taps(const std::vector<int>&, int64_t, const std::function<bool(const float*, int64_t, int64_t, std::string&)>&,
                 std::string& err) {
    err = "ascend window: DFlash2 prompt taps are not supported on Ascend";
    return false;
}

bool prompt_run(const int64_t* tokens, int64_t n, int64_t pos0, const float* const* embd_rows,
                const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err) {
    if (!g_the_window || !g_the_window->ready()) {
        err = "ascend prompt: decode window is not initialized";
        return false;
    }
    return g_the_window->prompt(tokens, n, pos0, embd_rows, on_chunk, err);
}

bool mtp_load(const std::string&, int, std::string& err) {
    err = "ascend window: neural MTP draft layer is not ported yet; omit --mtp or use suffix speculation (--spec 2)";
    return false;
}

void mtp_bind_state(strata::core::QsaState&) {}

bool mtp_prefill(const int32_t*, int64_t, int64_t, std::string& err) {
    err = "ascend window: neural MTP draft layer is not ported yet";
    return false;
}

bool mtp_draft(int, const int32_t*, int64_t, int, int, int32_t*, float*, float, int*, std::string& err) {
    err = "ascend window: neural MTP draft layer is not ported yet";
    return false;
}

}  // namespace strata::metal_engine

namespace strata::kernels {
void mrope_table_set(const int32_t* device_table) { strata::metal_engine::g_mrope_dev = device_table; }
const int32_t* mrope_table() { return strata::metal_engine::g_mrope_dev; }
}  // namespace strata::kernels

namespace strata::core {
Dflash2Drafter::~Dflash2Drafter() {}
bool Dflash2Drafter::load(const std::string&, int, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::bind(const WeightTable&, const NativeHead*, const std::string&, std::string& err, int) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::bind_raw(EmbedFn, const void*, int, int64_t, const std::vector<int32_t>&, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::clone_alloc(const Dflash2Drafter&, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::clone_bind(const Dflash2Drafter&, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::copy_state_from(Dflash2Drafter&, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
float* Dflash2Drafter::defer_rows(int64_t, int, std::string& err) { err = "dflash2: not supported on Ascend"; return nullptr; }
bool Dflash2Drafter::catch_up(int64_t, std::string&) { return true; }
void Dflash2Drafter::reset() { have_ctx_ = 0; cur_.hist.clear(); ++epoch_; ring_hi_ = 0; }
bool Dflash2Drafter::begin_request(const int32_t*, int64_t, int64_t, bool* cold) { if (cold) *cold = false; return true; }
void Dflash2Drafter::init_ring_log() {}
void Dflash2Drafter::copy_log_state_from(const Dflash2Drafter&) {}
uint64_t Dflash2Drafter::ring_log_bytes() const { return 0; }
uint64_t Dflash2Drafter::ring_stash_bytes() const { return 0; }
void Dflash2Drafter::note_rows(const int32_t*, int64_t, int64_t) {}
bool Dflash2Drafter::context(const float*, int64_t, int64_t, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::context_async(const float*, int64_t, int64_t, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::context_wait(std::string&) { return true; }
bool Dflash2Drafter::draft(const float*, int, int64_t, int32_t, int32_t*, float*, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::draft_launch(const float*, int, int64_t, int32_t, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::draft_finish(int32_t*, float*, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::prepare(const float*, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
void Dflash2Drafter::set_max_drafts(int n) { ndraft_ = n; }
bool Dflash2Drafter::debug_rows(std::vector<float>&, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
bool Dflash2Drafter::debug_ring(std::vector<float>&, std::vector<int64_t>&, std::string& err) { err = "dflash2: not supported on Ascend"; return false; }
}  // namespace strata::core
