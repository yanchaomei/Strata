// ascend/src/engine/window_ascend.cpp - Extreme Multi-NPU Huawei Ascend (CANN 9.1 / dav-m200 + INT8/FP16 Cube FRACTAL_NZ) engine for Strata.
//
// Implements strata::metal_engine::* across 1..8 Ascend 310P3 NPUs:
//   - Dense projections (HC down/inject/up, GDN fused qkv/z/alpha/beta + ssm_out, QSA fused q/k/v/iq/ik + attn_out,
//     MoE router + shared expert gate/up/sg + down, PLE key/value, and 248320-row LM head) run on DaVinci FP16 Cube
//     Cores (aclnnMm + ACL_FORMAT_FRACTAL_NZ).
//   - Routed MoE experts (E=512, top_k=10) dynamically fill up to ~98.5% of every NPU's HBM with 280-TOPS INT8
//     FRACTAL_NZ Grouped Cube weights (aclnnQuantGroupedMatmulDequantWeightNZ), falling back to 8-core vectorized
//     planar Q2_0 only for the final few layers if HBM headroom requires it.
//   - Host DDR4 RAM caches pre-packed FRACTAL_NZ dense/expert weights (/dev/shm tmpfs) and pre-dequantized FP32 token
//     embeddings, with DMA-pinned host staging (aclrtMallocHost) and NUMA-local thread affinity.

#include "ascend_ops.h"
#include "geo.h"
#include "gguf_lite.h"
#include "ref_quant.h"

#include "acl/acl.h"
#include "aclnnop/aclnn_mm.h"
#include "aclnnop/aclnn_quant_grouped_matmul_dequant_weight_nz.h"
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

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
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
constexpr int PROMPT_CHUNK_DEFAULT = 256;
constexpr int MAX_ROWS = 256;  // full-batch NPU pass for C=1..256 prompt chunks and T=1..8 decode/verify

constexpr int W_HC_DN_COLS = 336;   // 320 + 4 + 12 pad (multiple of 16)
constexpr int GDN_IN_COLS  = 16480; // 10240 + 6144 + 48 + 48
constexpr int QSA_IN_COLS  = 13952; // 12288 + 512 + 512 + 512 + 128
constexpr int MOE_IN_COLS  = 1808;  // 512 + 640 + 640 + 16
constexpr int QIDS_STRIDE  = 2056;  // >= BUD(2051), multiple of 8 ints (32B)

constexpr int DF2_RING    = 2064;
constexpr int DF2_ROWS    = 8;
constexpr int DF2_DRAFTS  = 7;
constexpr int DF2_CTX_MAX = MAX_ROWS; // 256-row Cube batch for prompt context updates

std::string g_df2_pack_path;

double now_ms() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

bool check_acl(aclError e, const char* what, std::string& err) {
    if (e == ACL_SUCCESS) return true;
    err = std::string("ascend window: ") + what + " failed (aclError=" + std::to_string((int)e) + ")";
    return false;
}

// Unlock all 0..63 host CPUs during parallel 64-core startup (generate.cpp pins the caller to CPU 0 beforehand!)
void unpin_all_cpus() {
#ifdef __linux__
    cpu_set_t all;
    CPU_ZERO(&all);
    const unsigned ncpu = std::max(64u, std::thread::hardware_concurrency());
    for (unsigned c = 0; c < std::min(ncpu, (unsigned)CPU_SETSIZE); ++c) CPU_SET((int)c, &all);
    (void)sched_setaffinity(0, sizeof(all), &all);
#endif
}

// Pin the main engine host thread to NUMA Node 1 (CPUs 32..63, local to PCIe 0000:8x:00.0 NPUs on TaiShan 200)
void pin_host_thread_numa1() {
#ifdef __linux__
    const char* no_pin = std::getenv("STRATA_ASCEND_NO_NUMA_PIN");
    if (no_pin && std::string(no_pin) == "1") return;
    cpu_set_t target;
    CPU_ZERO(&target);
    for (int c = 32; c < 64; ++c) CPU_SET(c, &target);
    (void)sched_setaffinity(0, sizeof(target), &target);
#endif
}

// DMA-pinned host staging buffer via aclrtMallocHost (with 64B-aligned fallback)
template <typename T>
struct HostPinned {
    T* ptr = nullptr;
    size_t count = 0;
    bool acl_pinned = false;

    HostPinned() = default;
    HostPinned(const HostPinned&) = delete;
    HostPinned& operator=(const HostPinned&) = delete;
    ~HostPinned() { reset(); }

    void reset() {
        if (ptr) {
            if (acl_pinned) (void)aclrtFreeHost(ptr);
            else std::free(ptr);
            ptr = nullptr;
            count = 0;
            acl_pinned = false;
        }
    }

    bool alloc(size_t n) {
        reset();
        if (n == 0) return true;
        const size_t bytes = ((n * sizeof(T) + 63) & ~size_t(63)) + 64;
        void* p = nullptr;
        if (aclrtMallocHost(&p, bytes) == ACL_SUCCESS && p != nullptr) {
            ptr = reinterpret_cast<T*>(p);
            count = n;
            acl_pinned = true;
            std::memset(ptr, 0, bytes);
            return true;
        }
        if (posix_memalign(&p, 64, bytes) == 0 && p != nullptr) {
            ptr = reinterpret_cast<T*>(p);
            count = n;
            acl_pinned = false;
            std::memset(ptr, 0, bytes);
            return true;
        }
        return false;
    }

    T* data() { return ptr; }
    const T* data() const { return ptr; }
    T& operator[](size_t i) { return ptr[i]; }
    const T& operator[](size_t i) const { return ptr[i]; }
};

// RAM-backed binary cache (/dev/shm/chaomei_strata_nz) for instant zero-repack startup
struct CacheStore {
    std::string dir;
    bool enabled = false;

    void init(size_t shard_bytes, int nexp) {
        const char* env = std::getenv("STRATA_ASCEND_CACHE_DIR");
        std::string root = (env && env[0] != '\0') ? std::string(env) : std::string("/dev/shm/chaomei_strata_nz");
        if (root == "0" || root == "off" || root == "none") {
            enabled = false;
            return;
        }
        (void)::mkdir(root.c_str(), 0700);
        dir = root + "/s_" + std::to_string(shard_bytes) + "_e" + std::to_string(nexp);
        if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
            enabled = false;
            return;
        }
        enabled = true;
    }

    bool has_space_for(size_t bytes) const {
        if (!enabled) return false;
        struct statvfs vfs{};
        if (::statvfs(dir.c_str(), &vfs) != 0) return false;
        const uint64_t avail = (uint64_t)vfs.f_bavail * (uint64_t)vfs.f_frsize;
        // Always leave >= 32 GiB free in /dev/shm so system/other processes are never starved
        const uint64_t reserve = 32ULL << 30;
        return avail >= (uint64_t)bytes + reserve;
    }

    std::string path_for(const std::string& key) const {
        return dir + "/" + key;
    }

    static bool read_exact_fd(int fd, void* dst, size_t bytes) {
        uint8_t* p = reinterpret_cast<uint8_t*>(dst);
        size_t off = 0;
        while (off < bytes) {
            const size_t chunk = std::min<size_t>(bytes - off, 1ULL << 28);
            ssize_t r = ::pread(fd, p + off, chunk, (off_t)off);
            if (r <= 0) return false;
            off += (size_t)r;
        }
        return true;
    }

    static bool write_exact_fd(int fd, const void* src, size_t bytes) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
        size_t off = 0;
        while (off < bytes) {
            const size_t chunk = std::min<size_t>(bytes - off, 1ULL << 28);
            ssize_t w = ::write(fd, p + off, chunk);
            if (w <= 0) return false;
            off += (size_t)w;
        }
        return true;
    }

    bool load_blob(const std::string& key, void* dst, size_t bytes) const {
        if (!enabled) return false;
        const std::string p = path_for(key);
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        bool ok = (::fstat(fd, &st) == 0 && (size_t)st.st_size == bytes && read_exact_fd(fd, dst, bytes));
        ::close(fd);
        return ok;
    }

    void save_blob(const std::string& key, const void* src, size_t bytes) const {
        if (!has_space_for(bytes)) return;
        const std::string final_p = path_for(key);
        struct stat st{};
        if (::stat(final_p.c_str(), &st) == 0 && (size_t)st.st_size == bytes) return;
        const std::string tmp_p = final_p + ".tmp." + std::to_string((int)::getpid());
        int fd = ::open(tmp_p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) return;
        bool ok = write_exact_fd(fd, src, bytes);
        ::close(fd);
        if (ok) {
            if (::rename(tmp_p.c_str(), final_p.c_str()) != 0) (void)::unlink(tmp_p.c_str());
        } else {
            (void)::unlink(tmp_p.c_str());
        }
    }
};

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

    bool upload_cached_nz(int dev, const uint16_t* host_nz, int64_t k, int64_t n, std::string& err) {
        K = k;
        N_cols = n;
        const size_t sz = (size_t)K * (size_t)N_cols * sizeof(uint16_t);
        if (!mem.alloc(dev, sz, err, false)) return false;
        if (!mem.upload(host_nz, sz, err)) return false;
        int64_t shape[2] = {K, N_cols};
        int64_t stride[2] = {N_cols, 1};
        desc = aclCreateTensor(shape, 2, ACL_FLOAT16, stride, 0, ACL_FORMAT_FRACTAL_NZ, shape, 2, mem.ptr);
        return desc != nullptr;
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

// Static 280-TOPS INT8 FRACTAL_NZ Grouped Expert Weight [E, N_out, K] -> storage [E, K/32, N_out/16, 16, 32] + scale [E, N_out]
struct Int8GroupNzMat {
    DevMem w_nz;
    DevMem w_scale;
    int64_t E = 0;
    int64_t N_out = 0;
    int64_t K = 0;
    aclTensor* desc_w = nullptr;
    aclTensor* desc_scale = nullptr;

    Int8GroupNzMat() = default;
    Int8GroupNzMat(const Int8GroupNzMat&) = delete;
    Int8GroupNzMat& operator=(const Int8GroupNzMat&) = delete;
    Int8GroupNzMat(Int8GroupNzMat&& o) noexcept
        : w_nz(std::move(o.w_nz)), w_scale(std::move(o.w_scale)),
          E(o.E), N_out(o.N_out), K(o.K), desc_w(o.desc_w), desc_scale(o.desc_scale) {
        o.desc_w = nullptr;
        o.desc_scale = nullptr;
    }
    Int8GroupNzMat& operator=(Int8GroupNzMat&& o) noexcept {
        if (this != &o) {
            if (desc_w) aclDestroyTensor(desc_w);
            if (desc_scale) aclDestroyTensor(desc_scale);
            w_nz = std::move(o.w_nz);
            w_scale = std::move(o.w_scale);
            E = o.E;
            N_out = o.N_out;
            K = o.K;
            desc_w = o.desc_w;
            desc_scale = o.desc_scale;
            o.desc_w = nullptr;
            o.desc_scale = nullptr;
        }
        return *this;
    }
    ~Int8GroupNzMat() {
        if (desc_w) aclDestroyTensor(desc_w);
        if (desc_scale) aclDestroyTensor(desc_scale);
    }

    bool upload_prepacked(int dev, const int8_t* host_nz, const float* host_scale,
                          int64_t e, int64_t n_out, int64_t k, std::string& err) {
        E = e;
        N_out = n_out;
        K = k;
        const size_t w_sz = (size_t)E * (size_t)N_out * (size_t)K;
        const size_t s_sz = (size_t)E * (size_t)N_out * sizeof(float);
        if (!w_nz.alloc(dev, w_sz, err, false) || !w_nz.upload(host_nz, w_sz, err)) return false;
        if (!w_scale.alloc(dev, s_sz, err, false) || !w_scale.upload(host_scale, s_sz, err)) return false;

        int64_t wView[3] = {E, N_out, K};
        int64_t wStr[3]  = {N_out * K, K, 1};
        int64_t wStor[5] = {E, K / 32, N_out / 16, 16, 32};
        int64_t sDim[2]  = {E, N_out};
        int64_t sStr[2]  = {N_out, 1};

        desc_w = aclCreateTensor(wView, 3, ACL_INT8, wStr, 0, ACL_FORMAT_FRACTAL_NZ, wStor, 5, w_nz.ptr);
        desc_scale = aclCreateTensor(sDim, 2, ACL_FLOAT, sStr, 0, ACL_FORMAT_ND, sDim, 2, w_scale.ptr);
        return desc_w != nullptr && desc_scale != nullptr;
    }
};

// Column-chunked INT8 FRACTAL_NZ matrix for >64K vocabulary heads (avoids CANN dav-m200 uint16 byte-stride overflow at N_out > 65536!)
struct Int8ChunkedMat {
    static constexpr int64_t MAX_CHUNK_COLS = 62080; // 248320 / 4 = 62080 (multiple of 128, < 65536!)
    std::vector<Int8GroupNzMat> chunks;
    DevMem stitch_tmp; // [MAXT, total_cols] f16 scratch when M > 1 && chunks.size() > 1
    int64_t K = 0;
    int64_t chunk_cols = 0;
    int64_t total_cols = 0;
};

struct LayerWeights {
    int dev = 0;
    bool is_qsa = false;
    bool use_int8_cube = false;

    // Hyper-connection attn & ffn
    DevMem hc_attn_norm;   // [4 * 2560] f32
    NzMat  W_hc_attn_dn;   // [10240, 336] f16 NZ
    NzMat  W_hc_attn_up;   // [320, 10240] f16 NZ

    DevMem hc_ffn_norm;    // [4 * 2560] f32
    NzMat  W_hc_ffn_dn;    // [10240, 336] f16 NZ
    NzMat  W_hc_ffn_up;    // [320, 10240] f16 NZ

    // GDN weights (when !is_qsa): 280-TOPS INT8 FRACTAL_NZ Cube QGM(E=1)
    Int8GroupNzMat W_gdn_in;   // [1, 16480, 2560] INT8 NZ
    Int8GroupNzMat W_ssm_out;  // [1, 2560, 6144]  INT8 NZ
    DevMem ssm_conv_w;         // [10240 * 4] f32
    DevMem ssm_dt;             // [64] f32 (first 48 used)
    DevMem ssm_a;              // [64] f32 (first 48 used)
    DevMem ssm_norm;           // [128] f32

    // QSA weights (when is_qsa): 280-TOPS INT8 FRACTAL_NZ Cube QGM(E=1)
    Int8GroupNzMat W_qsa_in;   // [1, 13952, 2560] INT8 NZ
    Int8GroupNzMat W_attn_out; // [1, 2560, 6144]  INT8 NZ
    DevMem q_norm;             // [256] f32
    DevMem k_norm;             // [256] f32
    DevMem iq_norm;            // [128] f32
    DevMem ik_norm;            // [128] f32

    // MoE router + shared expert
    NzMat  W_moe_in;       // [2560, 1808] f16 NZ
    NzMat  W_shexp_down;   // [640, 2560] f16 NZ

    // Primary 280-TOPS INT8 FRACTAL_NZ Cube routed experts (when use_int8_cube == true)
    Int8GroupNzMat W_moe_w1; // [nexp, 1280, 2560] INT8 NZ ([gate_640 | up_640])
    Int8GroupNzMat W_moe_w2; // [nexp, 2560, 640]  INT8 NZ (down)

    // Fallback planar Q2_0 routed experts (when use_int8_cube == false)
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
    DevMem ws;             // 128 MB Cube workspace
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

    // MoE shared + INT8 Cube + Q2_0 scratch
    DevMem moe_in_f16;         // [MAX_ROWS, 1808] f16
    DevMem ids_i32;            // [MAX_ROWS, 16] i32
    DevMem wts_f32;            // [MAX_ROWS, 16] f32
    DevMem sh_h_f16;           // [MAX_ROWS, 640] f16
    DevMem ysh_f16;            // [MAX_ROWS, 2560] f16

    // INT8 Cube MoE scratch (supports full MAX_ROWS * 10 = 2560 routed rows!)
    DevMem group_cumsum_i64;   // [512] i64
    aclTensor* desc_group_cumsum = nullptr;
    DevMem d_one_m;            // [(MAX_ROWS + 1) * 4] i64: pre-staged single-group M = 1..MAX_ROWS (32B aligned per M!)
    aclTensor* desc_one_m[MAX_ROWS + 1] = {};
    DevMem d_tap;              // [n_taps_on_dev * MAX_ROWS * 2560] f32 (allocated lazily when DFlash2 taps are enabled)
    DevMem sorted_tok_i32;     // [MAX_ROWS * TOPK + 8] i32
    DevMem tok_perm_i32;       // [MAX_ROWS, 16] i32
    DevMem sorted_x_f16;       // [MAX_ROWS * TOPK, 2560] f16
    DevMem moe_gu_f16;         // [MAX_ROWS * TOPK, 1280] f16
    DevMem moe_h_f16;          // [MAX_ROWS * TOPK, 640] f16
    DevMem moe_ye_f16;         // [MAX_ROWS * TOPK, 2560] f16

    // Fallback Q2_0 MoE scratch (8-row windows)
    DevMem plan_i32;           // [P_INTS] i32
    DevMem sg_f32;             // [64] f32
    DevMem h_f32;              // [128, 640] f32
    DevMem ye_f32;             // [128, 2560] f32

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

    bool qgm_with_glist(const void* x_ptr, const Int8GroupNzMat& W, aclTensor* glist_desc,
                        void* out_ptr, int64_t M, std::string& err) {
        if (!check_acl(aclrtSetDevice(dev), "aclrtSetDevice", err)) return false;
        int64_t xShape[2] = {M, W.K}, xStride[2] = {W.K, 1};
        int64_t oShape[2] = {M, W.N_out}, oStride[2] = {W.N_out, 1};
        aclTensor* tX = aclCreateTensor(xShape, 2, ACL_FLOAT16, xStride, 0, ACL_FORMAT_ND, xShape, 2, const_cast<void*>(x_ptr));
        aclTensor* tO = aclCreateTensor(oShape, 2, ACL_FLOAT16, oStride, 0, ACL_FORMAT_ND, oShape, 2, out_ptr);
        temp_tensors.push_back(tX);
        temp_tensors.push_back(tO);

        char qmode[] = "pertoken";
        uint64_t ws_need = 0;
        aclOpExecutor* exec = nullptr;
        if (!check_acl(aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize(
                           tX, W.desc_w, W.desc_scale, glist_desc,
                           nullptr, nullptr, nullptr, nullptr, qmode, true, tO, &ws_need, &exec),
                       "aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize", err)) {
            return false;
        }
        if (ws_need > ws.bytes) {
            if (!check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(grow ws qgm)", err)) return false;
            if (!ws.alloc(dev, std::max<size_t>((size_t)ws_need, ws.bytes * 2), err, false)) return false;
        }
        return check_acl(aclnnQuantGroupedMatmulDequantWeightNZ(ws.ptr, ws_need, exec, stream),
                         "aclnnQuantGroupedMatmulDequantWeightNZ", err);
    }

    bool qgm(const void* x_ptr, const Int8GroupNzMat& W, void* out_ptr, int64_t M, std::string& err) {
        return qgm_with_glist(x_ptr, W, desc_group_cumsum, out_ptr, M, err);
    }

    bool qgm1(const void* x_ptr, const Int8GroupNzMat& W, void* out_ptr, int64_t M, std::string& err) {
        if (M < 1 || M > MAX_ROWS) {
            err = "ascend window: qgm1 M out of range";
            return false;
        }
        return qgm_with_glist(x_ptr, W, desc_one_m[M], out_ptr, M, err);
    }

    bool qgm1_chunked(const void* x_ptr, const Int8ChunkedMat& W, void* out_ptr, int64_t M, std::string& err) {
        const size_t nc = W.chunks.size();
        if (nc == 1) {
            return qgm1(x_ptr, W.chunks[0], out_ptr, M, err);
        }
        if (M == 1) {
            // M == 1: chunks write directly into contiguous slices of out_ptr [1, total_cols] with zero stitch!
            for (size_t s = 0; s < nc; ++s) {
                uint16_t* dst = reinterpret_cast<uint16_t*>(out_ptr) + s * (size_t)W.chunk_cols;
                if (!qgm1(x_ptr, W.chunks[s], dst, 1, err)) return false;
            }
            return true;
        }
        for (size_t s = 0; s < nc; ++s) {
            uint16_t* dst = reinterpret_cast<uint16_t*>(W.stitch_tmp.ptr) + s * (size_t)M * (size_t)W.chunk_cols;
            if (!qgm1(x_ptr, W.chunks[s], dst, M, err)) return false;
        }
        strata::ascend_ops::launch_stitch_cols_f16(
            stream, W.stitch_tmp.ptr, out_ptr, (uint32_t)M, (uint32_t)W.chunk_cols, (uint32_t)nc);
        return true;
    }

    ~NpuCtx() {
        clear_temp_tensors();
        for (int m = 0; m <= MAX_ROWS; ++m) {
            if (desc_one_m[m]) {
                aclDestroyTensor(desc_one_m[m]);
                desc_one_m[m] = nullptr;
            }
        }
        if (desc_group_cumsum) {
            aclDestroyTensor(desc_group_cumsum);
            desc_group_cumsum = nullptr;
        }
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

// Parallel L1-tiled dequantize + transpose of GGUF matrices sharing input width K into row-major [K, N_total] FP16
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

// Parallel 64-core packer for fused dense/draft matrices into 280-TOPS INT8 FRACTAL_NZ [1, K/32, N_out/16, 16, 32] + FP32 scale [1, N_out]
void pack_fused_int8_nz(const uint8_t* base_map, int64_t K, int64_t N_total,
                        const std::vector<MatSlice>& slices,
                        std::vector<int8_t>& out_nz, std::vector<float>& out_sc, unsigned nth) {
    out_nz.assign((size_t)K * (size_t)N_total, 0);
    out_sc.assign((size_t)N_total, 0.0f);
    const size_t n_blocks = (size_t)(N_total >> 4);
    const size_t k_blocks = (size_t)(K >> 5);

    for (const MatSlice& sl : slices) {
        if (!sl.t || sl.rows <= 0) continue;
        const size_t rb = ref::row_bytes(sl.t->type, (int)K);
        const uint8_t* src_base = base_map + sl.t->offset;
        const int64_t R = sl.rows;
        const int64_t col0 = sl.col_offset;
        const unsigned workers = std::max(1u, std::min<unsigned>(nth, (unsigned)R));
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (unsigned tid = 0; tid < workers; ++tid) {
            threads.emplace_back([&, tid] {
                std::vector<float> row_f32((size_t)K);
                std::vector<int8_t> row_i8((size_t)K);
                const int64_t r0 = (R * tid) / workers;
                const int64_t r1 = (R * (tid + 1)) / workers;
                for (int64_t r = r0; r < r1; ++r) {
                    ref::dequant(sl.t->type, src_base + (size_t)r * rb, row_f32.data(), (size_t)K);
                    float mx = 0.0f;
                    for (int64_t k = 0; k < K; ++k) mx = std::max(mx, std::fabs(row_f32[(size_t)k]));
                    const float sc = mx > 0.0f ? (mx / 127.0f) : 1.0f;
                    const float inv_sc = 1.0f / sc;
                    const size_t out_r = (size_t)(col0 + r);
                    out_sc[out_r] = sc;
                    for (int64_t k = 0; k < K; ++k) {
                        int q = (int)std::nearbyintf(row_f32[(size_t)k] * inv_sc);
                        row_i8[(size_t)k] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
                    }
                    const size_t r_hi = out_r >> 4, r_lo = out_r & 15;
                    for (size_t kh = 0; kh < k_blocks; ++kh) {
                        const size_t off = (((kh * n_blocks + r_hi) * 16 + r_lo) * 32);
                        std::memcpy(out_nz.data() + off, row_i8.data() + kh * 32, 32);
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
    }
}

bool df2_read_index(const std::string& path, std::vector<int>& taps, std::map<std::string, gguf::Tensor>& ten,
                    uint64_t& total, uint64_t& data_off, std::map<std::string, std::string>& cfg, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "dflash2: cannot open " + path; return false; }
    struct Closer { std::FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::memcmp(hdr, "DF2P", 4) != 0 || hdr[1] != 1) {
        err = "dflash2: " + path + " is not a DF2P v1 pack";
        return false;
    }
    std::string index(hdr[2], ' ');
    if (std::fread(index.data(), 1, hdr[2], f) != hdr[2]) {
        err = "dflash2: the pack's index is cut short";
        return false;
    }
    data_off = 12 + hdr[2];
    total = 0;
    for (size_t i = 0; i < index.size();) {
        const size_t j = index.find('\n', i);
        if (j == std::string::npos) break;
        const std::string line = index.substr(i, j - i);
        i = j + 1;
        char a[256], b[64], c[256];
        long long v1 = 0, v2 = 0, v3 = 0, v4 = 0;
        if (std::sscanf(line.c_str(), "cfg %63s %255s", b, c) == 2) {
            cfg[b] = c;
            if (std::string(b) == "taps") {
                taps.clear();
                for (const char* q = c; *q;) {
                    taps.push_back(std::atoi(q));
                    while (*q && *q != ',') ++q;
                    if (*q == ',') ++q;
                }
            }
        } else if (std::sscanf(line.c_str(), "t %255s %63s %lld %lld %lld %lld", a, b, &v1, &v2, &v3, &v4) == 6) {
            gguf::Tensor t;
            t.name = a;
            const std::string dt = b;
            t.type = dt == "q8_0" ? ref::Q8_0 : dt == "q4_0" ? ref::Q4_0 : dt == "f32" ? ref::F32 : dt == "bf16" ? ref::BF16 : -1;
            t.shape = {(uint64_t)v2, (uint64_t)v1}; // ggml order: [K_in, N_out]
            t.offset = (uint64_t)v3;
            t.bytes = (uint64_t)v4;
            ten[t.name] = t;
            total = std::max(total, t.offset + t.bytes);
        }
    }
    if (ten.empty()) {
        err = "dflash2: the pack has no tensors";
        return false;
    }
    return true;
}

void df2_rope_rows(const int32_t* pos, int rows, float theta, float* table) {
    constexpr double two_pi = 6.283185307179586476925286766559;
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < 64; ++i) {
            const double inv = std::pow((double)theta, -(double)(2 * i) / 128.0);
            double ang = (double)pos[r] * inv;
            ang -= two_pi * std::rint(ang / two_pi);
            table[(r * 2 + 0) * 64 + i] = (float)std::cos(ang);
            table[(r * 2 + 1) * 64 + i] = (float)std::sin(ang);
        }
    }
}

void dequant_cb_row256(int cb_type, const uint8_t* cb_base, int32_t tok_id, float* dst256) {
    const size_t rb = ref::row_bytes(cb_type, 256);
    ref::dequant(cb_type, cb_base + (size_t)std::max(0, tok_id) * rb, dst256, 256);
}

// Parallel 64-core Q2_0 -> INT8 FRACTAL_NZ [E, K/32, N_out/16, 16, 32] + FP32 scale [E, N_out] packer
void pack_q2_experts_int8_nz(const uint8_t* gate_base, const uint8_t* up_base, const uint8_t* down_base, int nexp,
                             std::vector<int8_t>& w1_nz, std::vector<float>& w1_sc,
                             std::vector<int8_t>& w2_nz, std::vector<float>& w2_sc, unsigned nth) {
    constexpr int GU = 2 * NFF; // 1280
    constexpr size_t gu_row_bytes = (N / 64) * 18;    // 720 B
    constexpr size_t dn_row_bytes = (NFF / 64) * 18;  // 180 B

    w1_nz.resize((size_t)nexp * (size_t)GU * (size_t)N);
    w1_sc.resize((size_t)nexp * (size_t)GU);
    w2_nz.resize((size_t)nexp * (size_t)N * (size_t)NFF);
    w2_sc.resize((size_t)nexp * (size_t)N);

    const unsigned workers = std::max(1u, std::min<unsigned>(nth, (unsigned)nexp));
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned tid = 0; tid < workers; ++tid) {
        threads.emplace_back([&, tid] {
            std::vector<float> row_f32((size_t)N);
            std::vector<int8_t> row_i8((size_t)N);

            auto pack_row = [](int8_t* dst, int e, int r, int N_out, int K_in, const int8_t* src) {
                const size_t r_hi = (size_t)(r >> 4), r_lo = (size_t)(r & 15);
                const size_t n_blocks = (size_t)(N_out >> 4), k_blocks = (size_t)(K_in >> 5);
                for (size_t kh = 0; kh < k_blocks; ++kh) {
                    const size_t off = ((((size_t)e * k_blocks + kh) * n_blocks + r_hi) * 16 + r_lo) * 32;
                    std::memcpy(dst + off, src + kh * 32, 32);
                }
            };

            for (int e = (int)tid; e < nexp; e += (int)workers) {
                for (int mat = 0; mat < 2; ++mat) {
                    const uint8_t* src = (mat == 0 ? gate_base : up_base) + (size_t)e * NFF * gu_row_bytes;
                    for (int r = 0; r < NFF; ++r) {
                        ref::dequant(ref::Q2_0, src + (size_t)r * gu_row_bytes, row_f32.data(), N);
                        float mx = 0.0f;
                        for (int k = 0; k < N; ++k) mx = std::max(mx, std::fabs(row_f32[(size_t)k]));
                        const float sc = mx > 0.0f ? (mx / 127.0f) : 1.0f;
                        const float inv_sc = 1.0f / sc;
                        const int out_r = mat * NFF + r;
                        w1_sc[(size_t)e * GU + (size_t)out_r] = sc;
                        for (int k = 0; k < N; ++k) {
                            int q = (int)std::nearbyintf(row_f32[(size_t)k] * inv_sc);
                            row_i8[(size_t)k] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
                        }
                        pack_row(w1_nz.data(), e, out_r, GU, N, row_i8.data());
                    }
                }

                const uint8_t* dsrc = down_base + (size_t)e * N * dn_row_bytes;
                for (int r = 0; r < N; ++r) {
                    ref::dequant(ref::Q2_0, dsrc + (size_t)r * dn_row_bytes, row_f32.data(), NFF);
                    float mx = 0.0f;
                    for (int k = 0; k < NFF; ++k) mx = std::max(mx, std::fabs(row_f32[(size_t)k]));
                    const float sc = mx > 0.0f ? (mx / 127.0f) : 1.0f;
                    const float inv_sc = 1.0f / sc;
                    w2_sc[(size_t)e * N + (size_t)r] = sc;
                    for (int k = 0; k < NFF; ++k) {
                        int q = (int)std::nearbyintf(row_f32[(size_t)k] * inv_sc);
                        row_i8[(size_t)k] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
                    }
                    pack_row(w2_nz.data(), e, r, N, NFF, row_i8.data());
                }
            }
        });
    }
    for (auto& th : threads) th.join();
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

std::string draft_vocab_default();

class WindowAscend {
public:
    bool ready() const { return ready_; }
    int64_t vocab() const { return n_vocab_; }
    void set_max_t(int t) { max_t_ = std::max(1, std::min(t, MAXT)); }

    bool init(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, const std::string& shard1,
              int max_t, int64_t max_cells, std::string& err) {
        unpin_all_cpus();
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
        int max_npus = (int)std::min<uint32_t>(dev_cnt, 8u);
        if (const char* em = std::getenv("STRATA_ASCEND_MAX_NPUS")) {
            int req = std::atoi(em);
            if (req >= 1 && req < max_npus) max_npus = req;
        }
        n_npu_ = force_single ? 1 : max_npus;

        max_cells_ = std::max<int64_t>(4096, ((max_cells + 63) / 64) * 64 + 64);
        max_blocks_ = ((max_cells_ / 4 + 16 + 7) / 8) * 8;  // multiple of 8 floats (32B)

        const gguf::Tensor* t_out = f_.find("output.weight");
        if (!t_out || t_out->shape.size() < 2) {
            err = "ascend window: missing output.weight in " + shard1;
            return false;
        }
        n_vocab_ = (int64_t)t_out->shape[1];
        vocab_pad_ = ((n_vocab_ + 127) / 128) * 128;

        CacheStore cache;
        cache.init(f_.file_bytes, nexp_);

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

        // Initialize per-NPU streams & scratch buffers (Pass 1a)
        std::vector<int64_t> h_one_m((size_t)(MAX_ROWS + 1) * 4, 0);
        for (int m = 1; m <= MAX_ROWS; ++m) h_one_m[(size_t)m * 4] = (int64_t)m;

        npu_.resize((size_t)n_npu_);
        for (int d = 0; d < n_npu_; ++d) {
            npu_[d] = std::make_unique<NpuCtx>();
            NpuCtx& c = *npu_[d];
            c.dev = d;
            if (!check_acl(aclrtSetDevice(d), "aclrtSetDevice", err) ||
                !check_acl(aclrtCreateStream(&c.stream), "aclrtCreateStream", err)) return false;
            if (!c.ws.alloc(d, 16ULL << 20, err, false) ||
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
                !c.xa_f32.alloc(d, 64, err) ||
                !c.xf_f16.alloc(d, (size_t)MAX_ROWS * N * 2, err) ||
                !c.xf_f32.alloc(d, (size_t)MAX_ROWS * N * 4, err) ||
                !c.gdn_in_f16.alloc(d, (size_t)MAX_ROWS * GDN_IN_COLS * 2, err) ||
                !c.yg_f16.alloc(d, (size_t)MAX_ROWS * ZV * 2, err) ||
                !c.qsa_in_f16.alloc(d, (size_t)MAX_ROWS * QSA_IN_COLS * 2, err) ||
                !c.qcur_f32.alloc(d, (size_t)MAX_ROWS * QOUT * 4, err) ||
                !c.qgate_f32.alloc(d, (size_t)MAX_ROWS * QOUT * 4, err) ||
                !c.qidx_f32.alloc(d, (size_t)MAX_ROWS * IQR * 4, err) ||
                !c.scores_f32.alloc(d, 64, err) ||
                !c.qids_i32.alloc(d, (size_t)MAX_ROWS * QIDS_STRIDE * 4, err) ||
                !c.xq_f16.alloc(d, (size_t)MAX_ROWS * QOUT * 2, err) ||
                !c.moe_in_f16.alloc(d, (size_t)MAX_ROWS * MOE_IN_COLS * 2, err) ||
                !c.ids_i32.alloc(d, (size_t)MAX_ROWS * 16 * 4, err) ||
                !c.wts_f32.alloc(d, (size_t)MAX_ROWS * 16 * 4, err) ||
                !c.sh_h_f16.alloc(d, (size_t)MAX_ROWS * NFF * 2, err) ||
                !c.ysh_f16.alloc(d, (size_t)MAX_ROWS * N * 2, err) ||
                !c.group_cumsum_i64.alloc(d, 512 * 8, err) ||
                !c.d_one_m.alloc(d, h_one_m.size() * sizeof(int64_t), err, false) ||
                !c.d_one_m.upload(h_one_m.data(), h_one_m.size() * sizeof(int64_t), err) ||
                !c.sorted_tok_i32.alloc(d, (size_t)(MAX_ROWS * TOPK + 8) * 4, err) ||
                !c.tok_perm_i32.alloc(d, (size_t)MAX_ROWS * 16 * 4, err) ||
                !c.sorted_x_f16.alloc(d, (size_t)MAX_ROWS * TOPK * N * 2, err) ||
                !c.moe_gu_f16.alloc(d, (size_t)MAX_ROWS * TOPK * (2 * NFF) * 2, err) ||
                !c.moe_h_f16.alloc(d, (size_t)MAX_ROWS * TOPK * NFF * 2, err) ||
                !c.moe_ye_f16.alloc(d, (size_t)MAX_ROWS * TOPK * N * 2, err) ||
                !c.plan_i32.alloc(d, strata::ascend_ops::P_INTS * 4, err) ||
                !c.sg_f32.alloc(d, ((MAX_ROWS + 7) / 8) * 8 * 4, err) ||
                !c.h_f32.alloc(d, 128 * NFF * 4, err) ||
                !c.ye_f32.alloc(d, 128 * N * 4, err)) {
                return false;
            }
            int64_t gDim[1] = {nexp_}, gStr[1] = {1};
            c.desc_group_cumsum = aclCreateTensor(gDim, 1, ACL_INT64, gStr, 0, ACL_FORMAT_ND, gDim, 1, c.group_cumsum_i64.ptr);
            if (!c.desc_group_cumsum) {
                err = "ascend window: aclCreateTensor(group_cumsum) failed";
                return false;
            }
            int64_t oneDim[1] = {1}, oneStr[1] = {1};
            for (int m = 1; m <= MAX_ROWS; ++m) {
                c.desc_one_m[m] = aclCreateTensor(
                    oneDim, 1, ACL_INT64, oneStr, 0, ACL_FORMAT_ND, oneDim, 1,
                    reinterpret_cast<int64_t*>(c.d_one_m.ptr) + (size_t)m * 4);
                if (!c.desc_one_m[m]) {
                    err = "ascend window: aclCreateTensor(desc_one_m) failed";
                    return false;
                }
            }
        }

        const unsigned nth = std::max(1u, std::min(64u, std::thread::hardware_concurrency()));
        const uint8_t* map_base = (const uint8_t*)f_.map;

        // Pre-dequantize token_embd.weight [N, n_vocab_] into Host DDR4 RAM (2.37 GiB) for zero-overhead lookup
        {
            const gguf::Tensor* te = f_.find("token_embd.weight");
            if (!te) { err = "ascend window: missing token_embd.weight"; return false; }
            const size_t emb_elems = (size_t)n_vocab_ * (size_t)N;
            h_tok_embd_f32_.resize(emb_elems);
            if (!cache.load_blob("tok_embd.f32", h_tok_embd_f32_.data(), emb_elems * 4)) {
                const size_t rb = ref::row_bytes(te->type, N);
                const uint8_t* sbase = map_base + te->offset;
                std::vector<std::thread> threads;
                for (unsigned tid = 0; tid < nth; ++tid) {
                    threads.emplace_back([&, tid] {
                        const int64_t r0 = (n_vocab_ * tid) / nth;
                        const int64_t r1 = (n_vocab_ * (tid + 1)) / nth;
                        for (int64_t r = r0; r < r1; ++r) {
                            ref::dequant(te->type, sbase + (size_t)r * rb, h_tok_embd_f32_.data() + (size_t)r * N, N);
                        }
                    });
                }
                for (auto& th : threads) th.join();
                cache.save_blob("tok_embd.f32", h_tok_embd_f32_.data(), emb_elems * 4);
            }
        }

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

        std::vector<uint16_t> h_kn;
        auto load_nz_mat = [&](int dev, const std::string& key, int64_t K_in, int64_t N_cols,
                               const std::vector<MatSlice>& slices, NzMat& mat, uint16_t pad_fill = 0) -> bool {
            NpuCtx& nc = *npu_[dev];
            const size_t sz = (size_t)K_in * (size_t)N_cols * sizeof(uint16_t);
            h_kn.resize((size_t)K_in * (size_t)N_cols);
            if (cache.load_blob(key, h_kn.data(), sz)) {
                return mat.upload_cached_nz(dev, h_kn.data(), K_in, N_cols, err);
            }
            build_fused_kn_f16(map_base, K_in, N_cols, slices, h_kn, nth, pad_fill);
            if (!mat.upload_and_format(dev, nc.stream, nc.ws.ptr, nc.ws.bytes, h_kn.data(), K_in, N_cols, err)) return false;
            if (cache.has_space_for(sz) && mat.mem.download(h_kn.data(), sz, err)) {
                cache.save_blob(key, h_kn.data(), sz);
            }
            err.clear();
            return true;
        };

        std::vector<int8_t> h_i8_nz;
        std::vector<float>  h_i8_sc;
        auto load_int8_mat = [&](int dev, const std::string& key, int64_t K_in, int64_t N_cols,
                                 const std::vector<MatSlice>& slices, Int8GroupNzMat& mat) -> bool {
            const size_t nz_sz = (size_t)K_in * (size_t)N_cols;
            const size_t sc_sz = (size_t)N_cols * sizeof(float);
            h_i8_nz.resize(nz_sz);
            h_i8_sc.resize((size_t)N_cols);
            if (!cache.load_blob(key + ".nz8", h_i8_nz.data(), nz_sz) ||
                !cache.load_blob(key + ".sc",  h_i8_sc.data(), sc_sz)) {
                pack_fused_int8_nz(map_base, K_in, N_cols, slices, h_i8_nz, h_i8_sc, nth);
                if (cache.has_space_for(nz_sz + sc_sz)) {
                    cache.save_blob(key + ".nz8", h_i8_nz.data(), nz_sz);
                    cache.save_blob(key + ".sc",  h_i8_sc.data(), sc_sz);
                }
            }
            return mat.upload_prepacked(dev, h_i8_nz.data(), h_i8_sc.data(), 1, N_cols, K_in, err);
        };

        // PLE setup on NPU 0
        ple_on_ = ss.ple.ready() && f_.find("blk.1.ple_key.weight") != nullptr;
        if (ple_on_) {
            if (!load_f32_vec("blk.1.ple_norm_key.weight", HCD, h_ple_nkey_) ||
                !load_f32_vec("blk.1.ple_norm_query.weight", HCD, h_ple_nqry_) ||
                !load_f32_vec("blk.1.ple_norm_conv.weight", HCD, h_ple_nconv_) ||
                !load_f32_vec("blk.1.ple_conv1d.weight", (size_t)strata::kernels::PLE_CONV_KERNEL * HCD, h_ple_conv_w_)) {
                return false;
            }
            for (float& v : h_ple_conv_w_) v = ref::f16_to_f32(ref::f32_to_f16(v));

            if (!load_nz_mat(0, "ple_kv.nz16", N, HCD + N,
                             {{f_.find("blk.1.ple_key.weight"), 0, HCD},
                              {f_.find("blk.1.ple_value.weight"), HCD, N}},
                             W_ple_kv_) ||
                !d_ple_emb_f32_.alloc(0, (size_t)MAX_ROWS * N * 4, err) ||
                !d_ple_emb_f16_.alloc(0, (size_t)MAX_ROWS * N * 2, err) ||
                !d_ple_kv_f16_.alloc(0, (size_t)MAX_ROWS * (HCD + N) * 2, err)) {
                return false;
            }
        }
        ple_hist_.assign((size_t)strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM, 0.0f);
        last_norm_.assign((size_t)MAX_ROWS * strata::kernels::NG_HC_DIM, 0.0f);

        // Assign balanced layer partition across n_npu_ devices
        layers_.resize(NL);
        gdn_states_.resize(NGDN);
        qsa_states_.resize(NQSA);
        std::vector<int> dev_layer_cnt((size_t)n_npu_, 0);
        for (int l = 0; l < NL; ++l) {
            const int d = std::min(n_npu_ - 1, (l * n_npu_) / NL);
            layers_[(size_t)l].dev = d;
            layers_[(size_t)l].is_qsa = (l % QI == QI - 1);
            dev_layer_cnt[(size_t)d] += 1;
        }

        std::fprintf(stderr, "strata ascend: Pass 1/2 loading dense FRACTAL_NZ + states onto %d NPU(s) (%u CPU threads, cache=%s)...\n",
                     n_npu_, nth, cache.enabled ? cache.dir.c_str() : "off");
        std::fflush(stderr);

        // Pass 1b: Allocate ALL dense weights + GDN/QSA states + Output Head FIRST so exact HBM headroom is known!
        for (int l = 0; l < NL; ++l) {
            LayerWeights& lw = layers_[(size_t)l];
            const int dev = lw.dev;
            const std::string b = "blk." + std::to_string(l) + ".";
            const std::string pfx = "l" + std::to_string(l) + "_";

            if (!upload_f32_dev(dev, b + "hc_attn_norm.weight", HCD, lw.hc_attn_norm) ||
                !upload_f32_dev(dev, b + "hc_ffn_norm.weight", HCD, lw.hc_ffn_norm)) return false;

            if (!load_nz_mat(dev, pfx + "hc_adn.nz16", HCD, W_HC_DN_COLS,
                             {{f_.find(b + "hc_attn_down.weight"), 0, LR},
                              {f_.find(b + "hc_attn_inject.weight"), LR, HC}}, lw.W_hc_attn_dn) ||
                !load_nz_mat(dev, pfx + "hc_aup.nz16", LR, HCD,
                             {{f_.find(b + "hc_attn_up.weight"), 0, HCD}}, lw.W_hc_attn_up) ||
                !load_nz_mat(dev, pfx + "hc_fdn.nz16", HCD, W_HC_DN_COLS,
                             {{f_.find(b + "hc_ffn_down.weight"), 0, LR},
                              {f_.find(b + "hc_ffn_inject.weight"), LR, HC}}, lw.W_hc_ffn_dn) ||
                !load_nz_mat(dev, pfx + "hc_fup.nz16", LR, HCD,
                             {{f_.find(b + "hc_ffn_up.weight"), 0, HCD}}, lw.W_hc_ffn_up)) return false;

            if (!lw.is_qsa) {
                const int gi = l - l / QI;
                if (!load_int8_mat(dev, pfx + "gdn_in", N, GDN_IN_COLS,
                                   {{f_.find(b + "attn_qkv.weight"),  0,             CONV},
                                    {f_.find(b + "attn_gate.weight"), CONV,          ZV},
                                    {f_.find(b + "ssm_alpha.weight"), CONV + ZV,     HV},
                                    {f_.find(b + "ssm_beta.weight"),  CONV + ZV + HV, HV}}, lw.W_gdn_in) ||
                    !load_int8_mat(dev, pfx + "ssm_out", ZV, N,
                                   {{f_.find(b + "ssm_out.weight"), 0, N}}, lw.W_ssm_out)) return false;

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
                if (!load_int8_mat(dev, pfx + "qsa_in", N, QSA_IN_COLS,
                                   {{f_.find(b + "attn_q.weight"),         0,                    QFULL},
                                    {f_.find(b + "attn_k.weight"),         QFULL,                KV},
                                    {f_.find(b + "attn_v.weight"),         QFULL + KV,           KV},
                                    {f_.find(b + "indexer.q_proj.weight"), QFULL + 2 * KV,       IQR},
                                    {f_.find(b + "indexer.k_proj.weight"), QFULL + 2 * KV + IQR, ID}}, lw.W_qsa_in) ||
                    !load_int8_mat(dev, pfx + "attn_out", QOUT, N,
                                   {{f_.find(b + "attn_output.weight"), 0, N}}, lw.W_attn_out)) return false;

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

            if (!load_nz_mat(dev, pfx + "moe_in.nz16", N, MOE_IN_COLS,
                             {{f_.find(b + "ffn_gate_inp.weight"),       0,             nexp_},
                              {f_.find(b + "ffn_gate_shexp.weight"),     512,           NFF},
                              {f_.find(b + "ffn_up_shexp.weight"),       512 + NFF,     NFF},
                              {f_.find(b + "ffn_gate_inp_shexp.weight"), 512 + 2 * NFF, 1}}, lw.W_moe_in) ||
                !load_nz_mat(dev, pfx + "shexp_dn.nz16", NFF, N,
                             {{f_.find(b + "ffn_down_shexp.weight"), 0, N}}, lw.W_shexp_down)) return false;
        }

        // Output HC + 280-TOPS INT8 Cube Chunked LM Head on final NPU before Pass 2
        const int last_dev = n_npu_ - 1;
        df2_dev_ = std::min(1, n_npu_ - 1);
        {
            const int64_t n_ch = (n_vocab_ + Int8ChunkedMat::MAX_CHUNK_COLS - 1) / Int8ChunkedMat::MAX_CHUNK_COLS;
            const int64_t ch_cols = (((n_vocab_ + n_ch - 1) / n_ch + 127) / 128) * 128;
            vocab_pad_ = n_ch * ch_cols;
            W_lm_head_.K = N;
            W_lm_head_.chunk_cols = ch_cols;
            W_lm_head_.total_cols = vocab_pad_;
            W_lm_head_.chunks.resize((size_t)n_ch);
            const size_t rb = ref::row_bytes(t_out->type, N);
            for (int64_t s = 0; s < n_ch; ++s) {
                const int64_t r0 = s * ch_cols;
                const int64_t rows = std::max<int64_t>(0, std::min<int64_t>(ch_cols, n_vocab_ - r0));
                gguf::Tensor sub = *t_out;
                sub.offset = t_out->offset + (size_t)r0 * rb;
                sub.shape = {(uint64_t)N, (uint64_t)rows};
                if (!load_int8_mat(last_dev, "lm_head_c" + std::to_string(s) + "_" + std::to_string(ch_cols),
                                   N, ch_cols, {{&sub, 0, rows}}, W_lm_head_.chunks[(size_t)s])) {
                    return false;
                }
            }
            if (n_ch > 1 && !W_lm_head_.stitch_tmp.alloc(last_dev, (size_t)MAXT * (size_t)vocab_pad_ * 2, err, false)) {
                return false;
            }
        }
        if (!upload_f32_dev(last_dev, "output_hc_norm.weight", HCD, out_hc_norm_) ||
            !load_nz_mat(last_dev, "out_hc_dn.nz16", HCD, W_HC_DN_COLS, {{f_.find("output_hc_down.weight"), 0, LR}}, W_out_hc_dn_) ||
            !load_nz_mat(last_dev, "out_hc_up.nz16", LR, HCD, {{f_.find("output_hc_up.weight"), 0, HCD}}, W_out_hc_up_) ||
            !d_lm_f16_.alloc(last_dev, (size_t)MAXT * vocab_pad_ * 2, err) ||
            !d_lm_f32_.alloc(last_dev, (size_t)MAXT * vocab_pad_ * 4, err) ||
            !d_arg_pv_.alloc(last_dev, (size_t)MAXT * 64 * 4, err) ||
            !d_arg_pi_.alloc(last_dev, (size_t)MAXT * 64 * 4, err) ||
            !d_tok_out_.alloc(last_dev, 8 * 4, err)) {
            return false;
        }
        h_kn = {};
        h_i8_nz = {};
        h_i8_sc = {};

        std::fprintf(stderr, "strata ascend: Pass 1/2 dense(INT8+FP16) + states + LM head ready in %.1f s; starting Pass 2/2 MoE expert packing...\n",
                     (now_ms() - t0) / 1000.0);
        std::fflush(stderr);

        // Pass 2: Auto-fill HBM on every NPU with 280-TOPS INT8 FRACTAL_NZ Cube MoE experts!
        const bool force_q2 = std::getenv("STRATA_ASCEND_FORCE_Q2") && std::string(std::getenv("STRATA_ASCEND_FORCE_Q2")) == "1";
        size_t hbm_reserve = 320ULL << 20; // 320 MiB runtime headroom (packs 48/48 INT8 Cube solo or 47/48 + DFlash2 on 3x48GB 310P3)
        if (const char* er = std::getenv("STRATA_ASCEND_HBM_RESERVE_MIB")) {
            int mib = std::atoi(er);
            if (mib >= 64) hbm_reserve = (size_t)mib << 20;
        }
        const size_t df2_hbm_reserve = !g_df2_pack_path.empty() ? (1080ULL << 20) : 0ULL;

        const size_t w1_nz_bytes = (size_t)nexp_ * (size_t)(2 * NFF) * (size_t)N;
        const size_t w1_sc_bytes = (size_t)nexp_ * (size_t)(2 * NFF) * sizeof(float);
        const size_t w2_nz_bytes = (size_t)nexp_ * (size_t)N * (size_t)NFF;
        const size_t w2_sc_bytes = (size_t)nexp_ * (size_t)N * sizeof(float);
        const size_t int8_layer_bytes = w1_nz_bytes + w1_sc_bytes + w2_nz_bytes + w2_sc_bytes + 8192;
        const size_t q2_layer_bytes   = (size_t)nexp_ * 2 * NFF * (640 + 80) + (size_t)nexp_ * N * (160 + 20) + 8192;

        std::vector<int8_t> h_w1_nz, h_w2_nz;
        std::vector<float>  h_w1_sc, h_w2_sc;
        std::vector<uint8_t> h_qs;
        std::vector<uint16_t> h_d;
        std::vector<int> rem_on_dev = dev_layer_cnt;
        int n_int8_layers = 0, n_q2_layers = 0;

        for (int l = 0; l < NL; ++l) {
            LayerWeights& lw = layers_[(size_t)l];
            const int dev = lw.dev;
            const int rem = rem_on_dev[(size_t)dev];
            rem_on_dev[(size_t)dev] -= 1;

            const std::string b = "blk." + std::to_string(l) + ".";
            const gguf::Tensor* tg = f_.find(b + "ffn_gate_exps.weight");
            const gguf::Tensor* tu = f_.find(b + "ffn_up_exps.weight");
            const gguf::Tensor* td = f_.find(b + "ffn_down_exps.weight");
            if (!tg || !tu || !td || tg->type != ref::Q2_0 || tu->type != ref::Q2_0 || td->type != ref::Q2_0) {
                err = "ascend window: routed experts in layer " + std::to_string(l) + " are not Q2_0";
                return false;
            }

            size_t free_b = 0, total_b = 0;
            (void)aclrtSetDevice(dev);
            (void)aclrtGetMemInfo(ACL_HBM_MEM, &free_b, &total_b);
            const size_t extra_df2 = (dev == df2_dev_) ? df2_hbm_reserve : 0ULL;
            const size_t min_for_int8 = int8_layer_bytes + (size_t)(rem - 1) * q2_layer_bytes + hbm_reserve + extra_df2;
            lw.use_int8_cube = (!force_q2 && free_b >= min_for_int8);

            if (lw.use_int8_cube) {
                const std::string pfx = "l" + std::to_string(l) + "_moe_";
                h_w1_nz.resize(w1_nz_bytes);
                h_w1_sc.resize(w1_sc_bytes / 4);
                h_w2_nz.resize(w2_nz_bytes);
                h_w2_sc.resize(w2_sc_bytes / 4);

                bool hit = cache.load_blob(pfx + "w1.nz8", h_w1_nz.data(), w1_nz_bytes) &&
                           cache.load_blob(pfx + "w1.sc",  h_w1_sc.data(), w1_sc_bytes) &&
                           cache.load_blob(pfx + "w2.nz8", h_w2_nz.data(), w2_nz_bytes) &&
                           cache.load_blob(pfx + "w2.sc",  h_w2_sc.data(), w2_sc_bytes);
                if (!hit) {
                    pack_q2_experts_int8_nz(map_base + tg->offset, map_base + tu->offset, map_base + td->offset,
                                            nexp_, h_w1_nz, h_w1_sc, h_w2_nz, h_w2_sc, nth);
                    if (cache.has_space_for(int8_layer_bytes)) {
                        cache.save_blob(pfx + "w1.nz8", h_w1_nz.data(), w1_nz_bytes);
                        cache.save_blob(pfx + "w1.sc",  h_w1_sc.data(), w1_sc_bytes);
                        cache.save_blob(pfx + "w2.nz8", h_w2_nz.data(), w2_nz_bytes);
                        cache.save_blob(pfx + "w2.sc",  h_w2_sc.data(), w2_sc_bytes);
                    }
                }
                if (!lw.W_moe_w1.upload_prepacked(dev, h_w1_nz.data(), h_w1_sc.data(), nexp_, 2 * NFF, N, err) ||
                    !lw.W_moe_w2.upload_prepacked(dev, h_w2_nz.data(), h_w2_sc.data(), nexp_, N, NFF, err)) {
                    // Self-healing fallback: if HBM hugepages fragment near the 98% ceiling, fall back cleanly to Q2_0!
                    lw.W_moe_w1 = {};
                    lw.W_moe_w2 = {};
                    err.clear();
                    lw.use_int8_cube = false;
                } else {
                    ++n_int8_layers;
                }
            }
            if (!lw.use_int8_cube) {
                split_q2_experts_gu(map_base + tg->offset, map_base + tu->offset, nexp_, h_qs, h_d, nth);
                if (!lw.gu_qs.alloc(dev, h_qs.size(), err, false) || !lw.gu_qs.upload(h_qs.data(), h_qs.size(), err) ||
                    !lw.gu_d.alloc(dev, h_d.size() * 2, err, false) || !lw.gu_d.upload(h_d.data(), h_d.size() * 2, err)) return false;

                split_q2_experts_down(map_base + td->offset, nexp_, h_qs, h_d, nth);
                if (!lw.down_qs.alloc(dev, h_qs.size(), err, false) || !lw.down_qs.upload(h_qs.data(), h_qs.size(), err) ||
                    !lw.down_d.alloc(dev, h_d.size() * 2, err, false) || !lw.down_d.upload(h_d.data(), h_d.size() * 2, err)) return false;
                ++n_q2_layers;
            }

            if ((l + 1) % 8 == 0 || l + 1 == NL) {
                std::fprintf(stderr, "strata ascend: loaded MoE %d / %d layers (INT8_Cube=%d, Q2_0=%d, %.1f s)\n",
                             l + 1, NL, n_int8_layers, n_q2_layers, (now_ms() - t0) / 1000.0);
                std::fflush(stderr);
            }
        }

        // Release temporary packing vectors so host heap stays clean
        h_w1_nz = {}; h_w2_nz = {}; h_w1_sc = {}; h_w2_sc = {}; h_qs = {}; h_d = {};

        // Allocate DMA-pinned host staging buffers
        if (!h_R_.alloc((size_t)MAX_ROWS * HCD) ||
            !h_bo_.alloc((size_t)MAX_ROWS * N) ||
            !h_inj_.alloc((size_t)(MAX_ROWS * HC + 16)) ||
            !h_ple_emb_.alloc((size_t)MAX_ROWS * N) ||
            !h_ple_kv_f16_.alloc((size_t)MAX_ROWS * (HCD + N)) ||
            !h_logits_f32_.alloc((size_t)MAXT * vocab_pad_) ||
            !h_tok_buf_.alloc(8)) {
            err = "ascend window: failed to allocate DMA-pinned host staging buffers";
            return false;
        }

        pin_host_thread_numa1();

        for (int d = 0; d < n_npu_; ++d) {
            size_t free_b = 0, total_b = 0;
            (void)aclrtSetDevice(d);
            (void)aclrtGetMemInfo(ACL_HBM_MEM, &free_b, &total_b);
            int d_i8 = 0, d_q2 = 0;
            for (int l = 0; l < NL; ++l) {
                if (layers_[(size_t)l].dev == d) {
                    if (layers_[(size_t)l].use_int8_cube) ++d_i8;
                    else ++d_q2;
                }
            }
            const double used_g = (double)(total_b - free_b) / (1024.0 * 1024.0 * 1024.0);
            const double tot_g  = (double)total_b / (1024.0 * 1024.0 * 1024.0);
            std::fprintf(stderr, "strata ascend: NPU %d (%s) HBM used %.2f / %.2f GiB (%.1f%%, INT8_Cube=%d, Q2_0=%d)\n",
                         d, aclrtGetSocName() ? aclrtGetSocName() : "Ascend",
                         used_g, tot_g, tot_g > 0.0 ? (100.0 * used_g / tot_g) : 0.0, d_i8, d_q2);
        }
        std::fprintf(stderr, "strata ascend: Qwen3.8-Flash-Next window ready in %.1f s (NPUs=%d, INT8_Cube=%d/%d, max_cells=%lld, PLE=%s)\n",
                     (now_ms() - t0) / 1000.0, n_npu_, n_int8_layers, NL, (long long)max_cells_, ple_on_ ? "on" : "off");
        std::fflush(stderr);
        ready_ = true;
        return true;
    }

    bool fold_pending(std::string& err, bool wait_sync = false, int skip_dev = -1) {
        bool any = false;
        for (int d = 0; d < n_npu_; ++d) {
            if (d != skip_dev && pend_fold_dev_[d] > 0) any = true;
        }
        if (any) {
            const int src_hp = pend_fold_hp_;
            for (int l = 0; l < NL; ++l) {
                if (l % QI == QI - 1) continue;
                const LayerWeights& lw = layers_[(size_t)l];
                if (lw.dev == skip_dev || pend_fold_dev_[lw.dev] <= 0) continue;
                const int nf = pend_fold_dev_[lw.dev];
                const int gi = l - l / QI;
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
                if (d != skip_dev) pend_fold_dev_[d] = 0;
            }
        }
        if (wait_sync) {
            for (int d = 0; d < n_npu_; ++d) {
                if (d != skip_dev && !npu_[d]->sync(err)) return false;
            }
        }
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

        // 1) Fast FP32 RAM token embeddings -> broadcast across 4 HC streams -> DMA-pinned H2D upload to NPU 0
        for (int t = 0; t < Tn; ++t) {
            float* r = h_R_.data() + (size_t)t * HCD;
            if (embd_rows && embd_rows[t]) {
                std::memcpy(r, embd_rows[t], (size_t)N * 4);
            } else {
                const int64_t tid = std::max<int64_t>(0, std::min<int64_t>(n_vocab_ - 1, tokens[t]));
                std::memcpy(r, h_tok_embd_f32_.data() + (size_t)tid * N, (size_t)N * 4);
            }
            for (int c = 1; c < HC; ++c) std::memcpy(r + (size_t)c * N, r, (size_t)N * 4);
        }

        NpuCtx& c0 = *npu_[0];
        if (!c0.R.upload(h_R_.data(), (size_t)Tn * HCD * 4, err)) return false;

        // 2) Gather PLE n-gram rows from locked Host RAM & launch W_ple_kv Cube matmul on NPU 0
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
        const bool want_taps = taps_on_ && (head || (tap_cb_ && pos0 + Tn > tap_first_));

        for (int l = 0; l < NL; ++l) {
            const LayerWeights& lw = layers_[(size_t)l];
            // Cross-NPU DMA-pinned activation handoff whenever pipeline stage transitions
            if (l > 0 && lw.dev != layers_[(size_t)(l - 1)].dev) {
                NpuCtx& prev_nc = *npu_[layers_[(size_t)(l - 1)].dev];
                if (!prev_nc.sync(err)) return false;
                const size_t r_bytes = (size_t)Tn * HCD * 4;
                const size_t b_bytes = (size_t)Tn * N * 4;
                const size_t i_bytes = (size_t)(((Tn * HC + 7) / 8) * 8) * 4;
                if (!prev_nc.R.download(h_R_.data(), r_bytes, err) ||
                    !prev_nc.bo_f32.download(h_bo_.data(), b_bytes, err) ||
                    !prev_nc.inj[cur_inj].download(h_inj_.data(), i_bytes, err)) return false;
                NpuCtx& next_nc = *npu_[lw.dev];
                if (!next_nc.R.upload(h_R_.data(), r_bytes, err) ||
                    !next_nc.bo_f32.upload(h_bo_.data(), b_bytes, err) ||
                    !next_nc.inj[cur_inj].upload(h_inj_.data(), i_bytes, err)) return false;
            }

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
                nc.stream, nc.xn_f32.ptr, nc.gate_f16.ptr, nc.xa_f16.ptr, /*out_f32=*/nullptr, (uint32_t)Tn);
            cur_inj = 1 - cur_inj;

            // ---- Mixer: GDN or QSA (280-TOPS INT8 FRACTAL_NZ Cube QGM(E=1)) ----
            if (!lw.is_qsa) {
                const int gi = l - l / QI;
                GdnState& gs = gdn_states_[(size_t)gi];
                if (!nc.qgm1(nc.xa_f16.ptr, lw.W_gdn_in, nc.gdn_in_f16.ptr, Tn, err)) return false;
                strata::ascend_ops::launch_gdn_step(
                    nc.stream, gs.state.ptr, gs.hist[hp_].ptr, gs.hist[1 - hp_].ptr,
                    nc.gdn_in_f16.ptr,
                    gdn_commit == 0 ? gs.qkv_save.ptr : nullptr,
                    gdn_commit == 0 ? gs.z_save.ptr   : nullptr,
                    gdn_commit == 0 ? gs.ab_save.ptr  : nullptr,
                    lw.ssm_conv_w.ptr, lw.ssm_dt.ptr, lw.ssm_a.ptr, lw.ssm_norm.ptr,
                    nc.yg_f16.ptr, (uint32_t)Tn, gdn_commit, /*from_saved=*/0u, eps_);
                if (!nc.qgm1(nc.yg_f16.ptr, lw.W_ssm_out, nc.bo_f16.ptr, Tn, err)) return false;
            } else {
                const int qi = l / QI;
                QsaLayerState& qs = qsa_states_[(size_t)qi];
                if (!nc.qgm1(nc.xa_f16.ptr, lw.W_qsa_in, nc.qsa_in_f16.ptr, Tn, err)) return false;
                strata::ascend_ops::launch_qsa_prep(
                    nc.stream, nc.qsa_in_f16.ptr,
                    lw.q_norm.ptr, lw.k_norm.ptr, lw.iq_norm.ptr, lw.ik_norm.ptr,
                    nc.rope_cs.ptr,
                    nc.qcur_f32.ptr, nc.qgate_f32.ptr, nc.qidx_f32.ptr,
                    qs.kq.ptr, qs.vq.ptr, qs.ksc.ptr, qs.vsc.ptr,
                    qs.tail.ptr, qs.dead.ptr, qs.pooled.ptr,
                    (uint32_t)pos0, 0u, (uint32_t)Tn, eps_);
                if (Tn > MAXT || pos0 + Tn > strata::ascend_ops::QSA_BUDGET) {
                    strata::ascend_ops::launch_qsa_select(
                        nc.stream, nc.qidx_f32.ptr, qs.pooled.ptr, qs.dead.ptr,
                        nc.scores_f32.ptr, nc.qids_i32.ptr,
                        (uint32_t)pos0, (uint32_t)max_blocks_, (uint32_t)Tn);
                }
                strata::ascend_ops::launch_qsa_attn_merge(
                    nc.stream, nc.qcur_f32.ptr, nc.qgate_f32.ptr,
                    qs.kq.ptr, qs.vq.ptr, qs.ksc.ptr, qs.vsc.ptr,
                    nc.qids_i32.ptr, nc.xq_f16.ptr,
                    (uint32_t)pos0, (uint32_t)Tn);
                if (!nc.qgm1(nc.xq_f16.ptr, lw.W_attn_out, nc.bo_f16.ptr, Tn, err)) return false;
            }

            // ---- FFN Hyper-Connection Read (apply=2u fuses bo_f16 -> f32 directly in UB!) ----
            strata::ascend_ops::launch_gr_norm(
                nc.stream, nc.R.ptr, nc.bo_f16.ptr, nc.inj[cur_inj].ptr, lw.hc_ffn_norm.ptr,
                nc.xn_f32.ptr, nc.xn_f16.ptr, (uint32_t)Tn, /*apply=*/2u, eps_);
            if (!nc.mm(nc.xn_f16.ptr, lw.W_hc_ffn_dn, nc.hc_dn_f16.ptr, Tn, err)) return false;
            strata::ascend_ops::launch_gr_post_down(
                nc.stream, nc.hc_dn_f16.ptr, nc.lo_f16.ptr, nc.inj[1 - cur_inj].ptr, (uint32_t)Tn);
            if (!nc.mm(nc.lo_f16.ptr, lw.W_hc_ffn_up, nc.gate_f16.ptr, Tn, err)) return false;
            strata::ascend_ops::launch_gr_post_up(
                nc.stream, nc.xn_f32.ptr, nc.gate_f16.ptr, nc.xf_f16.ptr,
                lw.use_int8_cube ? nullptr : nc.xf_f32.ptr, (uint32_t)Tn);
            cur_inj = 1 - cur_inj;

            // ---- MoE Sublayer ----
            if (!nc.mm(nc.xf_f16.ptr, lw.W_moe_in, nc.moe_in_f16.ptr, Tn, err)) return false;
            if (lw.use_int8_cube) {
                // Full-batch 280-TOPS INT8 FRACTAL_NZ Cube MoE pass (handles Tn = 1..256 -> M = 10..2560 in ONE call!)
                const uint32_t M_moe = (uint32_t)(Tn * TOPK);
                strata::ascend_ops::launch_moe_route_cube(
                    nc.stream, nc.moe_in_f16.ptr, nc.ids_i32.ptr, nc.wts_f32.ptr, nc.sh_h_f16.ptr,
                    nc.group_cumsum_i64.ptr, nc.sorted_tok_i32.ptr, nc.tok_perm_i32.ptr,
                    (uint32_t)Tn, (uint32_t)nexp_);
                if (!nc.mm(nc.sh_h_f16.ptr, lw.W_shexp_down, nc.ysh_f16.ptr, Tn, err)) return false;
                strata::ascend_ops::launch_moe_gather_x(
                    nc.stream, nc.xf_f16.ptr, nc.sorted_tok_i32.ptr, nc.sorted_x_f16.ptr, (uint32_t)Tn, M_moe);
                if (!nc.qgm(nc.sorted_x_f16.ptr, lw.W_moe_w1, nc.moe_gu_f16.ptr, (int64_t)M_moe, err)) return false;
                strata::ascend_ops::launch_moe_swiglu_f16(nc.stream, nc.moe_gu_f16.ptr, nc.moe_h_f16.ptr, M_moe);
                if (!nc.qgm(nc.moe_h_f16.ptr, lw.W_moe_w2, nc.moe_ye_f16.ptr, (int64_t)M_moe, err)) return false;
                strata::ascend_ops::launch_moe_combine_f16(
                    nc.stream, nc.moe_ye_f16.ptr, nc.tok_perm_i32.ptr, nc.wts_f32.ptr,
                    nc.ysh_f16.ptr, nc.bo_f32.ptr, (uint32_t)Tn, M_moe);
            } else {
                // Fallback planar Q2_0 path in 8-row windows
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
            }

            // ---- DFlash2 Tap Extraction (mean of completed HC streams after tapped layer) ----
            if (want_taps && tap_slot_[l] >= 0) {
                float* dst_slot = reinterpret_cast<float*>(nc.d_tap.ptr) + (size_t)tap_dev_slot_[l] * MAX_ROWS * N;
                strata::ascend_ops::launch_df2_tap(
                    nc.stream, nc.R.ptr, nc.bo_f32.ptr, nc.inj[cur_inj].ptr, dst_slot, (uint32_t)Tn);
            }

            // ---- PLE Block after Layer 0 ----
            if (l == 0 && ple_on_) {
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

        // 3) Final Hyper-Connection + 280-TOPS INT8 Cube LM Head on final NPU
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
                lc.stream, lc.xn_f32.ptr, lc.gate_f16.ptr, lc.xa_f16.ptr, /*out_f32=*/nullptr, (uint32_t)Tn);

            if (!lc.qgm1_chunked(lc.xa_f16.ptr, W_lm_head_, d_lm_f16_.ptr, Tn, err)) return false;
            strata::ascend_ops::launch_argmax_f16(
                lc.stream, d_lm_f16_.ptr, d_arg_pv_.ptr, d_arg_pi_.ptr, d_tok_out_.ptr,
                (uint32_t)Tn, (uint32_t)n_vocab_, (uint32_t)vocab_pad_);
        }

        for (int d = 0; d < n_npu_; ++d) {
            if (!npu_[d]->sync(err)) return false;
        }

        if (want_taps) {
            const size_t nt = tap_layers_.size();
            const size_t slot_bytes = (size_t)Tn * N * sizeof(float);
            for (size_t ti = 0; ti < nt; ++ti) {
                const int td = tap_dev_of_[ti];
                const int ls = tap_dev_local_[ti];
                if (!check_acl(aclrtSetDevice(td), "aclrtSetDevice(tap)", err)) return false;
                const void* src_ptr = reinterpret_cast<const uint8_t*>(npu_[td]->d_tap.ptr) + (size_t)ls * MAX_ROWS * N * sizeof(float);
                if (!check_acl(aclrtMemcpy(h_tap_raw_.data() + ti * MAX_ROWS * N, slot_bytes, src_ptr, slot_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                               "aclrtMemcpy(tap D2H)", err)) {
                    return false;
                }
            }
            float* dst_tap = head ? h_tap_dec_.data() : h_tap_pro_.data();
            for (int t = 0; t < Tn; ++t) {
                float* row_dst = dst_tap + (size_t)t * nt * N;
                for (size_t ti = 0; ti < nt; ++ti) {
                    std::memcpy(row_dst + ti * N, h_tap_raw_.data() + (ti * MAX_ROWS + (size_t)t) * N, (size_t)N * sizeof(float));
                }
            }
        }

        if (head) {
            if (!d_tok_out_.download(h_tok_buf_.data(), 8 * 4, err)) return false;
            for (int t = 0; t < Tn; ++t) {
                tok_out_[t] = h_tok_buf_[(size_t)t];
                if (out) out[t] = h_tok_buf_[(size_t)t];
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
            pend_fold_hp_ = hp_;
            hp_ = 1 - hp_;
            for (int d = 0; d < n_npu_; ++d) pend_fold_dev_[d] = n_keep;
            // Overlap GDN fold on non-DFlash2 NPUs (e.g. NPU 0 & NPU 2) concurrently with DFlash2::propose on df2_dev_!
            const int skip_dev = (df2_ && df2_->bound) ? df2_dev_ : -1;
            if (!fold_pending(err, /*wait_sync=*/false, skip_dev)) return false;
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
            if (!tap_deliver(k, pos0 + c, err)) return false;
            if (on_chunk && !on_chunk(nullptr, k, pos0 + c, err)) return false;
        }
        return true;
    }

    void reset() {
        std::memset(pend_fold_dev_, 0, sizeof(pend_fold_dev_));
        last_T_ = 0;
        std::string err;
        for (int d = 0; d < n_npu_; ++d) (void)npu_[d]->sync(err);
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
        if (!fold_pending(err, /*wait_sync=*/true)) return false;
        const size_t gs_b = (size_t)HV * SS * SS * 4;
        const size_t hs_b = (size_t)CONV * 3 * 4; // Strata shared contract: 3 active conv taps per channel
        c.gdn.resize(gdn_states_.size() * (gs_b + hs_b));
        std::vector<float> h4((size_t)CONV * 4);
        for (size_t i = 0; i < gdn_states_.size(); ++i) {
            uint8_t* dst = c.gdn.data() + i * (gs_b + hs_b);
            if (!gdn_states_[i].state.download(dst, gs_b, err) ||
                !gdn_states_[i].hist[hp_].download(h4.data(), h4.size() * 4, err)) return false;
            float* dst_h = reinterpret_cast<float*>(dst + gs_b);
            for (int ch = 0; ch < CONV; ++ch) {
                dst_h[(size_t)ch * 3 + 0] = h4[(size_t)ch * 4 + 0];
                dst_h[(size_t)ch * 3 + 1] = h4[(size_t)ch * 4 + 1];
                dst_h[(size_t)ch * 3 + 2] = h4[(size_t)ch * 4 + 2];
            }
        }
        c.ple.resize(ple_hist_.size() * 4);
        std::memcpy(c.ple.data(), ple_hist_.data(), c.ple.size());
        const size_t ts_b = 3 * ID * 4, ds_b = ID * 4; // Strata shared contract: rows 0..2 of the 4-row ring
        c.tails.resize(qsa_states_.size() * ts_b);
        c.dead.resize(qsa_states_.size() * ds_b);
        for (size_t j = 0; j < qsa_states_.size(); ++j) {
            if (!qsa_states_[j].tail.download(c.tails.data() + j * ts_b, ts_b, err) ||
                !qsa_states_[j].dead.download(c.dead.data() + j * ds_b, ds_b, err)) return false;
        }
        c.block_pos.assign(qsa_states_.size() * sizeof(int32_t), 0);
        return true;
    }

    bool checkpoint_restore(const strata::core::ConversationCheckpoint& c, std::string& err) {
        const size_t gs_b = (size_t)HV * SS * SS * 4;
        const size_t hs_b = (size_t)CONV * 3 * 4;
        const size_t ts_b = 3 * ID * 4, ds_b = ID * 4;
        if (c.gdn.size() != gdn_states_.size() * (gs_b + hs_b) ||
            c.ple.size() != ple_hist_.size() * 4 ||
            c.tails.size() != qsa_states_.size() * ts_b ||
            c.dead.size() != qsa_states_.size() * ds_b) {
            err = "ascend window: checkpoint size mismatch";
            return false;
        }
        std::memset(pend_fold_dev_, 0, sizeof(pend_fold_dev_));
        last_T_ = 0;
        for (int d = 0; d < n_npu_; ++d) (void)npu_[d]->sync(err);
        std::vector<float> h4((size_t)CONV * 4, 0.0f);
        for (size_t i = 0; i < gdn_states_.size(); ++i) {
            const uint8_t* src = c.gdn.data() + i * (gs_b + hs_b);
            const float* src_h = reinterpret_cast<const float*>(src + gs_b);
            for (int ch = 0; ch < CONV; ++ch) {
                h4[(size_t)ch * 4 + 0] = src_h[(size_t)ch * 3 + 0];
                h4[(size_t)ch * 4 + 1] = src_h[(size_t)ch * 3 + 1];
                h4[(size_t)ch * 4 + 2] = src_h[(size_t)ch * 3 + 2];
                h4[(size_t)ch * 4 + 3] = 0.0f;
            }
            if (!gdn_states_[i].state.upload(src, gs_b, err) ||
                !gdn_states_[i].hist[hp_].upload(h4.data(), h4.size() * 4, err)) return false;
        }
        std::memcpy(ple_hist_.data(), c.ple.data(), c.ple.size());
        std::vector<float> t4(4 * ID, 0.0f);
        for (size_t j = 0; j < qsa_states_.size(); ++j) {
            std::memcpy(t4.data(), c.tails.data() + j * ts_b, ts_b);
            if (!qsa_states_[j].tail.upload(t4.data(), t4.size() * 4, err) ||
                !qsa_states_[j].dead.upload(c.dead.data() + j * ds_b, ds_b, err)) return false;
        }
        return true;
    }

    // ---- Native DFlash2 Tap & Block Drafter Support on Ascend 310P3 ----
    bool taps_enable(const std::vector<int>& layers, std::string& err) {
        if (taps_on_) {
            if (layers != tap_layers_) {
                err = "ascend taps: already enabled for different layers";
                return false;
            }
            return true;
        }
        if (layers.empty()) {
            err = "ascend taps: empty layer list";
            return false;
        }
        for (int i = 0; i < 64; ++i) {
            tap_slot_[i] = -1;
            tap_dev_slot_[i] = -1;
        }
        std::vector<int> dev_cnt((size_t)n_npu_, 0);
        tap_dev_of_.resize(layers.size());
        tap_dev_local_.resize(layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            const int l = layers[i];
            if (l < 0 || l >= NL - 1) {
                err = "ascend taps: layer " + std::to_string(l) + " out of range";
                return false;
            }
            const int d = layers_[(size_t)l].dev;
            const int ls = dev_cnt[(size_t)d]++;
            tap_slot_[l] = (int)i;
            tap_dev_slot_[l] = ls;
            tap_dev_of_[i] = d;
            tap_dev_local_[i] = ls;
        }
        for (int d = 0; d < n_npu_; ++d) {
            if (dev_cnt[(size_t)d] > 0) {
                if (!npu_[d]->d_tap.alloc(d, (size_t)dev_cnt[(size_t)d] * MAX_ROWS * N * sizeof(float), err)) {
                    return false;
                }
            }
        }
        if (!h_tap_raw_.alloc(layers.size() * MAX_ROWS * N) ||
            !h_tap_dec_.alloc((size_t)MAXT * layers.size() * N) ||
            !h_tap_pro_.alloc((size_t)MAX_ROWS * layers.size() * N)) {
            err = "ascend taps: host pinned staging allocation failed";
            return false;
        }
        tap_layers_ = layers;
        taps_on_ = true;
        std::fprintf(stderr, "strata ascend: DFlash2 taps enabled (%zu layers)\n", layers.size());
        return true;
    }

    void taps_set_cb(int64_t first_pos, const std::function<bool(const float*, int64_t, int64_t, std::string&)>& cb) {
        tap_first_ = first_pos;
        tap_cb_ = cb;
    }

    const float* tap_dec_ptr() const { return h_tap_dec_.data(); }

    bool tap_deliver(int rows, int64_t pos0, std::string& err) {
        if (!taps_on_ || !tap_cb_) return true;
        const int64_t t0 = std::max<int64_t>(0, tap_first_ - pos0);
        if (t0 >= rows) return true;
        const size_t nt = tap_layers_.size();
        return tap_cb_(h_tap_pro_.data() + (size_t)t0 * nt * N, rows - t0, pos0 + t0, err);
    }

    struct Df2Layer {
        DevMem in_norm;       // [2560] f32
        DevMem post_norm;     // [2560] f32
        DevMem q_norm;        // [128] f32
        DevMem k_norm;        // [128] f32
        DevMem attn_base;     // [4 * 2560] f32
        DevMem mlp_base;      // [4 * 2560] f32
        NzMat  W_attn_conv;   // [2560, 640] f16 NZ
        NzMat  W_mlp_conv;    // [2560, 640] f16 NZ
        Int8GroupNzMat W_kv;  // [1, 2048, 2560] INT8 NZ ([k_1024 | v_1024])
        Int8GroupNzMat W_qkv; // [1, 6144, 2560] INT8 NZ ([q_4096 | k_1024 | v_1024])
        Int8GroupNzMat W_o;   // [1, 2560, 4096] INT8 NZ
        Int8GroupNzMat W_gu;  // [1, 15360, 2560] INT8 NZ ([gate_7680 | up_7680])
        Int8GroupNzMat W_down;// [1, 2560, 7680] INT8 NZ
    };

    struct Df2Ascend {
        std::string path;
        int dev = 0;
        int L = 5, NH = 32, NKV = 8, I = 7680, window = 2048, group = 16, rank = 256, mask_token = 0;
        float theta = 10000.0f, eps = 1e-6f;
        std::vector<int> taps;
        std::map<std::string, gguf::Tensor> ten;
        uint64_t data_off = 0, total = 0, pack_file_bytes = 0;

        int ndraft = DF2_DRAFTS;
        int64_t floor = 0;
        int64_t nd = 0, nd_pad = 0;
        bool use_map = false;
        bool bound = false;
        uint64_t vram_bytes = 0;
        double gpu_ms = 0.0;
        int64_t rounds = 0;

        Int8GroupNzMat W_fc;        // [1, 2560, 12800] INT8 NZ
        DevMem hidden_norm;         // [2560] f32
        DevMem final_norm;          // [2560] f32
        std::vector<Df2Layer> layers;
        Int8ChunkedMat W_dhead;     // column-chunked [1, nd_pad, 2560] INT8 NZ (<= 62080 cols per chunk!)
        DevMem d_vmap;              // [((nd + 7)/8)*8] i32
        NzMat  W_sel_proj;          // [2560, 256] f16 NZ

        int pred_cb_type = -1, succ_cb_type = -1;
        std::vector<uint8_t> pred_cb_raw, succ_cb_raw;
        std::vector<float> mask_emb;

        // KV ring + lineage rowlog
        DevMem kring[8], vring[8];
        DevMem rowlog;              // [DF2_RING, 2560] f16
        struct Stash {
            std::vector<int32_t> hist;
            int64_t hi = 0, floor = 0;
            uint64_t used = 0;
            std::vector<uint16_t> host_log; // kept in Host DDR4 RAM (0 MB HBM!)
        };
        std::vector<Stash> stash;
        uint64_t stash_tick = 0;
        int stash_max = 4;

        // Context & Block NPU scratch
        DevMem d_taps_f32, d_taps_f16, d_fcy_f16, d_hctx_f16, d_kvc_f16, d_ctx_pos, d_ctx_rope;
        DevMem d_h_f32, d_xn_f32, d_xn_f16, d_xc_f16, d_dyn_f16, d_qkv_f16;
        DevMem d_qn_f32, d_kblk_f32, d_vblk_f32, d_blk_rope;
        DevMem d_ao_f16, d_o_f16, d_gu_f16, d_sw_f16, d_mlp_f16, d_fn_f16;
        DevMem d_logits_f16, d_tk_pv, d_tk_pi, d_topval, d_toptok, d_hsel_f16;

        HostPinned<int32_t>  h_ctx_pos, h_toptok;
        HostPinned<float>    h_ctx_rope, h_blk_h, h_blk_rope, h_topval;
        HostPinned<uint16_t> h_hsel;
    };

    int df2_dev() const { return df2_dev_; }
    uint64_t df2_bytes() const { return df2_ ? df2_->vram_bytes : 0; }
    double df2_gpu_ms() const { return df2_ ? df2_->gpu_ms : 0.0; }
    void df2_set_cap(int n) { if (df2_) df2_->ndraft = std::max(1, std::min(n, DF2_DRAFTS)); }
    void df2_set_floor(int64_t pos) { if (df2_) df2_->floor = pos; }
    int64_t df2_floor() const { return df2_ ? df2_->floor : 0; }

    bool df2_ensure(const std::string& path, std::string& err) {
        if (df2_) return true;
        auto d = std::make_unique<Df2Ascend>();
        d->path = path;
        d->dev = df2_dev_;
        std::map<std::string, std::string> cfg;
        if (!df2_read_index(path, d->taps, d->ten, d->total, d->data_off, cfg, err)) return false;
        struct stat st{};
        if (::stat(path.c_str(), &st) == 0) d->pack_file_bytes = (uint64_t)st.st_size;
        auto geti = [&](const char* k, int def) { auto it = cfg.find(k); return it == cfg.end() ? def : std::atoi(it->second.c_str()); };
        d->L = geti("layers", 5); d->NH = geti("heads", 32); d->NKV = geti("kv_heads", 8); d->I = geti("inter", 7680);
        d->window = geti("window", 2048); d->group = geti("conv_group", 16); d->rank = geti("rank", 256);
        d->mask_token = geti("mask_token", 0);
        if (cfg.count("rope_theta")) d->theta = (float)std::atof(cfg["rope_theta"].c_str());
        if (cfg.count("eps")) d->eps = (float)std::atof(cfg["eps"].c_str());
        if (geti("hidden", 0) != N || d->NH != 32 || d->NKV != 8 || geti("head_dim", 0) != 128 || d->rank != 256 ||
            d->L != 5 || d->window + 16 > DF2_RING || d->I != 7680 || d->group != 16 || d->taps.empty()) {
            err = "dflash2: unsupported DF2P geometry for Qwen3.8-Flash-Next";
            return false;
        }
        df2_ = std::move(d);
        return true;
    }

    bool df2_bind(const std::string& vocab_file, std::string& err) {
        if (!df2_) { err = "dflash2: pack not opened"; return false; }
        Df2Ascend& d = *df2_;
        if (d.bound) return true;
        if (!taps_on_) { err = "dflash2: target window taps are not enabled"; return false; }

        unpin_all_cpus();
        const double t0 = now_ms();
        const unsigned nth = std::max(1u, std::min(64u, std::thread::hardware_concurrency()));
        const int dev = d.dev;
        NpuCtx& dc = *npu_[dev];

        size_t free_before = 0, total_hbm = 0;
        (void)aclrtSetDevice(dev);
        (void)aclrtGetMemInfo(ACL_HBM_MEM, &free_before, &total_hbm);

        // Mmap DF2P pack file for parallel 64-core packing
        int pfd = ::open(d.path.c_str(), O_RDONLY);
        if (pfd < 0) { err = "dflash2: cannot open " + d.path; return false; }
        struct stat pst{};
        if (::fstat(pfd, &pst) != 0 || (uint64_t)pst.st_size < d.data_off + d.total) {
            ::close(pfd);
            err = "dflash2: pack file truncated";
            return false;
        }
        void* pmap = ::mmap(nullptr, (size_t)pst.st_size, PROT_READ, MAP_PRIVATE, pfd, 0);
        ::close(pfd);
        if (pmap == MAP_FAILED) { err = "dflash2: mmap failed on " + d.path; return false; }
        struct Unmapper { void* p; size_t sz; ~Unmapper() { if (p != MAP_FAILED) ::munmap(p, sz); } } unmap{pmap, (size_t)pst.st_size};
        const uint8_t* pack_data = reinterpret_cast<const uint8_t*>(pmap) + d.data_off;

        CacheStore cache;
        cache.init(f_.file_bytes, nexp_);
        const std::string cpfx = "df2_" + std::to_string(d.pack_file_bytes) + "_";

        auto find_pt = [&](const std::string& name) -> const gguf::Tensor* {
            auto it = d.ten.find(name);
            return it == d.ten.end() ? nullptr : &it->second;
        };

        auto upload_pack_f32 = [&](const std::string& name, size_t expect_ne, DevMem& dm) -> bool {
            const gguf::Tensor* t = find_pt(name);
            if (!t) { err = "dflash2: missing tensor " + name; return false; }
            uint64_t ne = 1;
            for (uint64_t s : t->shape) ne *= s;
            std::vector<float> tmp(std::max<size_t>(expect_ne, (size_t)ne), 0.0f);
            ref::dequant(t->type, pack_data + t->offset, tmp.data(), (size_t)ne);
            return dm.alloc(dev, tmp.size() * 4, err, false) && dm.upload(tmp.data(), tmp.size() * 4, err);
        };

        std::vector<uint16_t> h_kn;
        auto upload_pack_nz16 = [&](const std::string& key, int64_t K_in, int64_t N_cols,
                                    const std::vector<MatSlice>& slices, NzMat& mat) -> bool {
            const size_t sz = (size_t)K_in * (size_t)N_cols * sizeof(uint16_t);
            h_kn.resize((size_t)K_in * (size_t)N_cols);
            if (cache.load_blob(cpfx + key + ".nz16", h_kn.data(), sz)) {
                return mat.upload_cached_nz(dev, h_kn.data(), K_in, N_cols, err);
            }
            build_fused_kn_f16(pack_data, K_in, N_cols, slices, h_kn, nth);
            if (!mat.upload_and_format(dev, dc.stream, dc.ws.ptr, dc.ws.bytes, h_kn.data(), K_in, N_cols, err)) return false;
            if (cache.has_space_for(sz) && mat.mem.download(h_kn.data(), sz, err)) {
                cache.save_blob(cpfx + key + ".nz16", h_kn.data(), sz);
            }
            err.clear();
            return true;
        };

        std::vector<int8_t> h_nz8;
        std::vector<float>  h_sc;
        auto upload_pack_int8 = [&](const std::string& key, int64_t K_in, int64_t N_cols,
                                    const std::vector<MatSlice>& slices, Int8GroupNzMat& mat) -> bool {
            const size_t nz_sz = (size_t)K_in * (size_t)N_cols;
            const size_t sc_sz = (size_t)N_cols * sizeof(float);
            h_nz8.resize(nz_sz);
            h_sc.resize((size_t)N_cols);
            if (!cache.load_blob(cpfx + key + ".nz8", h_nz8.data(), nz_sz) ||
                !cache.load_blob(cpfx + key + ".sc",  h_sc.data(),  sc_sz)) {
                pack_fused_int8_nz(pack_data, K_in, N_cols, slices, h_nz8, h_sc, nth);
                if (cache.has_space_for(nz_sz + sc_sz)) {
                    cache.save_blob(cpfx + key + ".nz8", h_nz8.data(), nz_sz);
                    cache.save_blob(cpfx + key + ".sc",  h_sc.data(),  sc_sz);
                }
            }
            return mat.upload_prepacked(dev, h_nz8.data(), h_sc.data(), 1, N_cols, K_in, err);
        };

        const int64_t tap_dim = (int64_t)d.taps.size() * N;
        if (!upload_pack_int8("fc", tap_dim, N, {{find_pt("fc.weight"), 0, N}}, d.W_fc) ||
            !upload_pack_f32("hidden_norm.weight", N, d.hidden_norm) ||
            !upload_pack_f32("norm.weight", N, d.final_norm)) {
            return false;
        }

        d.layers.resize((size_t)d.L);
        for (int l = 0; l < d.L; ++l) {
            Df2Layer& dl = d.layers[(size_t)l];
            const std::string p = "layers." + std::to_string(l) + ".";
            const std::string k = "l" + std::to_string(l) + "_";
            if (!upload_pack_f32(p + "input_layernorm.weight", N, dl.in_norm) ||
                !upload_pack_f32(p + "post_attention_layernorm.weight", N, dl.post_norm) ||
                !upload_pack_f32(p + "self_attn.q_norm.weight", 128, dl.q_norm) ||
                !upload_pack_f32(p + "self_attn.k_norm.weight", 128, dl.k_norm) ||
                !upload_pack_f32(p + "attention_conv.base_kernel", 4 * N, dl.attn_base) ||
                !upload_pack_f32(p + "mlp_conv.base_kernel", 4 * N, dl.mlp_base)) {
                return false;
            }
            if (!upload_pack_nz16(k + "aconv", N, 640, {{find_pt(p + "attention_conv.kernel_projection.weight"), 0, 640}}, dl.W_attn_conv) ||
                !upload_pack_nz16(k + "mconv", N, 640, {{find_pt(p + "mlp_conv.kernel_projection.weight"), 0, 640}}, dl.W_mlp_conv)) {
                return false;
            }
            if (!upload_pack_int8(k + "kv", N, 2048,
                                  {{find_pt(p + "self_attn.k_proj.weight"), 0,    1024},
                                   {find_pt(p + "self_attn.v_proj.weight"), 1024, 1024}}, dl.W_kv) ||
                !upload_pack_int8(k + "qkv", N, 6144,
                                  {{find_pt(p + "self_attn.q_proj.weight"), 0,    4096},
                                   {find_pt(p + "self_attn.k_proj.weight"), 4096, 1024},
                                   {find_pt(p + "self_attn.v_proj.weight"), 5120, 1024}}, dl.W_qkv) ||
                !upload_pack_int8(k + "o", 4096, N, {{find_pt(p + "self_attn.o_proj.weight"), 0, N}}, dl.W_o) ||
                !upload_pack_int8(k + "gu", N, 2 * d.I,
                                  {{find_pt(p + "mlp.gate_proj.weight"), 0,   d.I},
                                   {find_pt(p + "mlp.up_proj.weight"),   d.I, d.I}}, dl.W_gu) ||
                !upload_pack_int8(k + "down", d.I, N, {{find_pt(p + "mlp.down_proj.weight"), 0, N}}, dl.W_down)) {
                return false;
            }
        }

        if (!upload_pack_nz16("sel_proj", N, d.rank, {{find_pt("candidate_selector.hidden_projection.weight"), 0, d.rank}}, d.W_sel_proj)) {
            return false;
        }

        // Load selector codebooks into Host DDR4 RAM (saves ~72 MiB NPU HBM!)
        const gguf::Tensor* t_pred = find_pt("candidate_selector.predecessor_codebook");
        const gguf::Tensor* t_succ = find_pt("candidate_selector.successor_codebook");
        if (!t_pred || !t_succ) { err = "dflash2: missing selector codebooks"; return false; }
        d.pred_cb_type = t_pred->type;
        d.succ_cb_type = t_succ->type;
        d.pred_cb_raw.assign(pack_data + t_pred->offset, pack_data + t_pred->offset + t_pred->bytes);
        d.succ_cb_raw.assign(pack_data + t_succ->offset, pack_data + t_succ->offset + t_succ->bytes);
        if (d.pred_cb_type == ref::Q4_0) {
            std::fprintf(stderr, "strata ascend: DFlash2 selector codebooks: Q4_0 rows (in Host DDR4 RAM)\n");
        }

        // Load draft vocabulary list & pack INT8 FRACTAL_NZ draft head from target's output.weight
        std::vector<int32_t> vocab;
        const bool native_head = strata::core::Dflash2Drafter::head_native() || vocab_file == "full";
        if (!native_head) {
            auto try_load_vocab = [&](const std::string& p) {
                if (p.empty() || p == "full" || !vocab.empty()) return;
                if (std::FILE* vf = std::fopen(p.c_str(), "rb")) {
                    int32_t id = 0;
                    while (std::fread(&id, 4, 1, vf) == 1) {
                        if (id >= 0 && id < n_vocab_) vocab.push_back(id);
                    }
                    std::fclose(vf);
                }
            };
            try_load_vocab(vocab_file);
            try_load_vocab(draft_vocab_default());
            try_load_vocab("data/draft_vocab.bin");
        }
        d.nd = vocab.empty() ? n_vocab_ : (int64_t)vocab.size();
        const int64_t dh_nch = (d.nd + Int8ChunkedMat::MAX_CHUNK_COLS - 1) / Int8ChunkedMat::MAX_CHUNK_COLS;
        const int64_t dh_ch_cols = (((d.nd + dh_nch - 1) / dh_nch + 127) / 128) * 128;
        d.nd_pad = dh_nch * dh_ch_cols;
        d.use_map = !vocab.empty();

        const size_t vmap_aligned = (((size_t)d.nd + 7) / 8) * 8;
        std::vector<int32_t> h_vmap(std::max<size_t>(8, vmap_aligned), 0);
        if (!vocab.empty()) std::memcpy(h_vmap.data(), vocab.data(), vocab.size() * sizeof(int32_t));
        if (!d.d_vmap.alloc(dev, h_vmap.size() * sizeof(int32_t), err, false) ||
            !d.d_vmap.upload(h_vmap.data(), h_vmap.size() * sizeof(int32_t), err)) {
            return false;
        }

        {
            const gguf::Tensor* t_out = f_.find("output.weight");
            const uint8_t* out_base = reinterpret_cast<const uint8_t*>(f_.map) + t_out->offset;
            const size_t rb = ref::row_bytes(t_out->type, N);
            d.W_dhead.K = N;
            d.W_dhead.chunk_cols = dh_ch_cols;
            d.W_dhead.total_cols = d.nd_pad;
            d.W_dhead.chunks.resize((size_t)dh_nch);

            for (int64_t s = 0; s < dh_nch; ++s) {
                const int64_t r0 = s * dh_ch_cols;
                const int64_t rows = std::max<int64_t>(0, std::min<int64_t>(dh_ch_cols, d.nd - r0));
                const size_t nz_sz = (size_t)N * (size_t)dh_ch_cols;
                const size_t sc_sz = (size_t)dh_ch_cols * sizeof(float);
                const std::string hkey = "df2_dhead_" + std::to_string(d.nd) + "_c" + std::to_string(s) + "_" + std::to_string(dh_ch_cols);
                h_nz8.assign(nz_sz, 0);
                h_sc.assign((size_t)dh_ch_cols, 0.0f);

                if (!cache.load_blob(hkey + ".nz8", h_nz8.data(), nz_sz) ||
                    !cache.load_blob(hkey + ".sc",  h_sc.data(),  sc_sz)) {
                    const size_t n_blocks = (size_t)(dh_ch_cols >> 4);
                    const size_t k_blocks = (size_t)(N >> 5);
                    if (rows > 0) {
                        const unsigned workers = std::max(1u, std::min<unsigned>(nth, (unsigned)rows));
                        std::vector<std::thread> threads;
                        threads.reserve(workers);
                        for (unsigned tid = 0; tid < workers; ++tid) {
                            threads.emplace_back([&, tid] {
                                std::vector<float> row_f32((size_t)N);
                                std::vector<int8_t> row_i8((size_t)N);
                                const int64_t i0 = (rows * tid) / workers;
                                const int64_t i1 = (rows * (tid + 1)) / workers;
                                for (int64_t i = i0; i < i1; ++i) {
                                    const int64_t g_idx = r0 + i;
                                    const int64_t tok = vocab.empty() ? g_idx : (int64_t)vocab[(size_t)g_idx];
                                    ref::dequant(t_out->type, out_base + (size_t)tok * rb, row_f32.data(), N);
                                    float mx = 0.0f;
                                    for (int k = 0; k < N; ++k) mx = std::max(mx, std::fabs(row_f32[(size_t)k]));
                                    const float sc = mx > 0.0f ? (mx / 127.0f) : 1.0f;
                                    const float inv_sc = 1.0f / sc;
                                    h_sc[(size_t)i] = sc;
                                    for (int k = 0; k < N; ++k) {
                                        int q = (int)std::nearbyintf(row_f32[(size_t)k] * inv_sc);
                                        row_i8[(size_t)k] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
                                    }
                                    const size_t r_hi = (size_t)i >> 4, r_lo = (size_t)i & 15;
                                    for (size_t kh = 0; kh < k_blocks; ++kh) {
                                        const size_t off = (((kh * n_blocks + r_hi) * 16 + r_lo) * 32);
                                        std::memcpy(h_nz8.data() + off, row_i8.data() + kh * 32, 32);
                                    }
                                }
                            });
                        }
                        for (auto& th : threads) th.join();
                    }
                    if (cache.has_space_for(nz_sz + sc_sz)) {
                        cache.save_blob(hkey + ".nz8", h_nz8.data(), nz_sz);
                        cache.save_blob(hkey + ".sc",  h_sc.data(),  sc_sz);
                    }
                }
                if (!d.W_dhead.chunks[(size_t)s].upload_prepacked(dev, h_nz8.data(), h_sc.data(), 1, dh_ch_cols, N, err)) return false;
            }
            if (dh_nch > 1 && !d.W_dhead.stitch_tmp.alloc(dev, (size_t)DF2_ROWS * (size_t)d.nd_pad * 2, err, false)) {
                return false;
            }
        }
        h_kn = {}; h_nz8 = {}; h_sc = {};

        // Mask token embedding from pre-dequantized h_tok_embd_f32_
        const int64_t mt = std::max<int64_t>(0, std::min<int64_t>(n_vocab_ - 1, d.mask_token));
        d.mask_emb.assign(h_tok_embd_f32_.data() + (size_t)mt * N, h_tok_embd_f32_.data() + (size_t)(mt + 1) * N);

        // Allocate KV ring, rowlog, and scratch on df2_dev_
        for (int l = 0; l < d.L; ++l) {
            if (!d.kring[l].alloc(dev, (size_t)DF2_RING * 1024 * 4, err) ||
                !d.vring[l].alloc(dev, (size_t)DF2_RING * 1024 * 4, err)) return false;
        }
        if (!d.rowlog.alloc(dev, (size_t)DF2_RING * N * 2, err) ||
            !d.d_taps_f32.alloc(dev, (size_t)DF2_CTX_MAX * tap_dim * 4, err, false) ||
            !d.d_taps_f16.alloc(dev, (size_t)DF2_CTX_MAX * tap_dim * 2, err, false) ||
            !d.d_fcy_f16.alloc(dev, (size_t)DF2_CTX_MAX * N * 2, err, false) ||
            !d.d_hctx_f16.alloc(dev, (size_t)DF2_CTX_MAX * N * 2, err, false) ||
            !d.d_kvc_f16.alloc(dev, (size_t)DF2_CTX_MAX * 2048 * 2, err, false) ||
            !d.d_ctx_pos.alloc(dev, (size_t)DF2_CTX_MAX * 4, err) ||
            !d.d_ctx_rope.alloc(dev, (size_t)DF2_CTX_MAX * 128 * 4, err, false) ||
            !d.d_h_f32.alloc(dev, (size_t)DF2_ROWS * N * 4, err, false) ||
            !d.d_xn_f32.alloc(dev, (size_t)DF2_ROWS * N * 4, err, false) ||
            !d.d_xn_f16.alloc(dev, (size_t)DF2_ROWS * N * 2, err, false) ||
            !d.d_xc_f16.alloc(dev, (size_t)DF2_ROWS * N * 2, err, false) ||
            !d.d_dyn_f16.alloc(dev, (size_t)DF2_ROWS * 640 * 2, err, false) ||
            !d.d_qkv_f16.alloc(dev, (size_t)DF2_ROWS * 6144 * 2, err, false) ||
            !d.d_qn_f32.alloc(dev, (size_t)DF2_ROWS * 4096 * 4, err, false) ||
            !d.d_kblk_f32.alloc(dev, (size_t)DF2_ROWS * 1024 * 4, err, false) ||
            !d.d_vblk_f32.alloc(dev, (size_t)DF2_ROWS * 1024 * 4, err, false) ||
            !d.d_blk_rope.alloc(dev, (size_t)DF2_ROWS * 128 * 4, err, false) ||
            !d.d_ao_f16.alloc(dev, (size_t)DF2_ROWS * 4096 * 2, err, false) ||
            !d.d_o_f16.alloc(dev, (size_t)DF2_ROWS * N * 2, err, false) ||
            !d.d_gu_f16.alloc(dev, (size_t)DF2_ROWS * 2 * d.I * 2, err, false) ||
            !d.d_sw_f16.alloc(dev, (size_t)DF2_ROWS * d.I * 2, err, false) ||
            !d.d_mlp_f16.alloc(dev, (size_t)DF2_ROWS * N * 2, err, false) ||
            !d.d_fn_f16.alloc(dev, (size_t)DF2_ROWS * N * 2, err, false) ||
            !d.d_logits_f16.alloc(dev, (size_t)DF2_ROWS * (size_t)d.nd_pad * 2, err) ||
            !d.d_tk_pv.alloc(dev, (size_t)DF2_ROWS * 8 * 16 * 4, err) ||
            !d.d_tk_pi.alloc(dev, (size_t)DF2_ROWS * 8 * 16 * 4, err) ||
            !d.d_topval.alloc(dev, (size_t)DF2_ROWS * 16 * 4, err) ||
            !d.d_toptok.alloc(dev, (size_t)DF2_ROWS * 16 * 4, err) ||
            !d.d_hsel_f16.alloc(dev, (size_t)DF2_ROWS * 256 * 2, err)) {
            return false;
        }

        if (!d.h_ctx_pos.alloc(DF2_CTX_MAX) ||
            !d.h_ctx_rope.alloc((size_t)DF2_CTX_MAX * 128) ||
            !d.h_blk_h.alloc((size_t)DF2_ROWS * N) ||
            !d.h_blk_rope.alloc((size_t)DF2_ROWS * 128) ||
            !d.h_topval.alloc((size_t)DF2_ROWS * 16) ||
            !d.h_toptok.alloc((size_t)DF2_ROWS * 16) ||
            !d.h_hsel.alloc((size_t)DF2_ROWS * 256)) {
            err = "dflash2: failed to allocate pinned host buffers";
            return false;
        }

        pin_host_thread_numa1();

        size_t free_after = 0;
        (void)aclrtSetDevice(dev);
        (void)aclrtGetMemInfo(ACL_HBM_MEM, &free_after, &total_hbm);
        d.vram_bytes = free_before > free_after ? (free_before - free_after) : 0;
        d.bound = true;
        std::fprintf(stderr, "strata ascend: DFlash2 ready on NPU %d (%.0f MiB HBM, %.2f GiB free, %lld-token INT8 Cube draft head) in %.1f s\n",
                     dev, (double)d.vram_bytes / 1048576.0, (double)free_after / 1073741824.0, (long long)d.nd, (now_ms() - t0) / 1000.0);
        std::fflush(stderr);
        return true;
    }

    bool df2_prepare(std::string& err) {
        if (!df2_ || !df2_->bound) { err = "dflash2: not bound"; return false; }
        return true;
    }

    bool df2_enqueue_ctx_batch(int m, const float* host_taps, int64_t pos0, std::string& err) {
        Df2Ascend& d = *df2_;
        NpuCtx& dc = *npu_[d.dev];
        const size_t tap_dim = d.taps.size() * N;
        const int m_pos_aligned = ((m + 7) / 8) * 8;
        for (int i = 0; i < m_pos_aligned; ++i) {
            d.h_ctx_pos[(size_t)i] = (int32_t)(pos0 + std::min(i, m - 1));
        }
        df2_rope_rows(d.h_ctx_pos.data(), m, d.theta, d.h_ctx_rope.data());

        if (!d.d_taps_f32.upload(host_taps, (size_t)m * tap_dim * 4, err) ||
            !d.d_ctx_pos.upload(d.h_ctx_pos.data(), (size_t)m_pos_aligned * 4, err) ||
            !d.d_ctx_rope.upload(d.h_ctx_rope.data(), (size_t)m * 128 * 4, err)) {
            return false;
        }
        if (!check_acl(aclrtSetDevice(d.dev), "aclrtSetDevice(df2_ctx)", err)) return false;
        strata::ascend_ops::launch_f32_to_f16(dc.stream, d.d_taps_f32.ptr, d.d_taps_f16.ptr, (uint32_t)((size_t)m * tap_dim));
        if (!dc.qgm1(d.d_taps_f16.ptr, d.W_fc, d.d_fcy_f16.ptr, m, err)) return false;
        strata::ascend_ops::launch_df2_rms(
            dc.stream, d.d_fcy_f16.ptr, /*x_is_f16=*/1u, d.hidden_norm.ptr,
            /*y_f32=*/nullptr, d.d_hctx_f16.ptr, (uint32_t)m, d.eps,
            d.rowlog.ptr, d.d_ctx_pos.ptr, (uint32_t)DF2_RING);

        for (int l = 0; l < d.L; ++l) {
            Df2Layer& dl = d.layers[(size_t)l];
            if (!dc.qgm1(d.d_hctx_f16.ptr, dl.W_kv, d.d_kvc_f16.ptr, m, err)) return false;
            strata::ascend_ops::launch_df2_kv_write(
                dc.stream, d.d_kvc_f16.ptr, dl.k_norm.ptr, d.d_ctx_pos.ptr, d.d_ctx_rope.ptr,
                d.kring[l].ptr, d.vring[l].ptr, (uint32_t)m, (uint32_t)DF2_RING, d.eps);
        }
        return true;
    }

    bool df2_context_ptr(const float* taps, int64_t n, int64_t pos0, std::string& err) {
        if (!df2_ || !df2_->bound) { err = "dflash2: not bound"; return false; }
        Df2Ascend& d = *df2_;
        NpuCtx& dc = *npu_[d.dev];
        const size_t tap_dim = d.taps.size() * N;
        for (int64_t b0 = 0; b0 < n; b0 += DF2_CTX_MAX) {
            const int m = (int)std::min<int64_t>(DF2_CTX_MAX, n - b0);
            if (!df2_enqueue_ctx_batch(m, taps + (size_t)b0 * tap_dim, pos0 + b0, err)) return false;
            if (!dc.sync(err)) return false;
        }
        return true;
    }

    int64_t df2_restore_lineage(const std::vector<int32_t>& cur_hist, int64_t cur_hi, const int32_t* ids, int64_t resume, std::string& err) {
        if (!df2_ || !df2_->bound || df2_->stash_max <= 0 || resume <= 0) return 0;
        Df2Ascend& d = *df2_;
        NpuCtx& dc = *npu_[d.dev];
        const size_t log_elems = (size_t)DF2_RING * N;
        if (cur_hi > 0 && !cur_hist.empty()) {
            Df2Ascend::Stash s;
            s.hist = cur_hist; s.hi = cur_hi; s.floor = d.floor; s.used = ++d.stash_tick;
            s.host_log.resize(log_elems);
            if (d.rowlog.download(s.host_log.data(), log_elems * 2, err)) {
                d.stash.push_back(std::move(s));
            }
            err.clear();
        }
        const int64_t lo = std::max<int64_t>(0, resume - (d.window - 1));
        int best = -1;
        for (size_t i = 0; i < d.stash.size(); ++i) {
            const auto& s = d.stash[i];
            if (s.hi + 256 < resume || (int64_t)s.hist.size() < resume || s.hi - DF2_RING > lo) continue;
            int64_t k = 0;
            while (k < resume && s.hist[(size_t)k] == ids[k]) ++k;
            if (k < resume) continue;
            if (best < 0 || s.used > d.stash[(size_t)best].used) best = (int)i;
        }
        Df2Ascend::Stash match;
        if (best >= 0) {
            match = std::move(d.stash[(size_t)best]);
            d.stash.erase(d.stash.begin() + best);
        }
        while ((int)d.stash.size() > d.stash_max) {
            size_t old = 0;
            for (size_t i = 1; i < d.stash.size(); ++i) if (d.stash[i].used < d.stash[old].used) old = i;
            d.stash.erase(d.stash.begin() + old);
        }
        if (best < 0) return 0;
        if (!d.rowlog.upload(match.host_log.data(), log_elems * 2, err)) return 0;

        const int64_t r_end = std::min<int64_t>(resume, match.hi);
        const int64_t from = std::max<int64_t>(lo, match.floor);
        if (from >= r_end) return 0;

        for (int64_t b0 = from; b0 < r_end; b0 += DF2_CTX_MAX) {
            const int m = (int)std::min<int64_t>(DF2_CTX_MAX, r_end - b0);
            const int m_pos_aligned = ((m + 7) / 8) * 8;
            for (int i = 0; i < m_pos_aligned; ++i) d.h_ctx_pos[(size_t)i] = (int32_t)(b0 + std::min(i, m - 1));
            df2_rope_rows(d.h_ctx_pos.data(), m, d.theta, d.h_ctx_rope.data());
            if (!d.d_ctx_pos.upload(d.h_ctx_pos.data(), (size_t)m_pos_aligned * 4, err) ||
                !d.d_ctx_rope.upload(d.h_ctx_rope.data(), (size_t)m * 128 * 4, err)) return 0;
            if (!check_acl(aclrtSetDevice(d.dev), "aclrtSetDevice(df2_rebuild)", err)) return 0;
            strata::ascend_ops::launch_df2_gather_log(dc.stream, d.rowlog.ptr, d.d_ctx_pos.ptr, d.d_hctx_f16.ptr, (uint32_t)m, (uint32_t)DF2_RING);
            for (int l = 0; l < d.L; ++l) {
                Df2Layer& dl = d.layers[(size_t)l];
                if (!dc.qgm1(d.d_hctx_f16.ptr, dl.W_kv, d.d_kvc_f16.ptr, m, err)) return 0;
                strata::ascend_ops::launch_df2_kv_write(
                    dc.stream, d.d_kvc_f16.ptr, dl.k_norm.ptr, d.d_ctx_pos.ptr, d.d_ctx_rope.ptr,
                    d.kring[l].ptr, d.vring[l].ptr, (uint32_t)m, (uint32_t)DF2_RING, d.eps);
            }
            if (!dc.sync(err)) return 0;
        }
        d.floor = from;
        return r_end - from;
    }

    bool df2_draft(const float* taps, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) {
        if (!df2_ || !df2_->bound) { err = "dflash2: not bound"; return false; }
        if (n_new < 1 || n_new > DF2_ROWS) { err = "dflash2: draft n_new out of range"; return false; }
        Df2Ascend& d = *df2_;
        NpuCtx& dc = *npu_[d.dev];
        const double t0 = now_ms();

        const float* src_taps = taps ? taps : h_tap_dec_.data();
        if (!df2_enqueue_ctx_batch(n_new, src_taps, pos0, err)) return false;

        // Stage block embeddings (anchor + 7 mask tokens) and block RoPE table for positions s .. s + 7
        const int64_t s = pos0 + n_new;
        int32_t bpos[DF2_ROWS];
        for (int i = 0; i < DF2_ROWS; ++i) bpos[i] = (int32_t)(s + i);
        df2_rope_rows(bpos, DF2_ROWS, d.theta, d.h_blk_rope.data());

        const int64_t anc_id = std::max<int64_t>(0, std::min<int64_t>(n_vocab_ - 1, anchor));
        std::memcpy(d.h_blk_h.data(), h_tok_embd_f32_.data() + (size_t)anc_id * N, (size_t)N * 4);
        for (int i = 1; i < DF2_ROWS; ++i) {
            std::memcpy(d.h_blk_h.data() + (size_t)i * N, d.mask_emb.data(), (size_t)N * 4);
        }
        if (!d.d_h_f32.upload(d.h_blk_h.data(), (size_t)DF2_ROWS * N * 4, err) ||
            !d.d_blk_rope.upload(d.h_blk_rope.data(), (size_t)DF2_ROWS * 128 * 4, err)) {
            return false;
        }

        const int64_t lo = std::min<int64_t>(s, std::max<int64_t>(std::max<int64_t>(0, s - (d.window - 1)), d.floor));

        // Enqueue 5-layer bidirectional block transformer on dc.stream (back-to-back with context update!)
        for (int l = 0; l < d.L; ++l) {
            Df2Layer& dl = d.layers[(size_t)l];
            const float* base_a = reinterpret_cast<const float*>(dl.attn_base.ptr);
            const float* base_m = reinterpret_cast<const float*>(dl.mlp_base.ptr);

            strata::ascend_ops::launch_df2_rms(
                dc.stream, d.d_h_f32.ptr, /*x_is_f16=*/0u, dl.in_norm.ptr,
                d.d_xn_f32.ptr, d.d_xn_f16.ptr, DF2_ROWS, d.eps);
            if (!dc.mm(d.d_xn_f16.ptr, dl.W_attn_conv, d.d_dyn_f16.ptr, DF2_ROWS, err)) return false;
            strata::ascend_ops::launch_df2_conv(
                dc.stream, d.d_xn_f32.ptr, /*x_is_f16=*/0u, d.d_dyn_f16.ptr, /*dyn_col_off=*/0u,
                base_a, /*resid=*/nullptr, /*out_f32=*/nullptr, d.d_xc_f16.ptr, DF2_ROWS);
            if (!dc.qgm1(d.d_xc_f16.ptr, dl.W_qkv, d.d_qkv_f16.ptr, DF2_ROWS, err)) return false;
            strata::ascend_ops::launch_df2_qk_prep(
                dc.stream, d.d_qkv_f16.ptr, dl.q_norm.ptr, dl.k_norm.ptr, d.d_blk_rope.ptr,
                d.d_qn_f32.ptr, d.d_kblk_f32.ptr, d.d_vblk_f32.ptr, DF2_ROWS, d.eps);
            strata::ascend_ops::launch_df2_attn(
                dc.stream, d.d_qn_f32.ptr, d.kring[l].ptr, d.vring[l].ptr,
                d.d_kblk_f32.ptr, d.d_vblk_f32.ptr, d.d_ao_f16.ptr,
                (uint32_t)s, (uint32_t)lo, (uint32_t)DF2_RING, (uint32_t)d.window);
            if (!dc.qgm1(d.d_ao_f16.ptr, dl.W_o, d.d_o_f16.ptr, DF2_ROWS, err)) return false;
            strata::ascend_ops::launch_df2_conv(
                dc.stream, d.d_o_f16.ptr, /*x_is_f16=*/1u, d.d_dyn_f16.ptr, /*dyn_col_off=*/320u,
                base_a + 2 * N, d.d_h_f32.ptr, d.d_h_f32.ptr, /*out_f16=*/nullptr, DF2_ROWS);

            strata::ascend_ops::launch_df2_rms(
                dc.stream, d.d_h_f32.ptr, /*x_is_f16=*/0u, dl.post_norm.ptr,
                d.d_xn_f32.ptr, d.d_xn_f16.ptr, DF2_ROWS, d.eps);
            if (!dc.mm(d.d_xn_f16.ptr, dl.W_mlp_conv, d.d_dyn_f16.ptr, DF2_ROWS, err)) return false;
            strata::ascend_ops::launch_df2_conv(
                dc.stream, d.d_xn_f32.ptr, /*x_is_f16=*/0u, d.d_dyn_f16.ptr, /*dyn_col_off=*/0u,
                base_m, /*resid=*/nullptr, /*out_f32=*/nullptr, d.d_xc_f16.ptr, DF2_ROWS);
            if (!dc.qgm1(d.d_xc_f16.ptr, dl.W_gu, d.d_gu_f16.ptr, DF2_ROWS, err)) return false;
            strata::ascend_ops::launch_df2_swiglu_f16(dc.stream, d.d_gu_f16.ptr, d.d_sw_f16.ptr, DF2_ROWS, (uint32_t)d.I);
            if (!dc.qgm1(d.d_sw_f16.ptr, dl.W_down, d.d_mlp_f16.ptr, DF2_ROWS, err)) return false;
            strata::ascend_ops::launch_df2_conv(
                dc.stream, d.d_mlp_f16.ptr, /*x_is_f16=*/1u, d.d_dyn_f16.ptr, /*dyn_col_off=*/320u,
                base_m + 2 * N, d.d_h_f32.ptr, d.d_h_f32.ptr, /*out_f16=*/nullptr, DF2_ROWS);
        }

        // Final norm -> INT8 Cube Draft Head + Top-16 + Candidate Selector Projection over rows 1..ND
        strata::ascend_ops::launch_df2_rms(
            dc.stream, d.d_h_f32.ptr, /*x_is_f16=*/0u, d.final_norm.ptr,
            /*y_f32=*/nullptr, d.d_fn_f16.ptr, DF2_ROWS, d.eps);

        const int ND = d.ndraft;
        const uint16_t* fn_rows1 = reinterpret_cast<const uint16_t*>(d.d_fn_f16.ptr) + N;
        if (!dc.qgm1_chunked(fn_rows1, d.W_dhead, d.d_logits_f16.ptr, ND, err)) return false;
        if (!dc.mm(fn_rows1, d.W_sel_proj, d.d_hsel_f16.ptr, ND, err)) return false;
        strata::ascend_ops::launch_topk16_f16(
            dc.stream, d.d_logits_f16.ptr, d.d_tk_pv.ptr, d.d_tk_pi.ptr,
            d.d_topval.ptr, d.d_toptok.ptr, d.d_vmap.ptr,
            (uint32_t)ND, (uint32_t)d.nd, (uint32_t)d.nd_pad, d.use_map ? 1u : 0u);

        if (!dc.sync(err)) return false;

        if (!d.d_topval.download(d.h_topval.data(), (size_t)ND * 16 * 4, err) ||
            !d.d_toptok.download(d.h_toptok.data(), (size_t)ND * 16 * 4, err) ||
            !d.d_hsel_f16.download(d.h_hsel.data(), (size_t)ND * 256 * 2, err)) {
            return false;
        }

        // Exact IEEE FP32 left-to-right bilinear selector walk in Host RAM (~3 us!)
        int32_t prev = anchor;
        float vp[256], sv[256], sc[16];
        for (int i = 0; i < ND; ++i) {
            dequant_cb_row256(d.pred_cb_type, d.pred_cb_raw.data(), prev, vp);
            const uint16_t* hs = d.h_hsel.data() + (size_t)i * 256;
            for (int r = 0; r < 256; ++r) vp[r] *= ref::f16_to_f32(hs[r]);

            int bj = 0;
            float bv = -1e30f;
            for (int j = 0; j < 16; ++j) {
                const int32_t cand_tok = d.h_toptok[(size_t)i * 16 + (size_t)j];
                dequant_cb_row256(d.succ_cb_type, d.succ_cb_raw.data(), cand_tok, sv);
                float dot = 0.0f;
                for (int r = 0; r < 256; ++r) dot += sv[r] * vp[r];
                sc[j] = d.h_topval[(size_t)i * 16 + (size_t)j] + dot;
                if (sc[j] > bv) { bv = sc[j]; bj = j; }
            }
            float z = 0.0f;
            for (int j = 0; j < 16; ++j) z += std::exp(sc[j] - bv);
            drafts[i] = d.h_toptok[(size_t)i * 16 + (size_t)bj];
            if (probs) probs[i] = 1.0f / z;
            prev = drafts[i];
        }
        for (int i = ND; i < DF2_DRAFTS; ++i) {
            drafts[i] = 0;
            if (probs) probs[i] = 0.0f;
        }

        const double dt = now_ms() - t0;
        d.gpu_ms += dt;
        ++d.rounds;
        static const bool df2_stats = std::getenv("STRATA_DF2_STATS") != nullptr;
        if (df2_stats && d.rounds % 50 == 0) {
            std::fprintf(stderr, "strata ascend: DFlash2 after %lld rounds: %.2f ms/round avg (last %.2f ms)\n",
                         (long long)d.rounds, d.gpu_ms / (double)d.rounds, dt);
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
        const bool greedy = sp.greedy || sp.temperature <= 0.0f;
        const float* bias_host = sp.logit_bias ? (const float*)strata::gpu::host_view(sp.logit_bias) : nullptr;
        const int32_t* hist_host = (hist_in && hl > 0 && sp.penalty_last_n > 0 &&
                                    (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f))
                                       ? (const int32_t*)strata::gpu::host_view(hist_in)
                                       : nullptr;
        const int pen_n = hist_host ? std::min(sp.penalty_last_n, hl) : 0;

        // Fast zero-copy path: NPU launch_argmax_f16 already computed exact tok_out_[0..last_T_-1] in run_forward!
        if (greedy && !bias_host && pen_n == 0) {
            return true;
        }

        if (!ensure_host_logits(row0 + n, err)) return false;

        std::vector<float> row;
        if (bias_host || pen_n > 0) row.resize((size_t)n_vocab_);
        std::vector<int> pen_cnt;
        if (pen_n > 0) pen_cnt.assign((size_t)n_vocab_, 0);

        for (int r = 0; r < n; ++r) {
            const float* src = h_logits_f32_.data() + (size_t)(row0 + r) * vocab_pad_;
            const float* rptr = src;
            if (bias_host || pen_n > 0) {
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
                rptr = row.data();
            }

            if (greedy) {
                int32_t best_i = 0;
                float best_v = rptr[0];
                for (int64_t i = 1; i < n_vocab_; ++i) {
                    if (rptr[(size_t)i] > best_v) {
                        best_v = rptr[(size_t)i];
                        best_i = (int32_t)i;
                    }
                }
                tok_out_[row0 + r] = best_i;
                continue;
            }

            const size_t kk = (size_t)((sp.top_k > 0 && sp.top_k < 64) ? sp.top_k : 64);
            std::pair<float, int32_t> cand[64];
            for (size_t k = 0; k < kk; ++k) cand[k] = {-1e30f, 0x7fffffff};
            float min_v = -1e30f;
            for (int64_t i = 0; i < n_vocab_; ++i) {
                const float v = rptr[(size_t)i];
                if (v <= min_v) continue; // ascending index scan skips >99.9% of vocabulary in 1 cycle!
                size_t pos = kk - 1;
                while (pos > 0 && v > cand[pos - 1].first) {
                    cand[pos] = cand[pos - 1];
                    --pos;
                }
                cand[pos] = {v, (int32_t)i};
                min_v = cand[kk - 1].first;
            }

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
    int max_t_ = 1;
    int nexp_ = NEXP;
    int64_t max_cells_ = 0;
    int64_t max_blocks_ = 0;
    int64_t n_vocab_ = 0;
    int64_t vocab_pad_ = 0;
    float eps_ = 1e-6f;

    std::vector<float> h_tok_embd_f32_;
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
    NzMat W_out_hc_dn_, W_out_hc_up_;
    Int8ChunkedMat W_lm_head_;
    DevMem d_lm_f16_, d_lm_f32_, d_arg_pv_, d_arg_pi_, d_tok_out_;

    HostPinned<float> h_R_, h_bo_, h_inj_, h_ple_emb_, h_logits_f32_;
    HostPinned<uint16_t> h_ple_kv_f16_;
    HostPinned<int32_t> h_tok_buf_;

    // DFlash2 taps & drafter state
    int df2_dev_ = 0;
    bool taps_on_ = false;
    std::vector<int> tap_layers_;
    int tap_slot_[64] = {};
    int tap_dev_slot_[64] = {};
    std::vector<int> tap_dev_of_, tap_dev_local_;
    int64_t tap_first_ = 0;
    std::function<bool(const float*, int64_t, int64_t, std::string&)> tap_cb_;
    HostPinned<float> h_tap_raw_, h_tap_dec_, h_tap_pro_;
    std::unique_ptr<Df2Ascend> df2_;

    int hp_ = 0;
    int pend_fold_hp_ = 0;
    int pend_fold_dev_[8] = {};
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

bool window_taps(void* w, const std::vector<int>& layers, const float** tap_dev, std::string& err) {
    auto* x = (WindowAscend*)w;
    if (!x->taps_enable(layers, err)) return false;
    *tap_dev = x->tap_dec_ptr();
    return true;
}

bool prompt_taps(const std::vector<int>& layers, int64_t first_pos,
                 const std::function<bool(const float*, int64_t, int64_t, std::string&)>& cb,
                 std::string& err) {
    if (!g_the_window || !g_the_window->ready()) {
        err = "ascend prompt: decode window is not initialized";
        return false;
    }
    if (cb && !g_the_window->taps_enable(layers, err)) return false;
    g_the_window->taps_set_cb(first_pos, cb);
    return true;
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

bool Dflash2Drafter::load(const std::string& pack_path, int device, std::string& err) {
    device_ = device;
    std::vector<int> taps;
    std::map<std::string, gguf::Tensor> ten;
    std::map<std::string, std::string> cfg;
    uint64_t total = 0, data_off = 0;
    if (!strata::metal_engine::df2_read_index(pack_path, taps, ten, total, data_off, cfg, err)) return false;
    taps_ = taps;
    if (cfg.count("mask_token")) mask_token_ = std::atoi(cfg["mask_token"].c_str());
    strata::metal_engine::g_df2_pack_path = pack_path;
    return true;
}

bool Dflash2Drafter::bind(const WeightTable&, const NativeHead*, const std::string& vocab_file, std::string& err, int) {
    strata::metal_engine::WindowAscend* w = strata::metal_engine::g_the_window;
    if (w == nullptr || !w->ready()) {
        err = "dflash2: Ascend decode window is not initialized";
        return false;
    }
    if (bound_) return true;
    if (!w->df2_ensure(strata::metal_engine::g_df2_pack_path, err) || !w->df2_bind(vocab_file, err)) return false;
    w->df2_set_cap(ndraft_);
    device_ = w->df2_dev();
    vram_ = w->df2_bytes();
    bound_ = true;
    return true;
}

bool Dflash2Drafter::bind_raw(EmbedFn, const void*, int, int64_t, const std::vector<int32_t>&, std::string& err) {
    err = "dflash2: bind_raw is a test hook of the CUDA / HIP build";
    return false;
}

bool Dflash2Drafter::clone_alloc(const Dflash2Drafter&, std::string& err) { err = "dflash2: batch slot drafters are not enabled on Ascend"; return false; }
bool Dflash2Drafter::clone_bind(const Dflash2Drafter&, std::string& err) { err = "dflash2: batch slot drafters are not enabled on Ascend"; return false; }
bool Dflash2Drafter::copy_state_from(Dflash2Drafter&, std::string& err) { err = "dflash2: copy_state_from is not enabled on Ascend"; return false; }
float* Dflash2Drafter::defer_rows(int64_t, int, std::string& err) { err = "dflash2: deferred rows are not enabled on Ascend"; return nullptr; }
bool Dflash2Drafter::catch_up(int64_t, std::string&) { return true; }

void Dflash2Drafter::reset() {
    have_ctx_ = 0;
    cur_.hist.clear();
    ++epoch_;
    ring_hi_ = 0;
    if (strata::metal_engine::WindowAscend* w = strata::metal_engine::g_the_window) w->df2_set_floor(0);
}

bool Dflash2Drafter::begin_request(const int32_t* ids, int64_t n, int64_t resume, bool* cold) {
    resume = std::max<int64_t>(0, std::min(resume, n));
    int64_t lcp = 0;
    const int64_t lim = std::min<int64_t>((int64_t)cur_.hist.size(), resume);
    while (lcp < lim && cur_.hist[(size_t)lcp] == ids[lcp]) ++lcp;
    const bool stale = ring_hi_ > lcp && lcp < resume;
    bool rebuilt = false;
    last_rebuilt = 0;
    last_missing = 0;
    last_swapped = false;
    if (stale) {
        const double t0 = strata::metal_engine::now_ms();
        std::string rerr;
        if (strata::metal_engine::WindowAscend* w = strata::metal_engine::g_the_window) {
            const int64_t nr = w->df2_restore_lineage(cur_.hist, ring_hi_, ids, resume, rerr);
            if (nr > 0) {
                rebuilt = true;
                last_rebuilt = nr;
                last_swapped = true;
                last_ms = strata::metal_engine::now_ms() - t0;
            }
        }
        ++epoch_;
        ring_hi_ = rebuilt ? resume : 0;
    }
    if (rebuilt) {
        if (cold) *cold = false;
        cur_.hist.assign(ids, ids + n);
        return true;
    }
    if (strata::metal_engine::WindowAscend* w = strata::metal_engine::g_the_window) {
        w->df2_set_floor(stale ? resume : std::min<int64_t>(w->df2_floor(), resume));
    }
    if (cold) *cold = stale;
    cur_.hist.assign(ids, ids + n);
    return true;
}

void Dflash2Drafter::init_ring_log() {}
void Dflash2Drafter::copy_log_state_from(const Dflash2Drafter& src) { cur_.hist = src.cur_.hist; epoch_ = src.epoch_; ring_hi_ = src.ring_hi_; }
uint64_t Dflash2Drafter::ring_log_bytes() const { return 0; }
uint64_t Dflash2Drafter::ring_stash_bytes() const { return 0; }

void Dflash2Drafter::note_rows(const int32_t* toks, int64_t n, int64_t pos0) {
    if (pos0 < 0) return;
    if ((int64_t)cur_.hist.size() > pos0) cur_.hist.resize((size_t)pos0);
    if ((int64_t)cur_.hist.size() < pos0) return;
    cur_.hist.insert(cur_.hist.end(), toks, toks + n);
}

bool Dflash2Drafter::context(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    strata::metal_engine::WindowAscend* w = strata::metal_engine::g_the_window;
    if (w == nullptr || !w->ready()) { err = "dflash2: Ascend window not ready"; return false; }
    if (!bound_) {
        if (!w->df2_ensure(strata::metal_engine::g_df2_pack_path, err) || !w->df2_bind("", err)) return false;
        w->df2_set_cap(ndraft_);
        device_ = w->df2_dev();
        vram_ = w->df2_bytes();
        bound_ = true;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!w->df2_context_ptr(taps, n, pos0, err)) return false;
    ms_ctx += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n);
    return true;
}

bool Dflash2Drafter::context_async(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    return context(taps, n, pos0, err);
}

bool Dflash2Drafter::context_wait(std::string&) { return true; }

bool Dflash2Drafter::draft(const float* taps, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) {
    strata::metal_engine::WindowAscend* w = strata::metal_engine::g_the_window;
    if (w == nullptr || !bound_) { err = "dflash2: not bound"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    if (!w->df2_draft(taps, n_new, pos0, anchor, drafts, probs, err)) return false;
    ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ms_gpu = w->df2_gpu_ms();
    ++rounds;
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n_new);
    ring_hi_ = std::max<int64_t>(ring_hi_, pos0 + n_new);
    return true;
}

bool Dflash2Drafter::draft_launch(const float*, int, int64_t, int32_t, std::string& err) { err = "dflash2: split draft is not enabled on Ascend"; return false; }
bool Dflash2Drafter::draft_finish(int32_t*, float*, std::string& err) { err = "dflash2: split draft is not enabled on Ascend"; return false; }

bool Dflash2Drafter::prepare(const float*, std::string& err) {
    strata::metal_engine::WindowAscend* w = strata::metal_engine::g_the_window;
    if (w == nullptr || !bound_) { err = "dflash2: not bound"; return false; }
    return w->df2_prepare(err);
}

void Dflash2Drafter::set_max_drafts(int n) {
    n = std::max(1, std::min(n, (int)kDrafts));
    ndraft_ = n;
    if (strata::metal_engine::WindowAscend* w = strata::metal_engine::g_the_window) w->df2_set_cap(n);
}

bool Dflash2Drafter::debug_rows(std::vector<float>&, std::string& err) { err = "dflash2: debug_rows not enabled on Ascend"; return false; }
bool Dflash2Drafter::debug_ring(std::vector<float>&, std::vector<int64_t>&, std::string& err) { err = "dflash2: debug_ring not enabled on Ascend"; return false; }

}  // namespace strata::core
