// ascend/src/gpu/runtime.cpp - strata::gpu runtime shim over Huawei Ascend CANN (aclrt).
//
// In the Window-hook architecture (STRATA_GPU_METAL=1), the shared host code (generate.cpp, session.cpp, expert_cache.cpp)
// allocates bookkeeping arenas that the Window engine itself replaces with its own NPU buffers (Window::bind_session).
// Keeping those host-visible allocations backed by 64-byte aligned host memory (while reporting real Ascend 310P3 device
// count, properties, and HBM capacity via aclrt) satisfies host_view() (device.cu poison_, snapshots, mrope) with zero
// wasted NPU HBM, leaving 100% of NPU HBM for WindowAscend's FRACTAL_NZ dense weights, Q2_0 experts, and GDN/QSA states.
#include "strata/gpu/runtime.hpp"

#include "acl/acl.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace strata::gpu {

struct StreamImpl {
    int dev = 0;
    GraphImpl* rec = nullptr;
};

struct EventImpl {
    int dummy = 0;
};

struct Op {
    enum Kind { Copy, Fill, Host } kind = Copy;
    void* dst = nullptr;
    const void* src = nullptr;
    size_t bytes = 0;
    int val = 0;
    HostFn fn = nullptr;
    void* arg = nullptr;
};

struct GraphImpl {
    std::vector<Op> ops;
};

struct GraphExecImpl {
    std::vector<Op> ops;
};

namespace {

struct AllocInfo {
    uint8_t* ptr = nullptr;
    size_t bytes = 0;
    bool owned = true;
};

struct RuntimeState {
    std::mutex mu;
    bool acl_inited = false;
    int dev_count = 0;
    std::map<uintptr_t, AllocInfo> allocs;
    StreamImpl legacy{};

    void ensure_acl() {
        std::lock_guard<std::mutex> lk(mu);
        if (acl_inited) return;
        aclError e = aclInit(nullptr);
        // ACL_ERROR_REPEAT_INITIALIZE = 100002 is harmless if already initialized
        if (e != ACL_SUCCESS && e != 100002) {
            std::fprintf(stderr, "strata ascend: aclInit returned %d\n", (int) e);
        }
        uint32_t cnt = 0;
        if (aclrtGetDeviceCount(&cnt) == ACL_SUCCESS) dev_count = (int) cnt;
        if (dev_count > 0) (void) aclrtSetDevice(0);
        acl_inited = true;
    }
};

RuntimeState& st() {
    static RuntimeState* s = new RuntimeState();
    return *s;
}

thread_local int t_cur_dev = 0;
thread_local Error t_last_err = kSuccess;

Error set_err(Error e) {
    if (e != kSuccess && e != kNotReady) t_last_err = e;
    return e;
}

uint8_t* resolve_ptr(const void* p) {
    return const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(p));
}

}  // namespace

void ensure_acl_initialized() {
    st().ensure_acl();
}

void* host_view(const void* device_pointer) {
    return const_cast<void*>(device_pointer);
}

Error get_last_error() {
    Error e = t_last_err;
    t_last_err = kSuccess;
    return e;
}

Error peek_last_error() {
    return t_last_err;
}

const char* error_string(Error e) {
    switch (e) {
        case kSuccess: return "no error";
        case kNotReady: return "device not ready";
        case kErrorInvalidValue: return "invalid argument";
        case kErrorMemoryAllocation: return "out of memory";
        case kErrorUnsupported: return "operation not supported on Ascend backend";
        case kErrorLaunchFailure: return "kernel launch failure";
        default: return "ascend runtime error";
    }
}

Error get_device(int* dev) {
    st().ensure_acl();
    if (!dev) return set_err(kErrorInvalidValue);
    *dev = t_cur_dev;
    return kSuccess;
}

Error set_device(int dev) {
    st().ensure_acl();
    if (dev < 0 || (st().dev_count > 0 && dev >= st().dev_count)) return set_err(kErrorInvalidValue);
    t_cur_dev = dev;
    (void) aclrtSetDevice(dev);
    return kSuccess;
}

Error device_count(int* n) {
    st().ensure_acl();
    if (!n) return set_err(kErrorInvalidValue);
    *n = st().dev_count;
    return kSuccess;
}

Error device_synchronize() {
    st().ensure_acl();
    (void) aclrtSynchronizeDevice();
    return kSuccess;
}

Error mem_get_info(size_t* free_b, size_t* total_b) {
    st().ensure_acl();
    if (!free_b || !total_b) return set_err(kErrorInvalidValue);
    size_t f = 0, t = 0;
    (void) aclrtSetDevice(t_cur_dev);
    if (aclrtGetMemInfo(ACL_HBM_MEM, &f, &t) == ACL_SUCCESS && t > 0) {
        *free_b = f;
        *total_b = t;
        return kSuccess;
    }
    *free_b = 40ULL << 30;
    *total_b = 44ULL << 30;
    return kSuccess;
}

Error get_device_properties(DeviceProp* p, int dev) {
    st().ensure_acl();
    if (!p) return set_err(kErrorInvalidValue);
    std::memset(p, 0, sizeof(*p));
    const char* soc = aclrtGetSocName();
    std::snprintf(p->name, sizeof(p->name), "Huawei %s (NPU %d)", soc ? soc : "Ascend310P3", dev);
    p->major = 8;
    p->minor = 0;
    p->warpSize = 32;
    p->multiProcessorCount = 8;
    p->integrated = 0;
    size_t f = 0, t = 0;
    int prev = t_cur_dev;
    (void) aclrtSetDevice(dev);
    if (aclrtGetMemInfo(ACL_HBM_MEM, &f, &t) == ACL_SUCCESS && t > 0) p->totalGlobalMem = t;
    else p->totalGlobalMem = 44ULL << 30;
    (void) aclrtSetDevice(prev);
    return kSuccess;
}

Error device_integrated(int* v, int) {
    if (!v) return set_err(kErrorInvalidValue);
    *v = 0;
    return kSuccess;
}

Error device_sm_count(int* v, int) {
    if (!v) return set_err(kErrorInvalidValue);
    *v = 8;
    return kSuccess;
}

Error device_clock_khz(int* v, int) {
    if (!v) return set_err(kErrorInvalidValue);
    *v = 1060000;
    return kSuccess;
}

Error runtime_version(int* v) {
    if (!v) return set_err(kErrorInvalidValue);
    *v = 9010;
    return kSuccess;
}

Error device_can_access_peer(int* v, int d, int p) {
    if (!v) return set_err(kErrorInvalidValue);
    int32_t can = 0;
    if (d != p && aclrtDeviceCanAccessPeer(&can, d, p) == ACL_SUCCESS) *v = can;
    else *v = 0;
    return kSuccess;
}

Error device_enable_peer_access(int p, unsigned flags) {
    (void) aclrtDeviceEnablePeerAccess(p, flags);
    return kSuccess;
}

Error malloc_device_raw(void** p, size_t bytes) {
    if (!p) return set_err(kErrorInvalidValue);
    if (bytes == 0) { *p = nullptr; return kSuccess; }
    size_t aligned = ((bytes + 63) & ~size_t(63)) + 64;
    void* mem = nullptr;
    if (posix_memalign(&mem, 64, aligned) != 0 || !mem) return set_err(kErrorMemoryAllocation);
    std::lock_guard<std::mutex> lk(st().mu);
    st().allocs[(uintptr_t) mem] = AllocInfo{(uint8_t*) mem, bytes, true};
    *p = mem;
    return kSuccess;
}

Error free_device(void* p) {
    if (!p) return kSuccess;
    std::lock_guard<std::mutex> lk(st().mu);
    auto it = st().allocs.find((uintptr_t) p);
    if (it != st().allocs.end()) {
        if (it->second.owned) std::free(it->second.ptr);
        st().allocs.erase(it);
    } else {
        std::free(p);
    }
    return kSuccess;
}

Error host_alloc_raw(void** p, size_t bytes, unsigned) {
    return malloc_device_raw(p, bytes);
}

Error free_host(void* p) {
    return free_device(p);
}

Error host_device_pointer_raw(void** dev, void* host) {
    if (!dev || !host) return set_err(kErrorInvalidValue);
    *dev = host;
    return kSuccess;
}

Error host_register(void*, size_t, unsigned) { return kSuccess; }
Error host_unregister(void*) { return kSuccess; }

Error memset_device(void* p, int value, size_t bytes) {
    if (bytes == 0) return kSuccess;
    if (!p) return set_err(kErrorInvalidValue);
    std::memset(resolve_ptr(p), value, bytes);
    return kSuccess;
}

Error memset_async(void* p, int value, size_t bytes, Stream s) {
    if (s && s->rec) {
        s->rec->ops.push_back(Op{Op::Fill, p, nullptr, bytes, value, nullptr, nullptr});
        return kSuccess;
    }
    return memset_device(p, value, bytes);
}

Error memcpy_sync(void* dst, const void* src, size_t bytes, CopyKind) {
    if (bytes == 0) return kSuccess;
    if (!dst || !src) return set_err(kErrorInvalidValue);
    std::memmove(resolve_ptr(dst), resolve_ptr(src), bytes);
    return kSuccess;
}

Error memcpy_async(void* dst, const void* src, size_t bytes, CopyKind k, Stream s) {
    if (s && s->rec) {
        s->rec->ops.push_back(Op{Op::Copy, dst, src, bytes, 0, nullptr, nullptr});
        return kSuccess;
    }
    return memcpy_sync(dst, src, bytes, k);
}

Error memcpy_2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, CopyKind k, Stream s) {
    uint8_t* d = resolve_ptr(dst);
    const uint8_t* sp = resolve_ptr(src);
    for (size_t r = 0; r < height; ++r) {
        Error e = memcpy_async(d + r * dpitch, sp + r * spitch, width, k, s);
        if (e != kSuccess) return e;
    }
    return kSuccess;
}

Error memcpy_peer_async(void* dst, int, const void* src, int, size_t bytes, Stream s) {
    return memcpy_async(dst, src, bytes, kDeviceToDevice, s);
}

Error stream_create(Stream* s, unsigned) {
    if (!s) return set_err(kErrorInvalidValue);
    auto* impl = new StreamImpl();
    impl->dev = t_cur_dev;
    *s = impl;
    return kSuccess;
}

Error stream_destroy(Stream s) {
    delete s;
    return kSuccess;
}

Error stream_synchronize(Stream) { return kSuccess; }
Error stream_query(Stream) { return kSuccess; }
Error stream_wait_event(Stream, Event, unsigned) { return kSuccess; }

Error launch_host_func(Stream s, HostFn fn, void* arg) {
    if (!fn) return set_err(kErrorInvalidValue);
    if (s && s->rec) {
        s->rec->ops.push_back(Op{Op::Host, nullptr, nullptr, 0, 0, fn, arg});
        return kSuccess;
    }
    fn(arg);
    return kSuccess;
}

Error event_create(Event* e, unsigned) {
    if (!e) return set_err(kErrorInvalidValue);
    *e = new EventImpl();
    return kSuccess;
}

Error event_destroy(Event e) {
    delete e;
    return kSuccess;
}

Error event_record(Event, Stream) { return kSuccess; }
Error event_query(Event) { return kSuccess; }
Error event_synchronize(Event) { return kSuccess; }
Error event_elapsed_ms(float* ms, Event, Event) {
    if (ms) *ms = 0.0f;
    return kSuccess;
}

Error begin_capture(Stream s) {
    StreamImpl* si = s ? s : &st().legacy;
    if (si->rec) return set_err(kErrorInvalidValue);
    si->rec = new GraphImpl();
    return kSuccess;
}

Error end_capture(Stream s, Graph* g) {
    StreamImpl* si = s ? s : &st().legacy;
    if (!si->rec || !g) return set_err(kErrorInvalidValue);
    *g = si->rec;
    si->rec = nullptr;
    return kSuccess;
}

Error graph_node_count(Graph g, size_t* n) {
    if (!g || !n) return set_err(kErrorInvalidValue);
    *n = g->ops.size();
    return kSuccess;
}

Error graph_instantiate(GraphExec* x, Graph g) {
    if (!x || !g) return set_err(kErrorInvalidValue);
    auto* ex = new GraphExecImpl();
    ex->ops = g->ops;
    *x = ex;
    return kSuccess;
}

Error graph_launch(GraphExec x, Stream) {
    if (!x) return set_err(kErrorInvalidValue);
    for (const auto& op : x->ops) {
        if (op.kind == Op::Copy && op.bytes > 0) std::memmove(resolve_ptr(op.dst), resolve_ptr(op.src), op.bytes);
        else if (op.kind == Op::Fill && op.bytes > 0) std::memset(resolve_ptr(op.dst), op.val, op.bytes);
        else if (op.kind == Op::Host && op.fn) op.fn(op.arg);
    }
    return kSuccess;
}

Error graph_upload(GraphExec, Stream) { return kSuccess; }
Error graph_destroy(Graph g) { delete g; return kSuccess; }
Error graph_exec_destroy(GraphExec x) { delete x; return kSuccess; }

uint64_t purge_device_allocations(size_t) { return 0; }

}  // namespace strata::gpu
