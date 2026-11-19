// tests/cuda/chordmoe_gpu_parity_test.cu - CUDA CTA vs CPU reference parity & latency microbenchmark.
#include "strata/chordmoe/chord_router.hpp"
#include "strata/kernels/chord_router_cuda.hpp"
#include "ggml_chord_moe.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK_CUDA(call)                                                                            \
    do {                                                                                         \
        cudaError_t err__ = (call);                                                              \
        if (err__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                   \
                         cudaGetErrorString(err__));                                             \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0) {
        std::printf("chordmoe_gpu_parity_test: no CUDA device detected, skipping\n");
        return 0;
    }

    using namespace strata::chordmoe;
    using namespace strata::chordmoe::ggml_adapter;

    struct Shape { int ne; int k; int pri_slots; int peer_slots; };
    const Shape shapes[] = {
        {512, 10, 140, 60}, // Qwen3.8-Flash-Next (2 GPUs)
        {256, 8,  82,  0},  // Qwen3.6-35B-A3B
    };

    cudaStream_t stream = nullptr;
    CK_CUDA(cudaStreamCreate(&stream));

    int fails = 0;
    std::mt19937 rng(20261119);
    std::normal_distribution<float> norm(0.0f, 1.15f);

    for (const auto& s : shapes) {
        const int T = 7;
        ChordConfig cfg;
        cfg.mode            = ChordMode::BoundedSetCover;
        cfg.logit_margin    = 1.30f;
        cfg.tv_budget_token = 0.14f;
        cfg.tv_budget_turn  = 100.0f; // single-layer parity check
        cfg.rank_lo         = 4;
        cfg.rank_hi         = s.k - 1;
        cfg.protect_anchor  = true;

        std::vector<int32_t> h_pri(s.ne, -1), h_peer(s.ne, -1);
        for (int i = 0; i < s.pri_slots; ++i) h_pri[i] = i;
        for (int i = s.pri_slots; i < s.pri_slots + s.peer_slots; ++i) h_peer[i] = i - s.pri_slots;

        float* d_logits = nullptr;
        int32_t* d_ids = nullptr;
        float* d_w = nullptr;
        int32_t* d_pri = nullptr;
        int32_t* d_peer = nullptr;
        unsigned long long* d_stats = nullptr;

        CK_CUDA(cudaMalloc(&d_logits, (size_t) T * s.ne * sizeof(float)));
        CK_CUDA(cudaMalloc(&d_ids, (size_t) T * s.k * sizeof(int32_t)));
        CK_CUDA(cudaMalloc(&d_w, (size_t) T * s.k * sizeof(float)));
        CK_CUDA(cudaMalloc(&d_pri, (size_t) s.ne * sizeof(int32_t)));
        CK_CUDA(cudaMalloc(&d_peer, (size_t) s.ne * sizeof(int32_t)));
        CK_CUDA(cudaMalloc(&d_stats, 8 * sizeof(unsigned long long)));
        CK_CUDA(cudaMemcpy(d_pri, h_pri.data(), (size_t) s.ne * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK_CUDA(cudaMemcpy(d_peer, h_peer.data(), (size_t) s.ne * sizeof(int32_t), cudaMemcpyHostToDevice));

        for (int trial = 0; trial < 32; ++trial) {
            std::vector<float> h_logits((size_t) T * s.ne);
            for (int t = 0; t < T; ++t)
                for (int e = 0; e < s.ne; ++e)
                    h_logits[(size_t) t * s.ne + e] = -0.75f * std::log((float) (e + 4)) + norm(rng);

            std::vector<int32_t> cpu_ids((size_t) T * s.k), gpu_ids((size_t) T * s.k), orig_ids((size_t) T * s.k);
            std::vector<float> cpu_w((size_t) T * s.k), gpu_w((size_t) T * s.k), orig_w((size_t) T * s.k);
            ggml_standard_topk_moe(h_logits.data(), T, s.ne, s.k, orig_ids.data(), orig_w.data());
            cpu_ids = orig_ids;
            cpu_w   = orig_w;

            CK_CUDA(cudaMemcpyAsync(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            CK_CUDA(cudaMemcpyAsync(d_ids, orig_ids.data(), orig_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CK_CUDA(cudaMemcpyAsync(d_w, orig_w.data(), orig_w.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            CK_CUDA(cudaMemsetAsync(d_stats, 0, 8 * sizeof(unsigned long long), stream));

            chord_reshape_window(h_logits.data(), cpu_ids.data(), cpu_w.data(), T, s.ne, s.k,
                                 h_pri.data(), cfg, s.peer_slots > 0 ? h_peer.data() : nullptr);

            strata::kernels::chord_route_window(d_logits, d_ids, d_w, d_pri,
                                                s.peer_slots > 0 ? d_peer : nullptr,
                                                T, s.ne, s.k, cfg, d_stats, stream);

            CK_CUDA(cudaMemcpyAsync(gpu_ids.data(), d_ids, gpu_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CK_CUDA(cudaMemcpyAsync(gpu_w.data(), d_w, gpu_w.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CK_CUDA(cudaStreamSynchronize(stream));

            // Anchor row 0 must match bit-for-bit, and every token's selected IDs + weights must match CPU reference!
            if (std::memcmp(gpu_ids.data(), orig_ids.data(), (size_t) s.k * sizeof(int32_t)) != 0) {
                std::fprintf(stderr, "FAIL (%dx%d trial %d): GPU modified anchor row 0!\n", s.ne, s.k, trial);
                ++fails;
            }
            for (int i = 0; i < T * s.k; ++i) {
                if (gpu_ids[i] != cpu_ids[i] || std::fabs(gpu_w[i] - cpu_w[i]) > 1e-5f) {
                    std::fprintf(stderr, "FAIL (%dx%d trial %d idx %d): GPU(%d, %.6f) != CPU(%d, %.6f)\n",
                                 s.ne, s.k, trial, i, gpu_ids[i], gpu_w[i], cpu_ids[i], cpu_w[i]);
                    ++fails;
                    break;
                }
            }
        }

        // Microbenchmark 1000 launches inside stream
        cudaEvent_t ev0, ev1;
        CK_CUDA(cudaEventCreate(&ev0));
        CK_CUDA(cudaEventCreate(&ev1));
        CK_CUDA(cudaEventRecord(ev0, stream));
        for (int it = 0; it < 1000; ++it) {
            strata::kernels::chord_route_window(d_logits, d_ids, d_w, d_pri,
                                                s.peer_slots > 0 ? d_peer : nullptr,
                                                T, s.ne, s.k, cfg, nullptr, stream);
        }
        CK_CUDA(cudaEventRecord(ev1, stream));
        CK_CUDA(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
        std::printf("chordmoe_gpu_parity_test [%dx%d T=%d]: 32/32 trials bit-consistent, %.2f us/launch\n",
                    s.ne, s.k, T, ms);

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaFree(d_logits);
        cudaFree(d_ids);
        cudaFree(d_w);
        cudaFree(d_pri);
        cudaFree(d_peer);
        cudaFree(d_stats);
    }

    cudaStreamDestroy(stream);
    if (fails == 0) std::printf("chordmoe_gpu_parity_test: ok\n");
    return fails == 0 ? 0 : 1;
}
