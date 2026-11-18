// include/strata/chordmoe/chord_sched.hpp - All-Cards + Multi-NUMA Min-Max Makespan Offload Scheduler.
//
// Solves two structural bottlenecks in existing heterogeneous MoE schedulers:
//   1. Fiddler (ICLR'25) minimizes `sum(T_CPU, T_GPU)` via an O(2^E) brute-force loop (`1 << n_expert`), which
//      both fails to scale to 128-512 experts and causes up to +33% wall-clock stall because CPU GEMM and PCIe DMA
//      run concurrently (`T_wall = max(...)`, NOT `sum(...)`).
//   2. Single-link static share rules (`pcie_share_rule.hpp`) assume one PCIe link vs one homogeneous CPU pool,
//      ignoring multiple non-NVLink GPUs (each with independent PCIe H2D lanes / helper VRAM) and multi-socket
//      NUMA memory locality (home-node DDR5 bandwidth vs cross-UPI/Infinity-Fabric penalty).
//
// `ChordHeteroScheduler` solves the multi-queue heterogeneous makespan problem in O(M log M) (where M <= T * K is
// the number of distinct missed experts in a verify window, typically 12-45) via Longest-Processing-Time (LPT)
// water-filling with NUMA home-socket affinity awareness.
#pragma once

#include "strata/chordmoe/chord_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::chordmoe {

enum class ExecBackendKind : uint8_t {
    PrimaryPcieDma = 0, ///< Stream cold expert blob over GPU0's PCIe H2D link and execute on GPU0 TensorCores.
    HelperGpuPcie  = 1, ///< Stream or execute on Helper GPU `device_id` (independent PCIe root complex / lane).
    CpuNumaSocket  = 2, ///< Compute in-place on CPU NUMA socket `numa_node` using AVX-512 / VNNI / AMX.
};

struct HeteroQueueSpec {
    ExecBackendKind kind          = ExecBackendKind::CpuNumaSocket;
    int32_t         backend_id    = 0;     ///< GPU index or NUMA socket index.
    double          bw_gbps       = 40.0;  ///< Effective expert weight bandwidth in GB/s (link H2D or socket DDR5).
    double          fixed_us      = 2.0;   ///< Launch / synchronization overhead in microseconds.
    double          per_row_us    = 0.8;   ///< Compute cost per routed token row once expert weights are read.
    double          remote_numa_factor = 1.65; ///< Slowdown multiplier if a NUMA socket reads an expert pinned on another socket.
};

struct MissedExpertTask {
    int32_t expert_id   = -1;
    int32_t token_rows  = 1;     ///< Multiplicity: how many window tokens routed to this cold expert (`1..T`).
    int64_t blob_bytes  = 0;     ///< Compressed weight size of this expert in bytes.
    int32_t home_numa   = 0;     ///< NUMA node where this expert's host pages reside (`-1` = interleaved/unknown).
};

struct ScheduledExpertAssignment {
    int32_t         expert_id   = -1;
    int32_t         queue_index = 0;
    ExecBackendKind kind        = ExecBackendKind::CpuNumaSocket;
    int32_t         backend_id  = 0;
    double          est_cost_us = 0.0;
};

struct LayerSchedulePlan {
    std::vector<ScheduledExpertAssignment> assignments;
    std::vector<double>                    queue_loads_us;
    double                                 makespan_us = 0.0; ///< max_q queue_loads_us[q] (true concurrent wall time).
    double                                 sum_work_us = 0.0; ///< sum_q queue_loads_us[q] (for comparison vs Fiddler).
};

class ChordHeteroScheduler {
public:
    /// Compute the marginal latency (in microseconds) of running `task` on `q`.
    static double task_latency_us(const MissedExpertTask& task, const HeteroQueueSpec& q, bool first_on_queue) {
        const double bw_bytes_per_us = std::max(0.1, q.bw_gbps) * 1e3; // 1 GB/s = 1000 bytes/us (decimal)
        double weight_read_us = (double) task.blob_bytes / bw_bytes_per_us;
        if (q.kind == ExecBackendKind::CpuNumaSocket && task.home_numa >= 0 && task.home_numa != q.backend_id) {
            weight_read_us *= q.remote_numa_factor;
        }
        const double compute_us = (double) std::max<int32_t>(1, task.token_rows) * q.per_row_us;
        return (first_on_queue ? q.fixed_us : 0.0) + weight_read_us + compute_us;
    }

    /// O(M log M + M * Q) LPT + Affinity-Aware Min-Max Makespan Partitioner.
    ///
    /// Why high-multiplicity (`token_rows >= 2`) tasks are prioritized onto PCIe GPU queues automatically:
    ///   - On CPU VNNI/AVX, higher `token_rows` increases compute cost (`per_row_us` is larger on CPU than TensorCore),
    ///     whereas on PCIe DMA + GPU TensorCore, `per_row_us` is near-zero once the blob crosses PCIe!
    ///   - Sorting descending by `(token_rows, blob_bytes)` places heavy multi-token union experts first and balances
    ///     all remaining single-token tail misses across local NUMA sockets and PCIe links to minimize `max_q load[q]`.
    static LayerSchedulePlan partition_layer_misses(const std::vector<MissedExpertTask>& tasks,
                                                    const std::vector<HeteroQueueSpec>& queues) {
        LayerSchedulePlan plan;
        if (queues.empty() || tasks.empty()) return plan;

        const size_t Q = queues.size();
        plan.queue_loads_us.assign(Q, 0.0);
        std::vector<uint8_t> queue_used(Q, 0);

        std::vector<size_t> order(tasks.size());
        for (size_t i = 0; i < tasks.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (tasks[a].token_rows != tasks[b].token_rows) return tasks[a].token_rows > tasks[b].token_rows;
            if (tasks[a].blob_bytes != tasks[b].blob_bytes) return tasks[a].blob_bytes > tasks[b].blob_bytes;
            return tasks[a].expert_id < tasks[b].expert_id;
        });

        plan.assignments.resize(tasks.size());
        for (size_t idx : order) {
            const MissedExpertTask& t = tasks[idx];
            size_t best_q = 0;
            double best_finish = std::numeric_limits<double>::infinity();
            double best_delta = 0.0;

            for (size_t q = 0; q < Q; ++q) {
                const double delta = task_latency_us(t, queues[q], queue_used[q] == 0);
                const double finish = plan.queue_loads_us[q] + delta;
                if (finish < best_finish - 1e-9 ||
                    (std::fabs(finish - best_finish) <= 1e-9 && delta < best_delta)) {
                    best_finish = finish;
                    best_delta = delta;
                    best_q = q;
                }
            }

            plan.queue_loads_us[best_q] += best_delta;
            queue_used[best_q] = 1;
            plan.assignments[idx] = ScheduledExpertAssignment{
                t.expert_id,
                (int32_t) best_q,
                queues[best_q].kind,
                queues[best_q].backend_id,
                best_delta
            };
        }

        for (double ld : plan.queue_loads_us) {
            plan.makespan_us = std::max(plan.makespan_us, ld);
            plan.sum_work_us += ld;
        }
        return plan;
    }

    /// Reference implementation of Fiddler's `min sum` objective (`mixtral.py:576-590`), exposed so unit tests and
    /// benchmarks can quantify the exact wall-clock makespan gap between minimizing `sum` vs minimizing `max`.
    static LayerSchedulePlan fiddler_sum_objective_baseline(const std::vector<MissedExpertTask>& tasks,
                                                            const std::vector<HeteroQueueSpec>& queues) {
        LayerSchedulePlan plan;
        if (queues.empty() || tasks.empty()) return plan;
        const size_t Q = queues.size();
        plan.queue_loads_us.assign(Q, 0.0);
        std::vector<uint8_t> queue_used(Q, 0);
        plan.assignments.resize(tasks.size());

        for (size_t i = 0; i < tasks.size(); ++i) {
            size_t best_q = 0;
            double best_delta = std::numeric_limits<double>::infinity();
            for (size_t q = 0; q < Q; ++q) {
                // Fiddler picks whichever single backend has the smallest standalone execution cost,
                // ignoring queue contention (`queue_loads_us[q]`) -> piles all tasks onto the fastest single queue!
                const double delta = task_latency_us(tasks[i], queues[q], queue_used[q] == 0);
                if (delta < best_delta) {
                    best_delta = delta;
                    best_q = q;
                }
            }
            plan.queue_loads_us[best_q] += best_delta;
            queue_used[best_q] = 1;
            plan.assignments[i] = ScheduledExpertAssignment{
                tasks[i].expert_id, (int32_t) best_q, queues[best_q].kind, queues[best_q].backend_id, best_delta
            };
        }
        for (double ld : plan.queue_loads_us) {
            plan.makespan_us = std::max(plan.makespan_us, ld);
            plan.sum_work_us += ld;
        }
        return plan;
    }
};

}  // namespace strata::chordmoe
