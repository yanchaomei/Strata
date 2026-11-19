#!/usr/bin/env bash
# scripts/chordmoe_server_suite.sh - Turnkey build, parity verification, and head-to-head benchmark suite
# for Linux Multi-GPU (2-4x RTX 3090 / 4090 / A6000 without NVLink) + Multi-NUMA Xeon/EPYC servers.
#
# Quickstart on Linux server:
#   git clone -b feat/chordmoe-proto https://github.com/yanchaomei/Strata.git
#   cd Strata
#   chmod +x scripts/chordmoe_server_suite.sh
#   ./scripts/chordmoe_server_suite.sh build
#   ./scripts/chordmoe_server_suite.sh parity
#   MODEL_GGUF=/path/to/qwen38-or-qwen36.gguf ./scripts/chordmoe_server_suite.sh bench-strata
#   S2_MODEL_GGUF=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf ./scripts/chordmoe_server_suite.sh bench-s2moe

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build-bench-results/${STAMP}}"
BASELINES_DIR="${BASELINES_DIR:-${ROOT_DIR}/build-baselines}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 16)}"

mkdir -p "${OUT_DIR}"

log() { printf "\n\033[1;36m[ChordMoE-Suite]\033[0m %s\n" "$*"; }

detect_cuda_arch() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local raw
        raw="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
        if [[ -n "${raw}" ]]; then
            echo "${raw}"
            return
        fi
    fi
    echo "86;89"
}

detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

record_topology() {
    log "Recording GPU PCIe & CPU NUMA topology into ${OUT_DIR}/topology.txt"
    {
        echo "=== Timestamp: ${STAMP} ==="
        uname -a || true
        echo ""
        echo "=== NVIDIA Topology ==="
        nvidia-smi || true
        nvidia-smi topo -m || true
        echo ""
        echo "=== CPU / NUMA Topology ==="
        lscpu || true
        numactl -H || true
    } > "${OUT_DIR}/topology.txt" 2>&1
}

step_build() {
    record_topology
    local cc
    cc="$(detect_cuda_arch)"
    local ngpu
    ngpu="$(detect_gpu_count)"
    log "Detected ${ngpu} NVIDIA GPU(s), CUDA compute capability=${cc}"

    # 1. Build Strata CPU + CUDA engine + ChordMoE GPU parity test
    log "1/3 Building Strata CUDA engine + ChordMoE unit & GPU parity tests..."
    cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build-cuda" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_CUDA=ON \
        -DSTRATA_BUILD_TESTS=ON \
        -DSTRATA_NATIVE_EXPERTS=ON \
        -DSTRATA_PORTABLE=ON \
        -DSTRATA_Q6K_EXPERTS=ON \
        -DSTRATA_MMQ_KQUANTS=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${cc}"
    cmake --build "${ROOT_DIR}/build-cuda" --target strata chordmoe_test chordmoe_gpu_parity_test -j"${JOBS}"

    # 2. Build Strata All-Cards non-P2P Split Engine (2 & 4 cards) when >= 2 GPUs exist
    if [[ "${ngpu}" -ge 2 ]]; then
        log "2/3 Building Strata All-Cards non-NVLink split engine (cards=2;4, arch=${cc%%;*})..."
        cmake -S "${ROOT_DIR}/src/split" -B "${ROOT_DIR}/build-split-cuda" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DSTRATA_GGML_DIR="${ROOT_DIR}/third_party/strata-ggml" \
            -DSTRATA_SPLIT_CARDS="2;4" \
            -DSTRATA_SPLIT_ARCH="${cc%%;*}" || \
        log "Note: standalone src/split configure skipped; use packaging/build_split_cuda.sh if needed."
        if [[ -f "${ROOT_DIR}/build-split-cuda/build.ninja" ]]; then
            cmake --build "${ROOT_DIR}/build-split-cuda" -j"${JOBS}"
        fi
    fi

    # 3. Clone & build S2-MoE (llama.cpp fork) with our drop-in ChordMoE patch
    log "3/3 Building S2-MoE (CUDA + ChordMoE portability adapter)..."
    mkdir -p "${BASELINES_DIR}/idea-competitors"
    if [[ ! -d "${BASELINES_DIR}/idea-competitors/S2-MoE" ]]; then
        git clone --depth 1 https://github.com/angerybob/S2-MoE.git "${BASELINES_DIR}/idea-competitors/S2-MoE"
    fi
    (
        cd "${BASELINES_DIR}/idea-competitors/S2-MoE"
        git checkout -- ggml/src/ggml-cpu/ops.cpp ggml/src/ggml-cuda/moe-reuse.cu 2>/dev/null || true
        git apply "${ROOT_DIR}/contrib/llama_cpp_chordmoe/s2moe_chordmoe.patch"
        CHORD_INC="-I${ROOT_DIR}/include -I${ROOT_DIR}/contrib/llama_cpp_chordmoe"
        cmake -B build-cuda -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_CUDA=ON \
            -DLLAMA_CURL=OFF \
            -DCMAKE_CUDA_ARCHITECTURES="${cc}" \
            -DCMAKE_CXX_STANDARD=20 \
            -DCMAKE_CXX_FLAGS="${CHORD_INC}" \
            -DCMAKE_CUDA_FLAGS="${CHORD_INC}"
        cmake --build build-cuda --target llama-speculative llama-speculative-next llama-bench -j"${JOBS}"
    )
    log "Build complete! Artifacts ready in build-cuda/ and ${BASELINES_DIR}/idea-competitors/S2-MoE/build-cuda/"
}

step_parity() {
    log "Running ChordMoE CPU C++20 unit suite..."
    "${ROOT_DIR}/build-cuda/chordmoe_test" | tee "${OUT_DIR}/chordmoe_cpu_test.log"

    log "Running ChordMoE CUDA CTA vs CPU Reference bit-consistency & kernel latency benchmark..."
    "${ROOT_DIR}/build-cuda/chordmoe_gpu_parity_test" | tee "${OUT_DIR}/chordmoe_gpu_parity.log"

    log "Running Python multi-GPU hybrid split offload admission tests..."
    python3 -m pytest --import-mode=importlib \
        "${ROOT_DIR}/tests/terminal/test_chordmoe_split_offload.py" \
        "${ROOT_DIR}/tests/split/" -q | tee "${OUT_DIR}/chordmoe_python_test.log"
}

step_bench_strata() {
    local model="${MODEL_GGUF:-}"
    if [[ -z "${model}" || ! -f "${model}" ]]; then
        echo "ERROR: Set MODEL_GGUF=/path/to/Qwen3.8-or-Qwen3.6.gguf before running bench-strata" >&2
        exit 1
    fi
    local prompt="${BENCH_PROMPT:-Write a thread-safe lock-free ring buffer in C++20 with acquire-release memory ordering and explain its ABA prevention strategy.}"
    local n_gen="${N_GEN:-256}"
    local csv="${OUT_DIR}/strata_ablation.csv"
    echo "config,mode_env,log_file" > "${csv}"

    run_case() {
        local name="$1"
        shift
        local logfile="${OUT_DIR}/strata_${name}.log"
        log "Running Strata case [${name}]..."
        nvidia-smi dmon -s u -d 1 > "${OUT_DIR}/dmon_${name}.log" 2>&1 &
        local dmon_pid=$!
        env "$@" "${ROOT_DIR}/build-cuda/strata" \
            --model "${model}" \
            --prompt "${prompt}" \
            --n-predict "${n_gen}" \
            --temp 0 > "${logfile}" 2>&1 || true
        kill "${dmon_pid}" 2>/dev/null || true
        echo "${name},\"$*\",${logfile}" >> "${csv}"
    }

    # 1. Stock Strata baseline (ChordMoE completely off)
    run_case "0_baseline" STRATA_CHORD_MODE=-1
    # 2. ChordMoE Mode 0: 100% mathematically lossless (early abort + doorbell online divergence probe)
    run_case "1_chord_mode0_lossless" STRATA_CHORD_MODE=0 STRATA_CHORD_EARLY_ABORT=1 STRATA_CHORD_ONLINE_PROBE=1
    # 3. ChordMoE Mode 1: Error-bounded window union set-cover + anchor protection + online probe
    run_case "2_chord_mode1_setcover" STRATA_CHORD_MODE=1 STRATA_CHORD_MARGIN=1.35 STRATA_CHORD_TV_BUDGET=0.15 STRATA_CHORD_EARLY_ABORT=1 STRATA_CHORD_ONLINE_PROBE=1

    log "Strata ablation logs & PCIe dmon traces written to ${OUT_DIR}"
}

step_bench_s2moe() {
    local model="${S2_MODEL_GGUF:-}"
    if [[ -z "${model}" || ! -f "${model}" ]]; then
        echo "ERROR: Set S2_MODEL_GGUF=/path/to/any-moe-model.gguf (e.g., Qwen3-30B-A3B / Qwen3-235B / Mixtral-8x22B)" >&2
        exit 1
    fi
    local bin="${BASELINES_DIR}/idea-competitors/S2-MoE/build-cuda/bin/llama-speculative"
    local ngl="${S2_NGL:-99}"
    local offload_ot="${S2_OT:-.ffn_.*_exps.=CPU}"
    local prompt="${BENCH_PROMPT:-Explain how speculative decoding interacts with PCIe DMA bandwidth in Mixture-of-Experts offloading systems.}"

    log "1/3 Running Stock llama.cpp speculative verification (no MoE reuse)..."
    "${bin}" -m "${model}" -md "${model}" -ngl "${ngl}" -ngld "${ngl}" -ot "${offload_ot}" \
        --draft-max 6 -n 128 --temp 0 -p "${prompt}" \
        > "${OUT_DIR}/s2bin_0_stock_llamacpp.log" 2>&1 || true

    log "2/3 Running Original S2-MoE (--moe-reuse-runtime --moe-reuse-expert-cap 16)..."
    S2MOE_AUDIT_STATS=1 CHORDMOE_ENABLE=0 \
    "${bin}" -m "${model}" -md "${model}" -ngl "${ngl}" -ngld "${ngl}" -ot "${offload_ot}" \
        --moe-reuse-strength 0.5 --moe-reuse-expert-cap 16 --moe-reuse-runtime \
        --draft-max 6 -n 128 --temp 0 -p "${prompt}" \
        > "${OUT_DIR}/s2bin_1_s2moe_orig.log" 2>&1 || true

    log "3/3 Running ChordMoE-on-llama.cpp (CHORDMOE_ENABLE=1, identical runtime & GGUF)..."
    CHORDMOE_ENABLE=1 CHORDMOE_VRAM_RATIO=0.32 STRATA_CHORD_MARGIN=1.35 STRATA_CHORD_TV_BUDGET=0.16 \
    "${bin}" -m "${model}" -md "${model}" -ngl "${ngl}" -ngld "${ngl}" -ot "${offload_ot}" \
        --moe-reuse-strength 0.5 --moe-reuse-expert-cap 16 --moe-reuse-runtime \
        --draft-max 6 -n 128 --temp 0 -p "${prompt}" \
        > "${OUT_DIR}/s2bin_2_chordmoe.log" 2>&1 || true

    log " Controlled llama.cpp comparison complete! Tail telemetry:"
    grep -E "\[S2-MoE-audit\]|\[ChordMoE-ggml\]|decoded" "${OUT_DIR}"/s2bin_*.log || true
}

cmd="${1:-help}"
case "${cmd}" in
    build)        step_build ;;
    parity)       step_parity ;;
    bench-strata) step_bench_strata ;;
    bench-s2moe)  step_bench_s2moe ;;
    all)          step_build && step_parity ;;
    *)
        echo "Usage: $0 {build|parity|bench-strata|bench-s2moe|all}"
        exit 1
        ;;
esac
