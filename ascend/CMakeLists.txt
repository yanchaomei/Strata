# ascend/CMakeLists.txt - Strata on Huawei Ascend NPU (CANN 9.1 / Ascend C dav-m200 + Cube aclnnMm).
cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  if(EXISTS "/usr/local/Ascend/ascend-toolkit/latest")
    set(ENV{ASCEND_HOME_PATH} "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()

set(ASCEND_COMPUTE_UNIT "ascend310p" CACHE STRING "Target Ascend SoC" FORCE)
list(APPEND CMAKE_PREFIX_PATH "$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")

project(strata_ascend LANGUAGES C CXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/strata_geometry.cmake)

find_package(asc REQUIRED)
enable_language(ASC)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
# Note: CANN's config.cmake clears CMAKE_CXX_FLAGS_RELEASE; restore optimization flags here.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_ASC_FLAGS "${CMAKE_ASC_FLAGS} --cce-aicore-arch=dav-m200 -O2")

get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(GGML_COMMON "${STRATA_ROOT}/third_party/strata-ggml/strata-quant-common.h")

# ---- 1. Custom Ascend C AI Core Kernels (dav-m200) ---------------------------------------------------------
ascendc_library(strata_ascend_kernels_asc STATIC src/kernels/ascend_ops.asc)
ascendc_include_directories(strata_ascend_kernels_asc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels)

# ---- 2. Reference Quant & GGUF Reader ----------------------------------------------------------------------
add_library(strata_ascend_ref STATIC src/ref_quant.cpp src/gguf_lite.cpp)
target_include_directories(strata_ascend_ref PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${STRATA_ROOT}/third_party/ggml
  ${STRATA_ROOT}/include)
target_compile_definitions(strata_ascend_ref PRIVATE STRATA_GGML_COMMON_DEFAULT="${GGML_COMMON}")

# ---- 3. Standalone Operator Parity Test --------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/ascend_parity_test.cpp")
  add_executable(ascend_parity_test tests/ascend_parity_test.cpp)
  target_include_directories(ascend_parity_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn)
  target_link_directories(ascend_parity_test PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)
  target_link_libraries(ascend_parity_test PRIVATE
    strata_ascend_kernels_asc
    strata_ascend_ref
    opapi
    nnopbase
    ascendcl
    pthread)
endif()

# ---- 4. Full Strata Engine on Ascend -----------------------------------------------------------------------
option(STRATA_ASCEND_ENGINE "Build the full strata engine binary for Ascend" ON)
if(STRATA_ASCEND_ENGINE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/window_ascend.cpp")
  add_library(strata_gpu_ascend STATIC src/gpu/runtime.cpp)
  target_include_directories(strata_gpu_ascend BEFORE PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ascend_compat
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${STRATA_ROOT}/include
    $ENV{ASCEND_HOME_PATH}/include)
  target_compile_definitions(strata_gpu_ascend PUBLIC STRATA_GPU_METAL=1)
  target_link_directories(strata_gpu_ascend PUBLIC $ENV{ASCEND_HOME_PATH}/lib64)
  target_link_libraries(strata_gpu_ascend PUBLIC ascendcl pthread)

  include(${STRATA_ROOT}/cmake/strata_ggml.cmake)
  include(${STRATA_ROOT}/cmake/backends/metal.cmake)
  strata_ggml_resolve("${STRATA_ROOT}")
  strata_metal_ggml()

  option(STRATA_OWN_CPU_QUANT "Use Strata ARM64 NEON quant layer" ON)
  if(STRATA_OWN_CPU_QUANT)
    set(STRATA_QUANT_ROOT ${STRATA_ROOT})
    set(STRATA_CPU_X86 OFF)
    include(${STRATA_ROOT}/cmake/backends/quant.cmake)
    strata_quant_layer()
  endif()

  set(S ${STRATA_ROOT}/src)
  set(STRATA_ASCEND_HOST_SOURCES
    ${S}/core/device.cu ${S}/core/pinned.cu ${S}/platform/memory.cpp ${S}/core/graph.cpp ${S}/core/weights.cpp ${S}/core/layout.cpp
    ${S}/core/layer.cpp ${S}/core/session.cpp ${S}/core/expert_source.cpp ${S}/core/remote_experts.cpp ${S}/core/expert_cache.cpp
    ${S}/core/native_head.cpp ${S}/core/native_dense.cpp ${S}/core/verify.cpp ${S}/core/mtp.cpp
    ${S}/core/conversation_snapshot.cpp ${S}/core/conversation_state.cpp ${S}/core/conversation_memory.cpp
    ${S}/core/peer_experts.cpp ${S}/core/arch_defaults.cpp ${S}/core/vmm.cpp ${S}/core/conversation_file.cpp
    ${S}/kernels/ngram.cpp ${S}/ngram/ple_reader.cpp ${S}/platform/direct_file.cpp ${S}/platform/io_priority.cpp
    ${S}/prefill/prefill.cpp
    ${S}/spec/suffix_drafter.cpp ${S}/spec/controller.cpp ${S}/spec/draft_policy.cpp ${S}/spec/draft_source.cpp
    ${S}/kernels/cpu/pool.cpp ${S}/kernels/cpu/expert_layout.cpp ${S}/kernels/cpu/native_expert.cpp
    ${S}/kernels/cpu/hadamard.cpp ${S}/kernels/cpu/tq_cpu.cpp)
  set_source_files_properties(${S}/core/device.cu ${S}/core/pinned.cu PROPERTIES LANGUAGE CXX)

  add_library(strata_ascend_host STATIC ${STRATA_ASCEND_HOST_SOURCES})
  target_include_directories(strata_ascend_host PUBLIC
    ${STRATA_ROOT}/include ${S}/core ${STRATA_ROOT}/third_party/ggml
    ${_strata_ggml_root}/ggml/include ${_strata_ggml_root}/ggml/src ${_strata_ggml_root}/ggml/src/ggml-cpu)
  target_compile_definitions(strata_ascend_host PUBLIC STRATA_NATIVE_EXPERTS=1 STRATA_HAVE_DFLASH2=1 STRATA_VERSION="0.1.60-ascend")
  target_compile_options(strata_ascend_host PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations)
  target_link_libraries(strata_ascend_host PUBLIC strata_gpu_ascend ggml-cpu ggml-base)
  if(STRATA_OWN_CPU_QUANT)
    target_link_libraries(strata_ascend_host PUBLIC strata_quant)
    target_compile_definitions(strata_ascend_host PUBLIC STRATA_OWN_CPU_QUANT=1)
  endif()

  file(GLOB STRATA_ASCEND_ENGINE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cpp)
  add_library(strata_ascend_engine STATIC ${STRATA_ASCEND_ENGINE_SOURCES})
  target_include_directories(strata_ascend_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${STRATA_ROOT}/third_party/ggml
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn)
  target_compile_definitions(strata_ascend_engine PRIVATE
    STRATA_GGML_COMMON_DEFAULT="${GGML_COMMON}"
    STRATA_ASCEND_DATA_DEFAULT="${STRATA_ROOT}/data")
  target_link_directories(strata_ascend_engine PUBLIC $ENV{ASCEND_HOME_PATH}/lib64)
  target_link_libraries(strata_ascend_engine PUBLIC
    strata_ascend_host
    strata_ascend_ref
    strata_ascend_kernels_asc
    opapi
    nnopbase
    ascendcl)

  add_executable(strata ${S}/program/generate.cpp)
  target_link_libraries(strata PRIVATE strata_ascend_engine strata_ascend_host strata_ascend_engine)
endif()
