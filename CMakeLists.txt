# Strata-DeepSeek - the DeepSeek-V4.1-Flash engine, on its own.
#
# One executable, `strata-ds4`: the GGUF (IQ2_XS / IQ3_XXS / MXFP4 / K-quant) DeepSeek engine with multi-GPU
# expert lending, DSpark speculative decoding, Engram and chunked prompt processing.  Everything else in the
# upstream Maya repository (the GLM and Qwen engines, their packing tools, tests and fixtures) is not here.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_GGML_DIR=/path/to/llama.cpp
#   cmake --build build -j
#
# ggml (llama.cpp's CPU backend) comes from a llama.cpp checkout through STRATA_GGML_DIR, or the pinned commit
# in third_party/ggml/VERSION.txt is fetched on first configure.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.0 LANGUAGES CXX)

add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA engine (the only backend this repository ships)" ON)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
if(NOT STRATA_ENABLE_CUDA)
  message(FATAL_ERROR "Strata-DeepSeek is a CUDA engine; configure with -DSTRATA_ENABLE_CUDA=ON")
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
endif()
foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
  # Ampere (80), Ada (89) and Blackwell (120) are the intended targets; Turing (75) builds too.  Pre-sm_75 is
  # refused: the tf32 mma in the QSA scorer has an fp32 fallback, but the rest of the kernels assume Volta+.
  if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
    continue()
  endif()
  if(_base LESS 75)
    message(FATAL_ERROR
      "Strata-DeepSeek needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 / 30 / 40 / 50 series), "
      "but CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.")
  endif()
endforeach()
message(STATUS "Strata-DeepSeek: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
set(_strata_archs "${CMAKE_CUDA_ARCHITECTURES}")
if(_strata_archs MATCHES "native")
  set(_strata_archs "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
endif()
foreach(_a IN LISTS _strata_archs)
  string(REGEX REPLACE "-.*$" "" _base "${_a}")
  if(_base MATCHES "^6[0-9]$")
    string(APPEND CMAKE_CUDA_FLAGS " -Xptxas -dlcm=ca")
    message(STATUS "Strata-DeepSeek: Pascal (sm_${_base}): global loads cached in L1 (-Xptxas -dlcm=ca)")
    break()
  endif()
endforeach()
if(WIN32)
  set(_strata_gpu_runtime_target CUDA::cudart_static)
else()
  set(_strata_gpu_runtime_target CUDA::cudart)
endif()
set(_strata_gpu_blas_target CUDA::cublas)
set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd ggml, whose warnings are not this project's to fix.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>)
endif()

# ---- header-only boundaries -------------------------------------------------
# The GGUF reader and scalar dequantizers (include/strata/artifact) and the plan headers are header-only and
# have no state to hide; they stay INTERFACE libraries so no duplicate-symbol bug can enter code validated
# against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

# ---- strata_core (GPU runtime) ----------------------------------------------
add_library(strata_core STATIC src/core/device.cu src/core/pinned.cu src/platform/memory.cpp
                               src/core/graph.cpp src/core/weights.cpp src/core/layout.cpp)
target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# The runtime is PUBLIC because graph.cpp and weights.cpp are HOST translation units that call cudart
# themselves; a .cpp does not get cudart through the CUDA language rules.
target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${_strata_gpu_runtime_target})

# ---- strata_kernels (the GPU arithmetic) ------------------------------------
add_library(strata_kernels STATIC
  src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
  src/kernels/cuda/s_gemv.cu
  src/kernels/cuda/s2_gemv_quads.cu
  src/kernels/cuda/s2_gemv_fast.cu
  src/kernels/cuda/router_top10.cu
  src/kernels/cuda/router_sigmoid.cu
  src/kernels/cuda/quantize_act.cu
  src/kernels/cuda/rope.cu
  src/kernels/cuda/sampler.cu
  src/kernels/cuda/s2_gemv_q8.cu
  src/kernels/cuda/s2_expert_grouped.cu
  src/kernels/cuda/shared_expert.cu
  src/kernels/cuda/gr.cu
  src/kernels/cuda/gdn.cu
  src/kernels/cuda/glm_hc.cu
  src/kernels/cuda/glm_kda.cu
  src/kernels/cuda/glm_ffn.cu
  src/kernels/cuda/glm_dsa.cu
  src/kernels/cuda/qsa.cu
  src/kernels/cuda/elementwise.cu
  src/kernels/cuda/bf16_gemv.cu
  src/kernels/cuda/native_mmvq.cu
  src/kernels/cuda/native_bf16.cu
  src/kernels/cuda/native_gr_norm.cu
  src/kernels/cuda/native_moe.cu
  src/kernels/cuda/native_gdn.cu
  src/kernels/cuda/native_gr_postops.cu
  src/kernels/cuda/native_router.cu
  src/kernels/cuda/native_gdn_preprocess.cu
  src/kernels/cuda/native_qsa.cu
  src/kernels/cuda/native_rope.cu
  src/kernels/cuda/native_ple_postops.cu
  src/kernels/cuda/native_flash_attn.cu
  src/kernels/cuda/native_qsa_indexer.cu
  src/kernels/cuda/native_qsa_score.cu
  src/kernels/cuda/ple.cu
  src/kernels/cuda/kv_q8.cu
  src/kernels/cuda/kv_stream.cu
  src/kernels/cuda/kv_q4.cu
  src/kernels/cuda/fused_gr.cu
  src/kernels/cuda/cvec.cu
  src/kernels/cuda/qsa_decode_attn.cu
  src/kernels/cuda/qsa_prompt_attn.cu
  src/kernels/cuda/fused_gdn.cu
  src/kernels/cuda/dequant_bf16.cu
  src/kernels/cuda/qsa_select.cu
  src/kernels/cuda/verify_kernels.cu
  src/kernels/cuda/iq_kernels.cu
  src/kernels/cuda/glm_fast.cu
  src/kernels/cuda/glm_batch.cu
  src/kernels/cuda/exl3.cu
  src/kernels/cuda/exl3_legacy.cu
  src/kernels/cuda/glm_kv_stream.cu
  src/kernels/ngram.cpp
  src/ngram/ple_reader.cpp
  src/platform/direct_file.cpp)
target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
# llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
target_include_directories(strata_kernels PRIVATE third_party/ggml)
# Match the pinned llama.cpp CUDA arithmetic contract for these adapter files only.
set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
  src/kernels/cuda/native_gr_norm.cu
  src/kernels/cuda/native_moe.cu
  src/kernels/cuda/native_gdn.cu
  src/kernels/cuda/native_gr_postops.cu
  src/kernels/cuda/native_router.cu
  src/kernels/cuda/native_gdn_preprocess.cu
  src/kernels/cuda/native_qsa.cu
  src/kernels/cuda/native_rope.cu
  src/kernels/cuda/native_ple_postops.cu
  src/kernels/cuda/native_flash_attn.cu
  src/kernels/cuda/native_qsa_indexer.cu
  src/kernels/cuda/native_qsa_score.cu
  PROPERTIES COMPILE_OPTIONS "--use_fast_math")

# ---- strata_kernels_cpu (P2.S3) ---------------------------------------------
#
# The CPU expert lane: the AVX-512/AVX2 kernels are set per SOURCE so the scalar fallback and the CPU probe
# (which run on any x86-64) never inherit an AVX-512 build.
add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/exl3_ref.cpp
            src/kernels/cpu/exl3_cpu.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include)
if(MSVC)
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties(src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- the i-quant CPU experts through ggml-cpu --------------------------------
#
# The CPU computes a native expert with ggml-cpu's own dot products, so an IQ weight means exactly what it
# means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one with -DSTRATA_GGML_DIR=...,
# otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "The DeepSeek engine's CPU lane needs ggml-cpu (STRATA_NATIVE_EXPERTS=ON)")
endif()
enable_language(C)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()
if(STRATA_GGML_DIR)
  add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
else()
  include(FetchContent)
  FetchContent_Declare(strata_llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
    GIT_SHALLOW FALSE)
  FetchContent_GetProperties(strata_llamacpp)
  if(NOT strata_llamacpp_POPULATED)
    FetchContent_Populate(strata_llamacpp)
  endif()
  add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
endif()
target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                         src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx512.cpp
                                         src/kernels/cpu/iq_avx512_one.cpp)
target_include_directories(strata_kernels_cpu PRIVATE third_party/ggml)
if(MSVC)
  set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/kq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(src/kernels/cpu/iq_avx512_one.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
  set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma;-ffp-contract=off")
  set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off")
  set_source_files_properties(src/kernels/cpu/iq_avx512_one.cpp PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vbmi;-mbmi2;-mf16c;-mfma;-ffp-contract=off")
  set_source_files_properties(src/kernels/cpu/kq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma")
endif()
target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)

# ---- strata-ds4 ------------------------------------------------------------
# DeepSeek-V4.1-Flash (a GGUF made by tools/ds4/pack_gguf.py): the engine over the kernel libraries.
add_executable(strata-ds4 src/ds4/ds4_main.cpp src/ds4/ds4_model.cu src/ds4/ds4_kernels.cu src/ds4/ds4_vision.cu
                          src/ds4/ds4_gguf.cu src/ds4/ds4_gguf_cpu.cpp src/ds4/ds4_iq.cu src/ds4/ds4_iq_cpu.cpp
                          src/ds4/ds4_lt.cu src/ds4/ds4_host_ar.cpp)
target_include_directories(strata-ds4 PRIVATE ${_strata_gpu_include_directories} third_party/ggml)
target_link_libraries(strata-ds4 PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target}
                      ${_strata_gpu_blas_target})

install(TARGETS strata-ds4 RUNTIME DESTINATION bin)
