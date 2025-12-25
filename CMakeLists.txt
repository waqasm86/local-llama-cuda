cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(local-llama-cuda
  VERSION 0.1.0
  DESCRIPTION "High-performance on-device LLM inference with CUDA acceleration"
  LANGUAGES CXX CUDA
)

# ============================================================================
# Build Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# Options
# ============================================================================

option(ENABLE_MPI "Enable MPI distribution support" ON)
option(ENABLE_CUDA "Enable CUDA acceleration" ON)
option(ENABLE_BENCHMARKS "Build benchmark suite" ON)
option(ENABLE_TESTS "Build test suite" ON)
option(ENABLE_EXAMPLES "Build examples" ON)

# CUDA architecture (default: Maxwell 5.0)
set(CMAKE_CUDA_ARCHITECTURES "50" CACHE STRING "CUDA architecture")

# ============================================================================
# Dependencies
# ============================================================================

# CUDA Toolkit
if(ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# MPI
if(ENABLE_MPI)
  find_package(MPI REQUIRED)
  message(STATUS "MPI Found: ${MPI_CXX_COMPILER}")
  message(STATUS "MPI Include: ${MPI_CXX_INCLUDE_DIRS}")
endif()

# Threads
find_package(Threads REQUIRED)

# ============================================================================
# Compiler Flags
# ============================================================================

# C++ Compiler Flags
set(CXX_WARNING_FLAGS
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wconversion
  -Wsign-conversion
  -Wnull-dereference
  -Wdouble-promotion
  -Wformat=2
)

# CUDA Compiler Flags
set(CUDA_COMPILE_FLAGS
  -g
  -lineinfo
  --expt-relaxed-constexpr
  --use_fast_math
)

# Release optimizations
if(CMAKE_BUILD_TYPE MATCHES Release)
  list(APPEND CXX_WARNING_FLAGS -O3 -march=native)
  list(APPEND CUDA_COMPILE_FLAGS -O3)
endif()

# Debug flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
  list(APPEND CXX_WARNING_FLAGS -g -O0)
  list(APPEND CUDA_COMPILE_FLAGS -G -O0)
endif()

# ============================================================================
# Interface Libraries (Headers & Flags)
# ============================================================================

# Public headers
add_library(llcuda_headers INTERFACE)
target_include_directories(llcuda_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Compiler warnings
add_library(llcuda_warnings INTERFACE)
target_compile_options(llcuda_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:${CXX_WARNING_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_FLAGS}>
)

# ============================================================================
# Core Libraries
# ============================================================================

# Core inference engine
add_library(llcuda_core STATIC
  src/core/inference_engine.cpp
  src/core/model_manager.cpp
  src/core/request_processor.cpp
  src/core/metrics_collector.cpp
)
target_link_libraries(llcuda_core PUBLIC
  llcuda_headers
  llcuda_warnings
  Threads::Threads
)

# CUDA backend
if(ENABLE_CUDA)
  add_library(llcuda_cuda STATIC
    src/cuda/cuda_backend.cu
    src/cuda/custom_kernels.cu
    src/cuda/memory_manager.cu
    src/cuda/stream_manager.cu
  )
  target_link_libraries(llcuda_cuda PUBLIC
    llcuda_headers
    llcuda_warnings
    CUDA::cudart
  )
  set_target_properties(llcuda_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

# MPI distribution layer
if(ENABLE_MPI)
  add_library(llcuda_mpi_lib STATIC
    src/mpi/mpi_coordinator.cpp
    src/mpi/work_scheduler.cpp
    src/mpi/collective_ops.cpp
    src/mpi/device_mapper.cpp
  )
  target_link_libraries(llcuda_mpi_lib PUBLIC
    llcuda_headers
    llcuda_warnings
    MPI::MPI_CXX
  )
endif()

# Storage pipeline
add_library(llcuda_storage STATIC
  src/storage/content_addressed_store.cpp
  src/storage/manifest_manager.cpp
  src/storage/cache_manager.cpp
  src/storage/storage_client.cpp
  src/storage/sha256.cpp
)
target_link_libraries(llcuda_storage PUBLIC
  llcuda_headers
  llcuda_warnings
)

# TCP gateway
add_library(llcuda_tcp STATIC
  src/tcp/tcp_server.cpp
  src/tcp/protocol_handler.cpp
  src/tcp/connection_pool.cpp
  src/tcp/streaming_handler.cpp
)
target_link_libraries(llcuda_tcp PUBLIC
  llcuda_headers
  llcuda_warnings
)

# ============================================================================
# Utility Libraries
# ============================================================================

# HTTP client (for llama.cpp integration)
add_library(llcuda_http STATIC
  src/core/http_client.cpp
)
target_link_libraries(llcuda_http PUBLIC
  llcuda_headers
  llcuda_warnings
)

# ============================================================================
# Main Applications
# ============================================================================

# Main CLI tool
add_executable(llcuda apps/llcuda.cpp)
target_link_libraries(llcuda PRIVATE
  llcuda_core
  llcuda_storage
  llcuda_http
  llcuda_warnings
)
if(ENABLE_CUDA)
  target_link_libraries(llcuda PRIVATE llcuda_cuda)
endif()

# TCP server
add_executable(llcuda_server apps/llcuda_server.cpp)
target_link_libraries(llcuda_server PRIVATE
  llcuda_core
  llcuda_tcp
  llcuda_http
  llcuda_warnings
)
if(ENABLE_CUDA)
  target_link_libraries(llcuda_server PRIVATE llcuda_cuda)
endif()

# MPI distributed version
if(ENABLE_MPI)
  add_executable(llcuda_mpi apps/llcuda_mpi.cu)
  target_link_libraries(llcuda_mpi PRIVATE
    llcuda_core
    llcuda_mpi_lib
    llcuda_storage
    llcuda_http
    llcuda_warnings
  )
  if(ENABLE_CUDA)
    target_link_libraries(llcuda_mpi PRIVATE llcuda_cuda CUDA::cudart)
  endif()
  set_target_properties(llcuda_mpi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

# TCP client example
add_executable(llcuda_client apps/llcuda_client.cpp)
target_link_libraries(llcuda_client PRIVATE
  llcuda_tcp
  llcuda_warnings
)

# ============================================================================
# Benchmarks
# ============================================================================

if(ENABLE_BENCHMARKS)
  add_executable(bench_latency benchmarks/latency_bench.cu)
  target_link_libraries(bench_latency PRIVATE
    llcuda_core
    llcuda_storage
    llcuda_http
    llcuda_warnings
    CUDA::cudart
  )
  set_target_properties(bench_latency PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  add_executable(bench_throughput benchmarks/throughput_bench.cu)
  target_link_libraries(bench_throughput PRIVATE
    llcuda_core
    llcuda_storage
    llcuda_http
    llcuda_warnings
    CUDA::cudart
  )
  set_target_properties(bench_throughput PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  if(ENABLE_MPI)
    add_executable(bench_scaling benchmarks/scaling_bench.cpp)
    target_link_libraries(bench_scaling PRIVATE
      llcuda_core
      llcuda_mpi_lib
      llcuda_http
      llcuda_warnings
    )
  endif()
endif()

# ============================================================================
# Tests
# ============================================================================

if(ENABLE_TESTS)
  enable_testing()

  add_executable(test_core tests/test_core.cpp)
  target_link_libraries(test_core PRIVATE
    llcuda_core
    llcuda_warnings
  )
  add_test(NAME CoreTests COMMAND test_core)

  if(ENABLE_CUDA)
    add_executable(test_cuda tests/test_cuda.cu)
    target_link_libraries(test_cuda PRIVATE
      llcuda_cuda
      llcuda_warnings
    )
    add_test(NAME CUDATests COMMAND test_cuda)
  endif()

  add_executable(test_storage tests/test_storage.cpp)
  target_link_libraries(test_storage PRIVATE
    llcuda_storage
    llcuda_warnings
  )
  add_test(NAME StorageTests COMMAND test_storage)
endif()

# ============================================================================
# Examples
# ============================================================================

if(ENABLE_EXAMPLES)
  add_executable(example_basic examples/basic_inference.cpp)
  target_link_libraries(example_basic PRIVATE
    llcuda_core
    llcuda_http
    llcuda_warnings
  )

  add_executable(example_batch examples/batch_processing.cpp)
  target_link_libraries(example_batch PRIVATE
    llcuda_core
    llcuda_http
    llcuda_warnings
  )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS llcuda llcuda_server llcuda_client
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/llcuda
  DESTINATION include
)

# ============================================================================
# Build Summary
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "  local-llama-cuda Configuration")
message(STATUS "========================================")
message(STATUS "Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Support:     ${ENABLE_CUDA}")
message(STATUS "MPI Support:      ${ENABLE_MPI}")
message(STATUS "Benchmarks:       ${ENABLE_BENCHMARKS}")
message(STATUS "Tests:            ${ENABLE_TESTS}")
message(STATUS "Examples:         ${ENABLE_EXAMPLES}")
if(ENABLE_CUDA)
  message(STATUS "CUDA Version:     ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA Arch:        ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "========================================")
message(STATUS "")
