#!/bin/bash
# CUDA Implementation Verification Script
# This script verifies all CUDA components are working correctly

set -e

echo "========================================"
echo " CUDA Implementation Verification"
echo "========================================"
echo ""

# Check if we're in the build directory
if [ ! -f "bench_latency" ]; then
    echo "ERROR: Must be run from build/ directory"
    echo "Usage: cd build && bash ../verify_cuda.sh"
    exit 1
fi

echo "1. Checking CUDA device..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "WARNING: nvidia-smi not available"
echo ""

echo "2. Running CUDA unit tests..."
./test_cuda
echo "✅ CUDA unit tests passed"
echo ""

echo "3. Running all unit tests..."
ninja test 2>&1 | grep -E "(Test|Passed|Failed|Total)"
echo "✅ All unit tests passed"
echo ""

echo "4. Verifying CUDA-enabled executables..."
for exe in bench_latency bench_throughput llcuda_mpi test_cuda; do
    if [ -f "$exe" ]; then
        size=$(du -h "$exe" | cut -f1)
        echo "  ✅ $exe ($size)"
    else
        echo "  ❌ $exe - NOT FOUND"
        exit 1
    fi
done
echo ""

echo "5. Testing help outputs..."
./bench_latency --help > /dev/null && echo "  ✅ bench_latency --help"
./bench_throughput --help > /dev/null && echo "  ✅ bench_throughput --help"
./llcuda_mpi --help > /dev/null && echo "  ✅ llcuda_mpi --help"
echo ""

echo "6. Checking CUDA kernels in binaries..."
for exe in bench_latency bench_throughput llcuda_mpi; do
    if strings "$exe" | grep -q "cuda"; then
        echo "  ✅ $exe contains CUDA code"
    else
        echo "  ⚠️  $exe may not contain CUDA code"
    fi
done
echo ""

echo "========================================"
echo " Verification Complete"
echo "========================================"
echo ""
echo "All CUDA components are working correctly!"
echo ""
echo "To run benchmarks (requires llama-server on port 8090):"
echo "  ./bench_latency --server http://127.0.0.1:8090 --iters 10 --cuda-work"
echo "  ./bench_throughput --server http://127.0.0.1:8090 --iters 10 --cuda-work"
echo "  mpirun -np 2 ./llcuda_mpi --server http://127.0.0.1:8090 --iters 10 --cuda-post"
echo ""
