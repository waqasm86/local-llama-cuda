#!/bin/bash
# Setup script for running CUDA benchmarks
# This script prepares the build directory for benchmark execution

set -e

echo "========================================"
echo " local-llama-cuda Benchmark Setup"
echo "========================================"
echo ""

# Check if we're in the build directory
if [ ! -f "bench_latency" ]; then
    echo "Creating build/model.gguf (dummy file for HTTP backend)..."
    if [ -d "build" ]; then
        cd build
    else
        echo "ERROR: build/ directory not found"
        echo "Please run 'mkdir build && cd build && cmake .. && ninja' first"
        exit 1
    fi
fi

# Create dummy model.gguf if it doesn't exist
if [ ! -f "model.gguf" ]; then
    echo "Creating model.gguf (dummy file for HTTP backend)..."
    touch model.gguf
    echo "✅ Created model.gguf"
else
    echo "✅ model.gguf already exists"
fi

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "You can now run benchmarks:"
echo "  ./bench_latency --server http://127.0.0.1:8090 --iters 10 --cuda-work"
echo "  ./bench_throughput --server http://127.0.0.1:8090 --iters 10 --cuda-work"
echo "  mpirun -np 2 ./llcuda_mpi --server http://127.0.0.1:8090 --iters 10 --cuda-post"
echo ""
echo "Note: Make sure llama-server is running on port 8090 first!"
echo ""
