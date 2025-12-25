#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==================================="
echo "  Building local-llama-cuda"
echo "==================================="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DENABLE_MPI=ON \
  -DENABLE_BENCHMARKS=ON \
  -DENABLE_TESTS=ON \
  -DENABLE_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES=50

# Build
echo "Building..."
ninja -j$(nproc)

echo ""
echo "Build complete!"
echo "Executables in: build/"
echo ""
echo "Run tests with: ninja test"
echo "Install with: sudo ninja install"
