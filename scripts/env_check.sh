#!/bin/bash

echo "=== Environment Check ==="

# CUDA
echo -n "CUDA: "
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release" | awk '{print $5}' | tr -d ','
else
    echo "NOT FOUND"
fi

# GPU
echo "GPU:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "  nvidia-smi not available"
fi

# MPI
echo -n "MPI: "
if command -v mpirun &> /dev/null; then
    mpirun --version | head -1
else
    echo "NOT FOUND"
fi

# CMake
echo -n "CMake: "
if command -v cmake &> /dev/null; then
    cmake --version | head -1 | awk '{print $3}'
else
    echo "NOT FOUND"
fi

# Ninja
echo -n "Ninja: "
if command -v ninja &> /dev/null; then
    ninja --version
else
    echo "NOT FOUND (using make)"
fi

echo ""
