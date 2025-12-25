# Quick Start Guide - CUDA Benchmarks

## Problem Solved

**Issue:** Benchmarks were failing with "No successful inferences"

**Root Cause:** The `InferenceEngine` checks if a model file exists before making HTTP requests to llama-server. Since we're using llama-server as a remote backend (not loading the model locally), we need a dummy `model.gguf` file.

**Solution:** Create a dummy `model.gguf` file in the build directory.

---

## Step-by-Step Instructions

### Terminal 1: Start llama-server

```bash
cd /media/waqasm86/External1/Project-CPP/Project-GGML/llama-cpp-github-repo/llama.cpp/build/bin

./llama-server \
  -m gemma-3-1b-it-Q4_K_M.gguf \
  --port 8090 \
  --parallel 1 \
  -fit off \
  -ngl 8 \
  -b 1024 \
  -ub 256 \
  --flash-attn off \
  --cache-ram 0
```

### Terminal 2: Setup and Run Benchmarks

```bash
# Navigate to project
cd /media/waqasm86/External1/Project-Nvidia/local-llama-cuda

# Option 1: Use setup script (recommended)
bash setup_benchmarks.sh
cd build

# Option 2: Manual setup
cd build
touch model.gguf  # Create dummy file for HTTP backend

# Now run benchmarks!
```

---

## Running CUDA Benchmarks

### 1. Verify everything works
```bash
bash ../verify_cuda.sh
```

### 2. Quick tests (10 iterations each)

```bash
# Latency benchmark (basic)
./bench_latency --server http://127.0.0.1:8090 --iters 10

# Latency benchmark (with CUDA post-processing)
./bench_latency --server http://127.0.0.1:8090 --iters 10 --cuda-work --cuda-iters 2000

# Throughput benchmark (basic)
./bench_throughput --server http://127.0.0.1:8090 --iters 10

# Throughput benchmark (with CUDA)
./bench_throughput --server http://127.0.0.1:8090 --iters 10 --cuda-work

# MPI distributed (basic)
mpirun -np 2 ./llcuda_mpi --server http://127.0.0.1:8090 --iters 10

# MPI distributed (with CUDA)
mpirun -np 2 ./llcuda_mpi --server http://127.0.0.1:8090 --iters 10 --cuda-post --cuda-work 2000
```

### 3. Full benchmark suite (production testing)

```bash
# Comprehensive latency test (100 iterations with CUDA)
./bench_latency --server http://127.0.0.1:8090 \
  --iters 100 \
  --max-tokens 64 \
  --cuda-work \
  --cuda-iters 2000

# Comprehensive throughput test (50 iterations with CUDA)
./bench_throughput --server http://127.0.0.1:8090 \
  --iters 50 \
  --max-tokens 128 \
  --cuda-work

# Comprehensive MPI test (100 requests across 4 workers with CUDA)
mpirun --oversubscribe -np 5 ./llcuda_mpi \
  --server http://127.0.0.1:8090 \
  --iters 100 \
  --inflight 8 \
  --n_predict 64 \
  --cuda-post \
  --cuda-work 2000
```

---

## Expected Output

### Latency Benchmark
```
=================================================
  Latency Benchmark (CUDA-Accelerated)
=================================================
Server:      http://127.0.0.1:8090
Iterations:  10
Max Tokens:  64
CUDA Work:   Enabled
CUDA Iters:  2000
=================================================

CUDA Device: NVIDIA GeForce 940M

Running 10 iterations...
  Progress: 10/10

=================================================
  Latency Results (milliseconds)
=================================================
  Min:      5711.47 ms
  p50:      6008.10 ms
  p95:      6254.66 ms
  p99:      6254.66 ms
  Max:      6254.66 ms
=================================================
```

### Throughput Benchmark
```
=================================================
  Throughput Benchmark (CUDA-Accelerated)
=================================================
Server:      http://127.0.0.1:8090
Iterations:  10
Max Tokens:  64
CUDA Work:   Enabled
=================================================

CUDA Device: NVIDIA GeForce 940M

Running 10 iterations...

=================================================
  Results
=================================================
  Total Tokens:        640
  Total Time:          60.08 s
  Throughput:          10.65 tokens/sec
=================================================
```

### MPI Distributed
```
=================================================
  CUDA MPI Distributed Scheduler
=================================================
Server:       http://127.0.0.1:8090
Ranks:        2 (1 master, 1 worker)
Iterations:   10
Inflight:     4
Predict:      64
CUDA Post:    Enabled
CUDA Work:    2000
=================================================

Rank 1 completed request 0 in 6008.45 ms (CUDA: 12.34 ms)
Rank 1 completed request 1 in 6012.23 ms (CUDA: 11.98 ms)
...

=================================================
  Results Summary
=================================================
  Total Requests:      10
  Successful:          10
  Failed:              0
  Total Time:          60.12 s
  Throughput:          0.17 requests/sec

  Latency (ms):
    p50:               6010.34 ms
    p95:               6254.12 ms
    p99:               6254.66 ms

  CUDA Post (ms):
    p50:               12.15 ms
    p95:               13.87 ms
    p99:               14.23 ms
=================================================
```

---

## Troubleshooting

### Issue: "No successful inferences"
**Solution:** Make sure `model.gguf` exists in the build directory
```bash
cd build
touch model.gguf
```

### Issue: "Connection refused"
**Solution:** Make sure llama-server is running on port 8090
```bash
curl http://127.0.0.1:8090/health
# Should return: {"status":"ok"}
```

### Issue: "CUDA error: no CUDA-capable device"
**Solution:** Check GPU availability
```bash
nvidia-smi
```

### Issue: MPI fails with "not enough slots"
**Solution:** Use `--oversubscribe` flag
```bash
mpirun --oversubscribe -np 4 ./llcuda_mpi ...
```

---

## Why Create a Dummy model.gguf?

The `InferenceEngine` was originally designed to load models locally. When using llama-server as an HTTP backend:

1. The actual model is loaded in llama-server (not in our benchmark process)
2. Our benchmarks communicate with llama-server via HTTP `/completion` endpoint
3. However, `InferenceEngine::load_model()` still checks if a file exists
4. Creating an empty `model.gguf` satisfies this check without loading anything

**This is a harmless workaround** - the dummy file is never actually read or used. All inference happens in the llama-server process.

---

## Performance Notes

- **GeForce 940M** has limited VRAM (1GB) and compute capability (5.2)
- Expected throughput: **10-13 tokens/sec** for gemma-3-1b Q4_K_M
- Latency will be **5-7 seconds per 64 tokens** on this GPU
- This is normal for a mobile GPU from 2015

For better performance:
- Use a more powerful GPU (RTX 3060+)
- Reduce `--max-tokens` for faster iteration
- Increase `--ctx-size` in llama-server for longer contexts

---

## Monitor GPU Usage

```bash
# Terminal 3: Real-time GPU monitoring
watch -n 1 nvidia-smi
```

---

## Summary

✅ All CUDA benchmarks now working
✅ HTTP client successfully connecting to llama-server
✅ CUDA kernels executing on GeForce 940M
✅ All unit tests passing

For more details, see:
- [CUDA_IMPLEMENTATION_SUMMARY.md](CUDA_IMPLEMENTATION_SUMMARY.md)
- [RUNNING_CUDA_BENCHMARKS.md](RUNNING_CUDA_BENCHMARKS.md)
