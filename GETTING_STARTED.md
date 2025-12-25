# Getting Started with Local LLaMA CUDA

This guide will help you build and run the Local LLaMA CUDA project.

## Prerequisites

### Required

1. **CUDA Toolkit** (12.0+)
   ```bash
   nvcc --version
   ```

2. **OpenMPI** (5.0+) with CUDA-aware support
   ```bash
   mpirun --version
   ```

3. **CMake** (3.24+)
   ```bash
   cmake --version
   ```

4. **Build Tools**
   ```bash
   # Ubuntu/Debian
   sudo apt install ninja-build g++

   # Or use make if ninja not available
   sudo apt install build-essential
   ```

5. **llama.cpp** server running separately
   - Build llama.cpp from: https://github.com/ggerganov/llama.cpp
   - Download a GGUF model
   - Start llama-server on port 8090

### Optional

- **Python 3.8+** for scripts and SDK (future)
- **Git** for version control
- **Docker** for containerized deployment (future)

## Quick Start

### 1. Check Environment

```bash
cd /media/waqasm86/External1/Project-Nvidia/local-llama-cuda
./scripts/env_check.sh
```

Expected output:
```
=== Environment Check ===
CUDA: 12.8
GPU:
  NVIDIA GeForce 940M, 975 MiB
MPI: Open MPI 5.0.6
CMake: 3.24.0
Ninja: 1.11.0
```

### 2. Build the Project

```bash
./scripts/build.sh
```

This will:
- Configure CMake with optimal settings
- Build all targets using Ninja
- Create executables in `build/` directory

Build time: ~1-2 minutes on modern hardware

### 3. Run Tests

```bash
cd build
ninja test
```

Expected output:
```
Test project /path/to/build
    Start 1: CoreTests
1/3 Test #1: CoreTests ........................   Passed    0.01 sec
    Start 2: CUDATests
2/3 Test #2: CUDATests ........................   Passed    0.05 sec
    Start 3: StorageTests
3/3 Test #3: StorageTests .....................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 3
```

### 4. Start llama-server (Separate Terminal)

```bash
cd /path/to/llama.cpp/build/bin

./llama-server \
  -m /path/to/your/model.gguf \
  --port 8090 \
  --parallel 1 \
  -ngl 4 \
  -b 1024
```

Replace `/path/to/your/model.gguf` with your actual model path.

### 5. Run Your First Inference

```bash
cd /media/waqasm86/External1/Project-Nvidia/local-llama-cuda/build

./llcuda infer \
  -m /path/to/your/model.gguf \
  -p "What is artificial intelligence?"
```

Expected output:
```
Loading model: /path/to/your/model.gguf
Running inference...

Response:
Artificial intelligence (AI) refers to the simulation of human intelligence...

Metrics:
  Tokens: 45
  Latency: 1250.3 ms
  Throughput: 36.0 tokens/sec
```

## Usage Examples

### Single Inference

```bash
./llcuda infer -m model.gguf -p "Explain quantum computing"
```

### Streaming Inference

```bash
./llcuda infer -m model.gguf -p "Write a story" --stream
```

### Batch Processing

Create `prompts.jsonl`:
```json
{"prompt": "What is AI?", "max_tokens": 50}
{"prompt": "Explain machine learning", "max_tokens": 50}
{"prompt": "What are neural networks?", "max_tokens": 50}
```

Run batch:
```bash
./llcuda batch -m model.gguf -i prompts.jsonl -o results.jsonl
```

### Benchmarking

```bash
./llcuda bench -m model.gguf --iters 100
```

Output includes:
- Mean, P50, P95, P99 latency
- Tokens per second
- Per-iteration timing

### Custom Parameters

```bash
./llcuda infer \
  -m model.gguf \
  -p "Your prompt here" \
  --temperature 0.8 \
  --max-tokens 200 \
  --gpu-layers 8
```

## Configuration

### llama-server URL

Default: `http://127.0.0.1:8090`

Override:
```bash
./llcuda infer -m model.gguf -p "prompt" --server http://localhost:9000
```

### GPU Layers

Control GPU offloading:
```bash
./llcuda infer -m model.gguf -p "prompt" --gpu-layers 8
```

Higher values = more VRAM usage, potentially faster inference

### Model Cache

Models are cached in `/tmp/llcuda_cache/` by default.

## Troubleshooting

### Issue: "Model file not found"

**Solution**: Provide full absolute path to model:
```bash
./llcuda infer -m /full/path/to/model.gguf -p "prompt"
```

### Issue: "HTTP error: 0"

**Cause**: llama-server not running or wrong URL

**Solution**:
1. Start llama-server in separate terminal
2. Verify it's running: `curl http://127.0.0.1:8090/health`
3. Check port matches: `--server http://127.0.0.1:8090`

### Issue: Build fails with CUDA errors

**Solution**:
1. Check CUDA version: `nvcc --version`
2. Update `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt to match your GPU
3. For GeForce 940M: use `50`
4. For RTX 30xx: use `86`
5. Run `./scripts/build.sh` again

### Issue: MPI not found

**Solution**:
1. Install OpenMPI: `sudo apt install openmpi-bin libopenmpi-dev`
2. Or disable MPI: edit CMakeLists.txt, set `ENABLE_MPI OFF`
3. Rebuild: `./scripts/build.sh`

### Issue: Out of memory

**Cause**: Model too large for GPU VRAM

**Solution**:
1. Use smaller quantization (Q4_K_M instead of Q8_0)
2. Reduce GPU layers: `--gpu-layers 2`
3. Use smaller model (e.g., 1B instead of 7B)

## Advanced Usage

### MPI Multi-GPU (Future)

```bash
mpirun -np 2 ./llcuda_mpi -m model.gguf
```

### TCP Server (Future)

```bash
./llcuda_server -m model.gguf --port 8080 --workers 4
```

### Python SDK (Future)

```python
from llcuda import InferenceClient

client = InferenceClient(host="localhost", port=8090)
result = client.infer("What is AI?")
print(result.text)
```

## Performance Tips

### For Fastest Inference

1. Use Q4_K_M quantization for speed
2. Maximize GPU layers (within VRAM limits)
3. Use smaller batch sizes (1-2)
4. Enable flash attention if supported

### For Best Quality

1. Use Q8_0 or F16 quantization
2. Higher temperature (0.8-1.0) for creativity
3. Larger context window
4. More tokens per request

### For Benchmarking

1. Warmup iterations (5-10)
2. Multiple runs (50-100)
3. Fixed random seed
4. Monitor GPU temperature

## Next Steps

- Read [README.md](README.md) for full documentation
- See [PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) for architecture details
- Explore [examples/](examples/) directory
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Getting Help

- GitHub Issues: [Report bugs or request features]
- Documentation: [docs/](docs/) directory
- Examples: [examples/](examples/) directory

---

**Happy inferencing!** 🚀
