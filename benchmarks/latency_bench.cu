#include "llcuda/inference_engine.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

// CUDA kernel for latency post-processing work
extern "C" {
    __global__ void latency_test_kernel(float* data, int n, int iters) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float val = data[idx];
            for (int i = 0; i < iters; i++) {
                val = val * 1.001f + 0.1f;
            }
            data[idx] = val;
        }
    }
}

void run_cuda_latency_work(int num_elements, int iters) {
    float* d_data = nullptr;
    cudaMalloc(&d_data, num_elements * sizeof(float));

    std::vector<float> h_data(num_elements, 1.0f);
    cudaMemcpy(d_data, h_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    latency_test_kernel<<<(num_elements + 255) / 256, 256>>>(d_data, num_elements, iters);
    cudaDeviceSynchronize();

    cudaFree(d_data);
}

double percentile(std::vector<double>& data, double p) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end());
    size_t idx = static_cast<size_t>(p * data.size());
    if (idx >= data.size()) idx = data.size() - 1;
    return data[idx];
}

void print_usage(const char* prog) {
    std::cout << "CNSE Latency Benchmark - Percentile Distribution\\n\\n";
    std::cout << "Usage: " << prog << " [options]\\n\\n";
    std::cout << "Options:\\n";
    std::cout << "  --server <url>     llama-server URL (default: http://127.0.0.1:8090)\\n";
    std::cout << "  --iters <n>        Number of iterations (default: 100)\\n";
    std::cout << "  --max-tokens <n>   Tokens per request (default: 64)\\n";
    std::cout << "  --cuda-work        Enable CUDA latency work\\n";
    std::cout << "  --cuda-iters <n>   CUDA work iterations (default: 1000)\\n";
    std::cout << "  -h, --help         Show this help\\n";
}

int main(int argc, char** argv) {
    std::string server_url = "http://127.0.0.1:8090";
    int iters = 100;
    int max_tokens = 64;
    bool cuda_work = false;
    int cuda_iters = 1000;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--server") == 0 && i + 1 < argc) {
            server_url = argv[++i];
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cuda-work") == 0) {
            cuda_work = true;
        } else if (std::strcmp(argv[i], "--cuda-iters") == 0 && i + 1 < argc) {
            cuda_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "=================================================\\n";
    std::cout << "  Latency Benchmark (CUDA-Accelerated)\\n";
    std::cout << "=================================================\\n";
    std::cout << "Server:      " << server_url << "\\n";
    std::cout << "Iterations:  " << iters << "\\n";
    std::cout << "Max Tokens:  " << max_tokens << "\\n";
    std::cout << "CUDA Work:   " << (cuda_work ? "Enabled" : "Disabled") << "\\n";
    if (cuda_work) {
        std::cout << "CUDA Iters:  " << cuda_iters << "\\n";
    }
    std::cout << "=================================================\\n\\n";

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "CUDA Device: " << prop.name << "\\n\\n";
    }

    llcuda::InferenceEngine engine;
    llcuda::ModelConfig config;

    auto status = engine.load_model("model.gguf", config);
    if (!status) {
        std::cerr << "Warning: " << status.message << "\\n\\n";
    }

    std::cout << "Running " << iters << " iterations...\\n";

    std::vector<double> latencies_ms;
    latencies_ms.reserve(iters);

    for (int i = 0; i < iters; i++) {
        llcuda::InferRequest request;
        request.prompt = "Hello";
        request.max_tokens = max_tokens;

        auto t_start = std::chrono::high_resolution_clock::now();
        auto result = engine.infer(request);
        auto t_end = std::chrono::high_resolution_clock::now();

        if (result.success) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            latencies_ms.push_back(elapsed_ms);

            if (cuda_work) {
                run_cuda_latency_work(result.tokens_generated * 100, cuda_iters);
            }

            if ((i + 1) % 10 == 0) {
                std::cout << "  Progress: " << (i + 1) << "/" << iters << "\\r" << std::flush;
            }
        }
    }

    std::cout << "\\n\\n";

    if (latencies_ms.empty()) {
        std::cerr << "No successful inferences\\n";
        return 1;
    }

    double p50 = percentile(latencies_ms, 0.50);
    double p95 = percentile(latencies_ms, 0.95);
    double p99 = percentile(latencies_ms, 0.99);
    double min = latencies_ms.front();
    double max = latencies_ms.back();

    std::cout << "=================================================\\n";
    std::cout << "  Latency Results (milliseconds)\\n";
    std::cout << "=================================================\\n";
    printf("  Min:      %.2f ms\\n", min);
    printf("  p50:      %.2f ms\\n", p50);
    printf("  p95:      %.2f ms\\n", p95);
    printf("  p99:      %.2f ms\\n", p99);
    printf("  Max:      %.2f ms\\n", max);
    std::cout << "=================================================\\n";

    return 0;
}
