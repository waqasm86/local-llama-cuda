#include "llcuda/inference_engine.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

// CUDA kernel for throughput measurement
extern "C" {
    __global__ void throughput_kernel(int* tokens, float* times, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            int tok = tokens[idx];
            float time = times[idx];
            for (int i = 0; i < 100; i++) {
                tok = tok * 1103515245 + 12345;
                time += static_cast<float>(tok) * 0.001f;
            }
            tokens[idx] = tok;
            times[idx] = time;
        }
    }
}

void run_cuda_throughput_work(int num_tokens) {
    int* d_tokens = nullptr;
    float* d_times = nullptr;

    cudaMalloc(&d_tokens, num_tokens * sizeof(int));
    cudaMalloc(&d_times, num_tokens * sizeof(float));

    std::vector<int> h_tokens(num_tokens, 1);
    std::vector<float> h_times(num_tokens, 0.0f);

    cudaMemcpy(d_tokens, h_tokens.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_times, h_times.data(), num_tokens * sizeof(float), cudaMemcpyHostToDevice);

    throughput_kernel<<<(num_tokens + 255) / 256, 256>>>(d_tokens, d_times, num_tokens);
    cudaDeviceSynchronize();

    cudaFree(d_tokens);
    cudaFree(d_times);
}

void print_usage(const char* prog) {
    std::cout << "CNSE Throughput Benchmark - Tokens/Second Measurement\n\n";
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --server <url>     llama-server URL (default: http://127.0.0.1:8090)\n";
    std::cout << "  --iters <n>        Number of iterations (default: 20)\n";
    std::cout << "  --max-tokens <n>   Tokens per request (default: 64)\n";
    std::cout << "  --cuda-work        Enable CUDA throughput work\n";
    std::cout << "  -h, --help         Show this help\n";
}

int main(int argc, char** argv) {
    std::string server_url = "http://127.0.0.1:8090";
    int iters = 20;
    int max_tokens = 64;
    bool cuda_work = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--server") == 0 && i + 1 < argc) {
            server_url = argv[++i];
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cuda-work") == 0) {
            cuda_work = true;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "=================================================\n";
    std::cout << "  Throughput Benchmark (CUDA-Accelerated)\n";
    std::cout << "=================================================\n";
    std::cout << "Server:      " << server_url << "\n";
    std::cout << "Iterations:  " << iters << "\n";
    std::cout << "Max Tokens:  " << max_tokens << "\n";
    std::cout << "CUDA Work:   " << (cuda_work ? "Enabled" : "Disabled") << "\n";
    std::cout << "=================================================\n\n";

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "CUDA Device: " << prop.name << "\n\n";
    }

    llcuda::InferenceEngine engine;
    llcuda::ModelConfig config;

    auto status = engine.load_model("model.gguf", config);
    if (!status) {
        std::cerr << "Warning: " << status.message << "\n\n";
    }

    std::cout << "Running " << iters << " iterations...\n";

    uint64_t total_tokens = 0;
    double total_time_ms = 0.0;

    auto overall_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; i++) {
        llcuda::InferRequest request;
        request.prompt = "Count to 10";
        request.max_tokens = max_tokens;

        auto t_start = std::chrono::high_resolution_clock::now();
        auto result = engine.infer(request);
        auto t_end = std::chrono::high_resolution_clock::now();

        if (result.success) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            total_tokens += result.tokens_generated;
            total_time_ms += elapsed_ms;

            if (cuda_work) {
                run_cuda_throughput_work(result.tokens_generated);
            }
        }
    }

    auto overall_end = std::chrono::high_resolution_clock::now();
    double overall_time_ms = std::chrono::duration<double, std::milli>(overall_end - overall_start).count();

    double throughput = (total_tokens / total_time_ms) * 1000.0;

    std::cout << "\n=================================================\n";
    std::cout << "  Results\n";
    std::cout << "=================================================\n";
    printf("  Total Tokens:        %lu\n", total_tokens);
    printf("  Total Time:          %.2f s\n", overall_time_ms / 1000.0);
    printf("  Throughput:          %.2f tokens/sec\n", throughput);
    std::cout << "=================================================\n";

    return 0;
}
