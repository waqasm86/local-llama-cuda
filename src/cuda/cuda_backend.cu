#include "llcuda/types.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

namespace llcuda {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// Simple vector addition kernel for testing
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Spin kernel for synthetic workload (from cuda-tcp-llama.cpp)
__global__ void spin_kernel(uint32_t iters, uint32_t* out) {
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    for (uint32_t i = 0; i < iters; i++) {
        x = x * 1664525u + 1013904223u;  // LCG pseudo-random
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = x;
}

// Post-processing kernel (from cuda-mpi-llama-scheduler)
__global__ void post_kernel(int iters) {
    volatile int x = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < iters; i++) {
        x = x * 1103515245 + 12345;  // LCG
    }
}

class CUDABackend {
public:
    CUDABackend() : device_id_(0), initialized_(false) {}

    Status init(int device_id = 0) {
        try {
            device_id_ = device_id;
            CUDA_CHECK(cudaSetDevice(device_id_));

            // Get device properties
            CUDA_CHECK(cudaGetDeviceProperties(&device_props_, device_id_));

            initialized_ = true;
            return Status::Ok();
        } catch (const std::exception& e) {
            return Status::Error(e.what());
        }
    }

    Status run_vector_add(const float* h_a, const float* h_b, float* h_c, int n) {
        if (!initialized_) return Status::Error("Backend not initialized");

        try {
            const int size = n * sizeof(float);

            // Allocate device memory
            float *d_a, *d_b, *d_c;
            CUDA_CHECK(cudaMalloc(&d_a, size));
            CUDA_CHECK(cudaMalloc(&d_b, size));
            CUDA_CHECK(cudaMalloc(&d_c, size));

            // Copy to device
            CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

            // Launch kernel
            int threads_per_block = 256;
            int blocks = (n + threads_per_block - 1) / threads_per_block;
            vectorAdd<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy result back
            CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

            // Cleanup
            CUDA_CHECK(cudaFree(d_a));
            CUDA_CHECK(cudaFree(d_b));
            CUDA_CHECK(cudaFree(d_c));

            return Status::Ok();
        } catch (const std::exception& e) {
            return Status::Error(e.what());
        }
    }

    Status run_spin_kernel(uint32_t iters, uint32_t num_tokens, uint32_t& result) {
        if (!initialized_) return Status::Error("Backend not initialized");

        try {
            uint32_t* d_out = nullptr;
            CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint32_t)));

            // Run kernel for each token (simulated inference)
            for (uint32_t i = 0; i < num_tokens; i++) {
                spin_kernel<<<8, 256>>>(iters, d_out);
            }

            CUDA_CHECK(cudaDeviceSynchronize());

            // Get result
            CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaFree(d_out));
            return Status::Ok();
        } catch (const std::exception& e) {
            return Status::Error(e.what());
        }
    }

    double run_post_kernel_timed(int work_iters) {
        if (!initialized_) return -1.0;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        post_kernel<<<16, 128>>>(work_iters);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return static_cast<double>(ms);
    }

    void print_device_info() const {
        if (!initialized_) return;

        printf("CUDA Device %d: %s\n", device_id_, device_props_.name);
        printf("  Compute Capability: %d.%d\n", device_props_.major, device_props_.minor);
        printf("  Total Global Memory: %.2f MB\n",
               device_props_.totalGlobalMem / (1024.0 * 1024.0));
        printf("  Max Threads per Block: %d\n", device_props_.maxThreadsPerBlock);
        printf("  Multiprocessors: %d\n", device_props_.multiProcessorCount);
        printf("  Warp Size: %d\n", device_props_.warpSize);
        printf("  Memory Clock Rate: %.2f GHz\n",
               device_props_.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d-bit\n", device_props_.memoryBusWidth);
    }

    ~CUDABackend() {
        if (initialized_) {
            cudaDeviceReset();
        }
    }

private:
    int device_id_;
    bool initialized_;
    cudaDeviceProp device_props_;
};

// Global instance (for simple usage)
static CUDABackend g_backend;

// Public API functions
Status cuda_init(int device_id) {
    return g_backend.init(device_id);
}

Status cuda_vector_add(const float* a, const float* b, float* c, int n) {
    return g_backend.run_vector_add(a, b, c, n);
}

Status cuda_spin_work(uint32_t iters, uint32_t tokens, uint32_t& result) {
    return g_backend.run_spin_kernel(iters, tokens, result);
}

double cuda_post_process_ms(int work) {
    return g_backend.run_post_kernel_timed(work);
}

void cuda_print_device_info() {
    g_backend.print_device_info();
}

} // namespace llcuda
