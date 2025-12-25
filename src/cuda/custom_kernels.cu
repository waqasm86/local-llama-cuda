#include <cuda_runtime.h>
#include <cstdint>

namespace llcuda {

// Matrix multiplication kernel (simple implementation)
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ReLU activation kernel
__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// GELU activation kernel (approximate)
__global__ void gelu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        data[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// Quantization kernel (FP32 -> INT8)
__global__ void quantize_int8_kernel(const float* input, int8_t* output,
                                     float* scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        float absmax = 0.0f;
        for (int i = 0; i < n; i++) {
            absmax = fmaxf(absmax, fabsf(input[i]));
        }
        *scale = absmax / 127.0f;
    }
    __syncthreads();

    if (idx < n) {
        float val = input[idx] / (*scale);
        output[idx] = static_cast<int8_t>(roundf(fminf(fmaxf(val, -127.0f), 127.0f)));
    }
}

} // namespace llcuda
