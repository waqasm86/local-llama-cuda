#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>

namespace llcuda {

// Simple CUDA memory pool implementation
class CUDAMemoryPool {
public:
    CUDAMemoryPool(size_t initial_pool_size = 256 * 1024 * 1024)
        : pool_size_(initial_pool_size), allocated_bytes_(0) {
    }

    void* allocate(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->second >= bytes) {
                void* ptr = it->first;
                size_t block_size = it->second;
                free_blocks_.erase(it);
                allocated_blocks_[ptr] = block_size;
                return ptr;
            }
        }

        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                   cudaGetErrorString(err));
        }

        allocated_blocks_[ptr] = bytes;
        allocated_bytes_ += bytes;
        return ptr;
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) {
            return;
        }

        size_t size = it->second;
        allocated_blocks_.erase(it);
        free_blocks_[ptr] = size;
    }

    ~CUDAMemoryPool() {
        for (const auto& block : allocated_blocks_) {
            cudaFree(block.first);
        }
        for (const auto& block : free_blocks_) {
            cudaFree(block.first);
        }
    }

private:
    size_t pool_size_;
    size_t allocated_bytes_;
    std::map<void*, size_t> allocated_blocks_;
    std::map<void*, size_t> free_blocks_;
    mutable std::mutex mutex_;
};

} // namespace llcuda
