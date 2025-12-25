#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

namespace llcuda {

// CUDA stream manager for concurrent kernel execution
class CUDAStreamManager {
public:
    CUDAStreamManager(int num_streams = 4) : num_streams_(num_streams) {
        streams_.resize(num_streams);
        events_.resize(num_streams);

        for (int i = 0; i < num_streams; i++) {
            cudaError_t err = cudaStreamCreate(&streams_[i]);
            if (err != cudaSuccess) {
                cleanup();
                throw std::runtime_error(std::string("cudaStreamCreate failed: ") +
                                       cudaGetErrorString(err));
            }

            err = cudaEventCreate(&events_[i]);
            if (err != cudaSuccess) {
                cleanup();
                throw std::runtime_error(std::string("cudaEventCreate failed: ") +
                                       cudaGetErrorString(err));
            }
        }
    }

    ~CUDAStreamManager() {
        cleanup();
    }

    cudaStream_t get_stream(int idx) {
        if (idx < 0 || idx >= num_streams_) {
            idx = idx % num_streams_;
        }
        return streams_[idx];
    }

    void synchronize_all() {
        for (int i = 0; i < num_streams_; i++) {
            cudaStreamSynchronize(streams_[i]);
        }
    }

private:
    void cleanup() {
        for (auto stream : streams_) {
            if (stream) cudaStreamDestroy(stream);
        }
        for (auto event : events_) {
            if (event) cudaEventDestroy(event);
        }
    }

    int num_streams_;
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
};

} // namespace llcuda
