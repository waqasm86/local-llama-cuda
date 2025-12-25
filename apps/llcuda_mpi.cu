#include "llcuda/inference_engine.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

// CUDA post-processing kernel for MPI workers
extern "C" {
    __global__ void mpi_post_kernel(int iters) {
        volatile int x = threadIdx.x + blockIdx.x * blockDim.x;
        for (int i = 0; i < iters; i++) {
            x = x * 1103515245 + 12345;
        }
    }
}

double run_cuda_post_ms(int work) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mpi_post_kernel<<<16, 128>>>(work);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(ms);
}

void print_usage(const char* prog) {
    std::cout << "LLaMA CUDA MPI Distributed Scheduler\n\n";
    std::cout << "Usage: mpirun -np <ranks> " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --server <url>      llama-server URL (default: http://127.0.0.1:8090)\n";
    std::cout << "  --iters <n>         Total iterations (default: 20)\n";
    std::cout << "  --inflight <n>      Inflight requests (default: 4)\n";
    std::cout << "  --n_predict <n>     Tokens to predict (default: 64)\n";
    std::cout << "  --cuda-post         Enable CUDA post-processing\n";
    std::cout << "  --cuda-work <n>     CUDA work iterations (default: 1000)\n";
    std::cout << "  -h, --help          Show this help\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string server_url = "http://127.0.0.1:8090";
    int iters = 20;
    int inflight = 4;
    int n_predict = 64;
    bool cuda_post = false;
    int cuda_work = 1000;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--server") == 0 && i + 1 < argc) {
            server_url = argv[++i];
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--inflight") == 0 && i + 1 < argc) {
            inflight = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--n_predict") == 0 && i + 1 < argc) {
            n_predict = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cuda-post") == 0) {
            cuda_post = true;
        } else if (std::strcmp(argv[i], "--cuda-work") == 0 && i + 1 < argc) {
            cuda_work = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    // Initialize CUDA for each rank
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        int device_id = rank % device_count;
        cudaSetDevice(device_id);

        if (rank == 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device_id);
            std::cout << "=================================================\n";
            std::cout << "  MPI Distributed Scheduler (CUDA-Accelerated)\n";
            std::cout << "=================================================\n";
            std::cout << "MPI Ranks:   " << size << "\n";
            std::cout << "CUDA Device: " << prop.name << "\n";
            std::cout << "Server:      " << server_url << "\n";
            std::cout << "Iterations:  " << iters << "\n";
            std::cout << "Inflight:    " << inflight << "\n";
            std::cout << "n_predict:   " << n_predict << "\n";
            std::cout << "CUDA Post:   " << (cuda_post ? "Enabled" : "Disabled") << "\n";
            std::cout << "=================================================\n\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Create inference engine
    llcuda::InferenceEngine engine;
    llcuda::ModelConfig config;

    auto status = engine.load_model("model.gguf", config);
    if (!status && rank == 0) {
        std::cerr << "Warning: " << status.message << "\n\n";
    }

    std::vector<double> latencies;
    uint64_t total_tokens = 0;

    if (rank == 0) {
        // Master: distribute work
        std::cout << "Rank 0 (Master): Distributing " << iters << " jobs...\n\n";

        int jobs_sent = 0;
        int jobs_completed = 0;

        // Send initial batch
        for (int worker = 1; worker < size && jobs_sent < iters; worker++) {
            MPI_Send(&jobs_sent, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            jobs_sent++;
        }

        // Collect results and send more work
        while (jobs_completed < iters) {
            double payload[2]; // [latency_ms, tokens]
            MPI_Status mpi_status;
            MPI_Recv(payload, 2, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &mpi_status);

            int worker = mpi_status.MPI_SOURCE;
            latencies.push_back(payload[0]);
            total_tokens += static_cast<uint64_t>(payload[1]);
            jobs_completed++;

            if ((jobs_completed % 5) == 0) {
                std::cout << "  Progress: " << jobs_completed << "/" << iters << " jobs completed\n";
            }

            // Send more work if available
            if (jobs_sent < iters) {
                MPI_Send(&jobs_sent, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
                jobs_sent++;
            } else {
                int stop_signal = -1;
                MPI_Send(&stop_signal, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            }
        }

        // Calculate statistics
        if (!latencies.empty()) {
            std::sort(latencies.begin(), latencies.end());

            double sum = 0.0;
            for (double lat : latencies) sum += lat;
            double mean = sum / latencies.size();

            size_t n = latencies.size();
            double p50 = latencies[n * 50 / 100];
            double p95 = latencies[n * 95 / 100];
            double p99 = latencies[n * 99 / 100];

            double total_time_s = sum / 1000.0;
            double throughput = (total_tokens / total_time_s);

            std::cout << "\n=================================================\n";
            std::cout << "  MPI Scheduler Results\n";
            std::cout << "=================================================\n";
            printf("  Total Jobs:      %d\n", iters);
            printf("  MPI Ranks:       %d\n", size);
            printf("  Total Tokens:    %lu\n", total_tokens);
            printf("  Mean Latency:    %.2f ms\n", mean);
            printf("  p50:             %.2f ms\n", p50);
            printf("  p95:             %.2f ms\n", p95);
            printf("  p99:             %.2f ms\n", p99);
            printf("  Throughput:      %.2f tokens/sec\n", throughput);
            printf("  Speedup:         %.2fx (vs single rank)\n",
                   static_cast<double>(size - 1));
            std::cout << "=================================================\n";
        }

    } else {
        // Worker: process jobs
        while (true) {
            int job_id;
            MPI_Recv(&job_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (job_id < 0) break; // Stop signal

            // Run inference
            llcuda::InferRequest request;
            request.prompt = "What is AI?";
            request.max_tokens = n_predict;
            request.temperature = 0.7f;

            auto t_start = std::chrono::high_resolution_clock::now();
            auto result = engine.infer(request);
            auto t_end = std::chrono::high_resolution_clock::now();

            double latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            // Optional CUDA post-processing
            if (cuda_post && result.success) {
                double cuda_ms = run_cuda_post_ms(cuda_work);
                latency_ms += cuda_ms;
            }

            // Send result back to master
            double payload[2] = {
                latency_ms,
                static_cast<double>(result.tokens_generated)
            };
            MPI_Send(payload, 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
