#include "llcuda/inference_engine.hpp"
#include "llcuda/types.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

void print_usage() {
    std::cout << "Local LLaMA CUDA - High-Performance On-Device LLM Inference\n\n";
    std::cout << "Usage:\n";
    std::cout << "  llcuda infer -m <model> -p <prompt> [options]\n";
    std::cout << "  llcuda batch -m <model> -i <input> -o <output>\n";
    std::cout << "  llcuda bench -m <model> [options]\n";
    std::cout << "\n";
    std::cout << "Commands:\n";
    std::cout << "  infer     Single inference request\n";
    std::cout << "  batch     Batch processing from JSONL file\n";
    std::cout << "  bench     Run benchmarks\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model <path>       Path to GGUF model file\n";
    std::cout << "  -p, --prompt <text>      Inference prompt\n";
    std::cout << "  -i, --input <file>       Input file (JSONL for batch)\n";
    std::cout << "  -o, --output <file>      Output file\n";
    std::cout << "  -t, --temperature <f>    Temperature (default: 0.7)\n";
    std::cout << "  --max-tokens <n>         Max tokens to generate (default: 128)\n";
    std::cout << "  --gpu-layers <n>         GPU layers to offload (default: 0)\n";
    std::cout << "  --server <url>           llama-server URL (default: http://127.0.0.1:8090)\n";
    std::cout << "  --stream                 Enable streaming output\n";
    std::cout << "  --iters <n>              Benchmark iterations (default: 10)\n";
    std::cout << "  -h, --help               Show this help\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  llcuda infer -m gemma-1b.gguf -p \"What is AI?\"\n";
    std::cout << "  llcuda batch -m model.gguf -i prompts.jsonl -o results.jsonl\n";
    std::cout << "  llcuda bench -m model.gguf --iters 100\n";
}

int cmd_infer(int argc, char** argv) {
    std::string model_path;
    std::string prompt;
    std::string server_url = "http://127.0.0.1:8090";
    float temperature = 0.7f;
    uint32_t max_tokens = 128;
    uint32_t gpu_layers = 0;
    bool stream = false;

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--server" && i + 1 < argc) {
            server_url = argv[++i];
        } else if ((arg == "-t" || arg == "--temperature") && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--gpu-layers" && i + 1 < argc) {
            gpu_layers = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--stream") {
            stream = true;
        }
    }

    if (model_path.empty() || prompt.empty()) {
        std::cerr << "Error: Model path and prompt are required\n";
        return 1;
    }

    // Create inference engine
    llcuda::InferenceEngine engine;

    // Configure model
    llcuda::ModelConfig config;
    config.gpu_layers = gpu_layers;

    std::cout << "Loading model: " << model_path << "\n";
    auto status = engine.load_model(model_path, config);
    if (!status) {
        std::cerr << "Error loading model: " << status.message << "\n";
        return 1;
    }

    // Create request
    llcuda::InferRequest request;
    request.prompt = prompt;
    request.temperature = temperature;
    request.max_tokens = max_tokens;
    request.stream = stream;

    std::cout << "Running inference...\n";

    llcuda::InferResult result;
    if (stream) {
        result = engine.infer_stream(request, [](const std::string& chunk) {
            std::cout << chunk << std::flush;
        });
        std::cout << "\n";
    } else {
        result = engine.infer(request);
        if (result.success) {
            std::cout << "\nResponse:\n" << result.text << "\n";
        }
    }

    if (result.success) {
        std::cout << "\nMetrics:\n";
        std::cout << "  Tokens: " << result.tokens_generated << "\n";
        std::cout << "  Latency: " << result.latency_ms << " ms\n";
        std::cout << "  Throughput: " << result.tokens_per_sec << " tokens/sec\n";
        return 0;
    } else {
        std::cerr << "Error: " << result.error_message << "\n";
        return 1;
    }
}

int cmd_batch(int argc, char** argv) {
    std::string model_path;
    std::string input_file;
    std::string output_file;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            input_file = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    if (model_path.empty() || input_file.empty() || output_file.empty()) {
        std::cerr << "Error: Model, input, and output files are required\n";
        return 1;
    }

    // Load model
    llcuda::InferenceEngine engine;
    llcuda::ModelConfig config;

    std::cout << "Loading model: " << model_path << "\n";
    auto status = engine.load_model(model_path, config);
    if (!status) {
        std::cerr << "Error: " << status.message << "\n";
        return 1;
    }

    // Read prompts from JSONL
    std::ifstream input(input_file);
    if (!input.good()) {
        std::cerr << "Error: Cannot open input file\n";
        return 1;
    }

    std::ofstream output(output_file);
    if (!output.good()) {
        std::cerr << "Error: Cannot open output file\n";
        return 1;
    }

    std::string line;
    int count = 0;

    std::cout << "Processing batch...\n";

    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;

        // Simple JSON parsing - extract prompt field
        size_t prompt_start = line.find("\"prompt\"");
        if (prompt_start == std::string::npos) continue;

        size_t value_start = line.find(':', prompt_start);
        if (value_start == std::string::npos) continue;

        size_t quote1 = line.find('"', value_start);
        if (quote1 == std::string::npos) continue;

        size_t quote2 = line.find('"', quote1 + 1);
        if (quote2 == std::string::npos) continue;

        std::string prompt = line.substr(quote1 + 1, quote2 - quote1 - 1);

        llcuda::InferRequest request;
        request.prompt = prompt;

        double t0 = llcuda::now_ms();
        auto result = engine.infer(request);
        double t1 = llcuda::now_ms();

        // Write result as JSONL
        output << "{\"prompt\":\"" << prompt << "\",\"response\":\""
               << result.text << "\",\"latency_ms\":" << (t1 - t0)
               << ",\"success\":" << (result.success ? "true" : "false") << "}\n";

        count++;
        if (count % 10 == 0) {
            std::cout << "Processed " << count << " requests...\n";
        }
    }

    std::cout << "Completed " << count << " requests\n";

    // Print metrics
    auto metrics = engine.get_metrics();
    std::cout << "\nMetrics:\n";
    std::cout << "  Mean latency: " << metrics.latency.mean_ms << " ms\n";
    std::cout << "  P50: " << metrics.latency.p50_ms << " ms\n";
    std::cout << "  P95: " << metrics.latency.p95_ms << " ms\n";
    std::cout << "  P99: " << metrics.latency.p99_ms << " ms\n";
    std::cout << "  Total tokens: " << metrics.throughput.total_tokens << "\n";

    return 0;
}

int cmd_bench(int argc, char** argv) {
    std::string model_path;
    int iters = 10;
    uint32_t gpu_layers = 0;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--gpu-layers" && i + 1 < argc) {
            gpu_layers = static_cast<uint32_t>(std::stoi(argv[++i]));
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: Model path is required\n";
        return 1;
    }

    llcuda::InferenceEngine engine;
    llcuda::ModelConfig config;
    config.gpu_layers = gpu_layers;

    std::cout << "=== LLaMA CUDA Benchmark ===\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "GPU Layers: " << gpu_layers << "\n";
    std::cout << "Iterations: " << iters << "\n\n";

    std::cout << "Loading model...\n";
    auto status = engine.load_model(model_path, config);
    if (!status) {
        std::cerr << "Error: " << status.message << "\n";
        return 1;
    }

    // Warmup
    std::cout << "Warming up...\n";
    llcuda::InferRequest warmup_req;
    warmup_req.prompt = "Hello";
    warmup_req.max_tokens = 10;

    for (int i = 0; i < 3; ++i) {
        engine.infer(warmup_req);
    }

    engine.reset_metrics();

    // Benchmark
    std::cout << "Running benchmark...\n";
    llcuda::InferRequest bench_req;
    bench_req.prompt = "Explain quantum computing in simple terms.";
    bench_req.max_tokens = 50;

    for (int i = 0; i < iters; ++i) {
        auto result = engine.infer(bench_req);
        std::cout << "  Iteration " << (i + 1) << "/" << iters
                  << ": " << result.latency_ms << " ms\n";
    }

    // Results
    auto metrics = engine.get_metrics();
    std::cout << "\n=== Results ===\n";
    std::cout << "Latency (ms):\n";
    std::cout << "  Mean: " << metrics.latency.mean_ms << "\n";
    std::cout << "  P50:  " << metrics.latency.p50_ms << "\n";
    std::cout << "  P95:  " << metrics.latency.p95_ms << "\n";
    std::cout << "  P99:  " << metrics.latency.p99_ms << "\n";
    std::cout << "  Min:  " << metrics.latency.min_ms << "\n";
    std::cout << "  Max:  " << metrics.latency.max_ms << "\n";
    std::cout << "\nThroughput:\n";
    std::cout << "  Tokens/sec: " << metrics.throughput.tokens_per_sec << "\n";
    std::cout << "  Total tokens: " << metrics.throughput.total_tokens << "\n";

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];

    if (command == "infer") {
        return cmd_infer(argc, argv);
    } else if (command == "batch") {
        return cmd_batch(argc, argv);
    } else if (command == "bench") {
        return cmd_bench(argc, argv);
    } else if (command == "-h" || command == "--help") {
        print_usage();
        return 0;
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        print_usage();
        return 1;
    }
}
