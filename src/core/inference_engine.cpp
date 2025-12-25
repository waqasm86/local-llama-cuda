#include "llcuda/inference_engine.hpp"
#include "llcuda/model_manager.hpp"
#include "llcuda/metrics.hpp"
#include "llcuda/http_client.hpp"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <stdexcept>

namespace llcuda {

class InferenceEngine::Impl {
public:
    Impl() : metrics_(std::make_unique<MetricsCollector>()) {}

    ModelManager model_manager;
    std::unique_ptr<MetricsCollector> metrics_;
    ModelConfig current_config;
    std::string llama_server_url = "http://127.0.0.1:8090";
    bool model_loaded = false;
    mutable std::mutex mutex_;

    std::string build_request_json(const InferRequest& req) {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"prompt\": \"" << escape_json(req.prompt) << "\",\n";
        oss << "  \"n_predict\": " << req.max_tokens << ",\n";
        oss << "  \"temperature\": " << req.temperature << ",\n";
        oss << "  \"top_p\": " << req.top_p << ",\n";
        oss << "  \"top_k\": " << req.top_k << ",\n";
        if (req.seed != 0) {
            oss << "  \"seed\": " << req.seed << ",\n";
        }
        oss << "  \"stream\": " << (req.stream ? "true" : "false") << "\n";
        oss << "}";
        return oss.str();
    }

    std::string escape_json(const std::string& input) {
        std::string output;
        output.reserve(input.size() * 2);
        for (char c : input) {
            switch (c) {
                case '\n': output += "\\n"; break;
                case '\r': output += "\\r"; break;
                case '\t': output += "\\t"; break;
                case '"':  output += "\\\""; break;
                case '\\': output += "\\\\"; break;
                default:
                    if (static_cast<unsigned char>(c) < 32) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        output += buf;
                    } else {
                        output += c;
                    }
            }
        }
        return output;
    }

    std::string extract_json_field(const std::string& json, const std::string& field) {
        // Simple JSON field extraction - looks for "field":"value"
        std::string pattern = "\"" + field + "\":\"";
        size_t start = json.find(pattern);
        if (start == std::string::npos) {
            // Try without quotes (numeric fields)
            pattern = "\"" + field + "\":";
            start = json.find(pattern);
            if (start == std::string::npos) return "";
            start += pattern.size();
            size_t end = json.find_first_of(",}\n", start);
            if (end == std::string::npos) return "";
            return json.substr(start, end - start);
        }
        start += pattern.size();
        size_t end = json.find('"', start);
        if (end == std::string::npos) return "";
        return json.substr(start, end - start);
    }

    std::string unescape_json(const std::string& input) {
        std::string output;
        output.reserve(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            if (input[i] == '\\' && i + 1 < input.size()) {
                switch (input[i + 1]) {
                    case 'n': output += '\n'; ++i; break;
                    case 'r': output += '\r'; ++i; break;
                    case 't': output += '\t'; ++i; break;
                    case '"': output += '"'; ++i; break;
                    case '\\': output += '\\'; ++i; break;
                    default: output += input[i]; break;
                }
            } else {
                output += input[i];
            }
        }
        return output;
    }
};

InferenceEngine::InferenceEngine()
    : impl_(std::make_unique<Impl>()) {}

InferenceEngine::~InferenceEngine() = default;

InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;
InferenceEngine& InferenceEngine::operator=(InferenceEngine&&) noexcept = default;

Status InferenceEngine::load_model(const std::string& model_path, const ModelConfig& config) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);

    // Verify model file exists
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        return Status::Error("Model file not found: " + model_path);
    }

    // Store configuration
    impl_->current_config = config;
    impl_->current_config.model_path = model_path;

    // Load model metadata
    auto manifest = impl_->model_manager.load_model(model_path);
    if (manifest) {
        impl_->current_config.sha256_hash = manifest->sha256;
    }

    impl_->model_loaded = true;
    return Status::Ok();
}

InferResult InferenceEngine::infer(const InferRequest& request) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);

    if (!impl_->model_loaded) {
        InferResult result;
        result.success = false;
        result.error_message = "No model loaded";
        return result;
    }

    InferResult result;
    double t0 = now_ms();

    try {
        // Build request JSON
        std::string req_json = impl_->build_request_json(request);

        // Call llama-server via HTTP
        HttpClient client;
        auto response = client.post(impl_->llama_server_url + "/completion", req_json);

        double t1 = now_ms();
        result.latency_ms = t1 - t0;

        if (response.status >= 200 && response.status < 300) {
            std::string resp_json(response.body.begin(), response.body.end());

            // Extract completion text (try multiple field names)
            result.text = impl_->extract_json_field(resp_json, "content");
            if (result.text.empty()) {
                result.text = impl_->extract_json_field(resp_json, "response");
            }
            if (result.text.empty()) {
                result.text = impl_->extract_json_field(resp_json, "completion");
            }
            if (result.text.empty()) {
                result.text = impl_->extract_json_field(resp_json, "text");
            }

            result.text = impl_->unescape_json(result.text);

            // Extract token count if available
            std::string tokens_str = impl_->extract_json_field(resp_json, "tokens_predicted");
            if (!tokens_str.empty()) {
                result.tokens_generated = static_cast<uint32_t>(std::stoi(tokens_str));
            } else {
                // Estimate based on whitespace
                result.tokens_generated = static_cast<uint32_t>(
                    std::count(result.text.begin(), result.text.end(), ' ') + 1
                );
            }

            if (result.latency_ms > 0) {
                result.tokens_per_sec = (result.tokens_generated * 1000.0) / result.latency_ms;
            }

            result.success = true;

            // Record metrics
            impl_->metrics_->record_latency(result.latency_ms);
            impl_->metrics_->record_tokens(result.tokens_generated);
            impl_->metrics_->record_request();

        } else {
            result.success = false;
            result.error_message = "HTTP error: " + std::to_string(response.status);
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Exception: ") + e.what();
    }

    return result;
}

InferResult InferenceEngine::infer_stream(const InferRequest& request, StreamCallback callback) {
    // For now, implement as non-streaming with chunked callback
    auto result = infer(request);

    if (result.success && callback) {
        // Chunk the output for streaming simulation
        const size_t chunk_size = 32;
        for (size_t i = 0; i < result.text.size(); i += chunk_size) {
            size_t len = std::min(chunk_size, result.text.size() - i);
            callback(result.text.substr(i, len));
        }
    }

    return result;
}

std::vector<InferResult> InferenceEngine::infer_batch(const std::vector<InferRequest>& requests) {
    std::vector<InferResult> results;
    results.reserve(requests.size());

    for (const auto& req : requests) {
        results.push_back(infer(req));
    }

    return results;
}

void InferenceEngine::unload_model() {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->model_loaded = false;
}

bool InferenceEngine::is_model_loaded() const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    return impl_->model_loaded;
}

SystemMetrics InferenceEngine::get_metrics() const {
    return impl_->metrics_->get_system_metrics();
}

void InferenceEngine::reset_metrics() {
    impl_->metrics_->reset();
}

} // namespace llcuda
