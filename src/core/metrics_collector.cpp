#include "llcuda/metrics.hpp"
#include <algorithm>
#include <numeric>

namespace llcuda {

MetricsCollector::MetricsCollector() 
    : start_time_(std::chrono::system_clock::now()) {
}

MetricsCollector::~MetricsCollector() = default;

void MetricsCollector::record_latency(double latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    latency_samples_.push_back(latency_ms);
    latency_sum_ += latency_ms;
}

void MetricsCollector::record_tokens(uint64_t token_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_tokens_ += token_count;
}

void MetricsCollector::record_request() {
    std::lock_guard<std::mutex> lock(mutex_);
    total_requests_++;
}

void MetricsCollector::record_gpu_metrics(const GPUMetrics& metrics) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_gpu_metrics_ = metrics;
}

LatencyMetrics MetricsCollector::get_latency_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    LatencyMetrics metrics;
    metrics.sample_count = latency_samples_.size();
    
    if (latency_samples_.empty()) {
        return metrics;
    }
    
    auto sorted = latency_samples_;
    metrics.mean_ms = latency_sum_ / static_cast<double>(latency_samples_.size());
    metrics.p50_ms = calculate_percentile(sorted, 0.50);
    metrics.p95_ms = calculate_percentile(sorted, 0.95);
    metrics.p99_ms = calculate_percentile(sorted, 0.99);
    metrics.min_ms = *std::min_element(sorted.begin(), sorted.end());
    metrics.max_ms = *std::max_element(sorted.begin(), sorted.end());
    
    return metrics;
}

ThroughputMetrics MetricsCollector::get_throughput_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    ThroughputMetrics metrics;
    metrics.total_tokens = total_tokens_;
    metrics.total_requests = total_requests_;
    
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration<double>(now - start_time_).count();
    
    if (elapsed > 0) {
        metrics.tokens_per_sec = static_cast<double>(total_tokens_) / elapsed;
        metrics.requests_per_sec = static_cast<double>(total_requests_) / elapsed;
    }
    
    return metrics;
}

GPUMetrics MetricsCollector::get_gpu_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_gpu_metrics_;
}

SystemMetrics MetricsCollector::get_system_metrics() const {
    SystemMetrics metrics;
    metrics.latency = get_latency_metrics();
    metrics.throughput = get_throughput_metrics();
    metrics.gpu = get_gpu_metrics();
    metrics.timestamp = std::chrono::system_clock::now();
    return metrics;
}

void MetricsCollector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    latency_samples_.clear();
    latency_sum_ = 0.0;
    total_tokens_ = 0;
    total_requests_ = 0;
    start_time_ = std::chrono::system_clock::now();
}

std::vector<double> MetricsCollector::get_latency_samples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latency_samples_;
}

double MetricsCollector::calculate_percentile(const std::vector<double>& samples, double p) {
    if (samples.empty()) return 0.0;
    
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    
    double idx = p * static_cast<double>(sorted.size() - 1);
    size_t i0 = static_cast<size_t>(idx);
    size_t i1 = std::min(i0 + 1, sorted.size() - 1);
    double frac = idx - static_cast<double>(i0);
    
    return sorted[i0] * (1.0 - frac) + sorted[i1] * frac;
}

} // namespace llcuda
