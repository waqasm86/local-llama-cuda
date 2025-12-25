#include "llcuda/model_manager.hpp"
#include "llcuda/sha256.hpp"
#include <fstream>
#include <stdexcept>

namespace llcuda {

class ModelManager::Impl {
public:
    std::string cache_dir_ = "/tmp/llcuda_cache";
};

ModelManager::ModelManager() : impl_(std::make_unique<Impl>()) {}
ModelManager::~ModelManager() = default;
ModelManager::ModelManager(ModelManager&&) noexcept = default;
ModelManager& ModelManager::operator=(ModelManager&&) noexcept = default;

std::optional<ModelManifest> ModelManager::load_model(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        return std::nullopt;
    }

    ModelManifest manifest;
    manifest.original_name = model_path;

    // Compute SHA256 hash
    manifest.sha256 = sha256_file(model_path);

    // Get file size
    file.seekg(0, std::ios::end);
    manifest.size_bytes = file.tellg();

    return manifest;
}

std::optional<ModelManifest> ModelManager::load_by_hash(const std::string&) {
    return std::nullopt;  // Stub
}

ModelManifest ModelManager::store_model(const std::string& model_path) {
    auto manifest = load_model(model_path);
    if (!manifest) {
        throw std::runtime_error("Failed to load model");
    }
    return *manifest;
}

std::optional<ModelManifest> ModelManager::get_manifest(const std::string&) const {
    return std::nullopt;  // Stub
}

std::vector<ModelManifest> ModelManager::list_cached_models() const {
    return {};  // Stub
}

bool ModelManager::remove_from_cache(const std::string&) {
    return false;  // Stub
}

void ModelManager::clear_cache(size_t) {
    // Stub
}

std::string ModelManager::get_cache_dir() const {
    return impl_->cache_dir_;
}

void ModelManager::set_cache_dir(const std::string& path) {
    impl_->cache_dir_ = path;
}

} // namespace llcuda
