# Contributing to Local LLaMA CUDA

Thank you for your interest in contributing to Local LLaMA CUDA!

## Development Setup

1. Install dependencies:
   - CUDA Toolkit 12.0+
   - OpenMPI 5.0+ (with CUDA-aware support)
   - CMake 3.24+
   - Ninja build system

2. Clone and build:
   ```bash
   git clone <repository-url>
   cd local-llama-cuda
   ./scripts/build.sh
   ```

3. Run tests:
   ```bash
   cd build && ninja test
   ```

## Code Style

- Follow C++20 modern practices
- Use RAII for resource management
- Document public APIs with Doxygen comments
- Keep functions focused and testable

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit pull request

## Reporting Issues

Please include:
- System information (GPU, CUDA version, OS)
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages
