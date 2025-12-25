#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/build"

echo "=== Running Quick Test ==="

# Ensure build exists
if [ ! -f "llcuda" ]; then
    echo "Error: Build first with ./scripts/build.sh"
    exit 1
fi

# Run help command
echo "Testing help command..."
./llcuda --help

echo ""
echo "=== Quick Test Complete ==="
echo ""
echo "To run full tests: cd build && ninja test"
echo "To run inference: ./llcuda infer -m <model> -p \"prompt\""
