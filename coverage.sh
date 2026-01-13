#!/bin/bash
set -e

# Run cargo-llvm-cov with provided arguments, or default to text summary
if [ $# -eq 0 ]; then
    echo "Running coverage with text summary..."
    cargo llvm-cov --workspace
else
    echo "Running coverage with arguments: $@"
    cargo llvm-cov --workspace "$@"
fi
