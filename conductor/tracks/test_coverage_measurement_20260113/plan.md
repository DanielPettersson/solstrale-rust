# Plan: Add Test Coverage Measurement

## Phase 1: Local Tooling Setup [checkpoint: 798ed5b]
- [x] Task: Install `cargo-llvm-cov` and required LLVM components locally
- [x] Task: Verify `cargo-llvm-cov` can generate a text summary report
- [x] Task: Verify `cargo-llvm-cov` can generate an `lcov.info` report
- [x] Task: Create a helper script `coverage.sh` for easy local execution
- [x] Task: Conductor - User Manual Verification 'Local Tooling Setup' (Protocol in workflow.md)

## Phase 2: CI Integration
- [x] Task: Update `.github/workflows/ci.yaml` to install `cargo-llvm-cov` and its dependencies
- [x] Task: Add a step to `ci.yaml` to execute coverage and output the text summary to logs
- [x] Task: Verify that the CI pipeline completes successfully and displays the coverage summary
- [ ] Task: Conductor - User Manual Verification 'CI Integration' (Protocol in workflow.md)
