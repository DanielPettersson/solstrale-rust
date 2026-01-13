# Plan: Add Test Coverage Measurement

## Phase 1: Local Tooling Setup
- [x] Task: Install `cargo-llvm-cov` and required LLVM components locally
- [x] Task: Verify `cargo-llvm-cov` can generate a text summary report
- [x] Task: Verify `cargo-llvm-cov` can generate an `lcov.info` report
- [ ] Task: Create a helper script `coverage.sh` for easy local execution
- [ ] Task: Conductor - User Manual Verification 'Local Tooling Setup' (Protocol in workflow.md)

## Phase 2: CI Integration
- [ ] Task: Update `.github/workflows/ci.yaml` to install `cargo-llvm-cov` and its dependencies
- [ ] Task: Add a step to `ci.yaml` to execute coverage and output the text summary to logs
- [ ] Task: Verify that the CI pipeline completes successfully and displays the coverage summary
- [ ] Task: Conductor - User Manual Verification 'CI Integration' (Protocol in workflow.md)
