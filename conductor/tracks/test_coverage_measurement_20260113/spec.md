# Specification: Add Test Coverage Measurement

## Overview
This track aims to implement test coverage measurement for the Solstrale Rust project. We will utilize `cargo-llvm-cov` to provide accurate and fast coverage analysis. The goal is to enable developers to assess test completeness locally and to provide visibility into code coverage within the CI pipeline.

## Functional Requirements
1.  **Tooling:** Implement `cargo-llvm-cov` for measuring code coverage.
2.  **Reporting Formats:**
    *   Generate **Lcov** reports (standard format for tools and integrations).
    *   Generate **Text Summary** reports (printed to stdout for quick feedback).
3.  **CI Integration:**
    *   Update the existing GitHub Actions workflow (`.github/workflows/ci.yaml`).
    *   Add a step to install `cargo-llvm-cov`.
    *   Run the coverage command in CI.
    *   Ensure the coverage summary is visible in the CI logs.

## Non-Functional Requirements
*   **Performance:** Use LLVM source-based coverage to ensure minimal overhead during test execution.
*   **Maintainability:** Ensure the setup is easy to use for other developers (e.g., documented in `README.md` or a script).

## Acceptance Criteria
*   [ ] A command is available (or documented) to run tests with coverage locally.
*   [ ] Running the local command produces a text summary in the terminal.
*   [ ] Running the local command generates an `lcov.info` file (or similar Lcov output).
*   [ ] The GitHub Actions CI pipeline passes successfully.
*   [ ] The CI run logs show the test coverage summary.

## Out of Scope
*   Uploading coverage reports to third-party services (e.g., Codecov) is deferred for a future track.
*   Enforcing a specific coverage percentage threshold in CI.
