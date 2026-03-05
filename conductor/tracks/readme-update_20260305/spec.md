# Specification: Update README.md

## Overview
Update the `README.md` file to accurately reflect the project's current status, features, and usage instructions, ensuring it aligns with the `product.md` and the existing codebase.

## Functional Requirements
1. **Update Features List:** Ensure the following features are highlighted as implemented:
   - **Core Engine:** WGPU-based GPU Monte Carlo path tracer with global illumination, caustics, reflections, and refractions.
   - **Performance & Loading:** OBJ model loading with full material support and multithreaded BVH creation using Rayon.
   - **Post-Processing:** Custom GPU-accelerated Bloom and Saturation filters via compute shaders.
2. **Update Installation & Usage:** Provide clear, updated instructions for:
   - Building the project using `cargo build`.
   - Running the test suite using `cargo test`.
   - Any other essential usage commands.
3. **General Cleanup:** Remove any outdated or placeholder information that no longer reflects the project's state.

## Acceptance Criteria
- `README.md` includes a comprehensive "Features" section with Core Engine, Performance, and Post-Processing details.
- Build and test instructions are accurate and follow the project's workflow.
- No obsolete or misleading information remains in the file.

## Out of Scope
- Documenting the `coverage.sh`, `profile.sh`, and `build.sh` scripts (as per user preference).
- Adding or modifying implementation code.
- Updating non-README documentation.
