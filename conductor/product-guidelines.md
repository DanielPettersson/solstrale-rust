# Product Guidelines

## Tone and Voice
- **Minimalist and Pragmatic:** Documentation and code comments should be succinct and focused strictly on API usage, implementation mechanics, and expected behavior. Avoid verbose explanations of general graphics theory unless critical for using a specific API.

## Documentation Priorities
- **API Documentation (Rustdoc):** The highest priority is comprehensive, standard Rust documentation for all public structs, enums, and traits to ensure the library is consumable.
- **Example Scenes:** Maintain a robust collection of example scenes that practically demonstrate the library's features and configuration options.

## Visual Identity
- **Photorealistic:** Rendered output examples and demo scenes should aim for high visual fidelity and realism to showcase the engine's capabilities.

## Performance & Engineering Standards
- **Strict Efficiency:**
    - Adhere to low-level optimization patterns as a default standard.
    - Minimize heap allocations in the hot path (rendering loop).
    - Prioritize data locality and cache-friendly structures.
    - Leverage multithreading (Rayon) effectively for all parallelizable workloads.
