# Rust 2024 Edition Research Notes

## Breaking Changes
- **Temporary Scope Changes**:
    - `if let` temporaries are dropped after the `if` branch, not after the `else` block.
    - Tail expression temporaries in blocks are dropped before local variables.
- **Unsafe Code**:
    - `unsafe_op_in_unsafe_fn` is now warn-by-default.
    - `extern` blocks must be marked `unsafe`.
    - Attributes like `no_mangle`, `export_name`, `link_section` require `unsafe`.
    - `static mut` references are deny-by-default.
    - `std::env::set_var` and `remove_var` are now `unsafe`.
- **Keywords**:
    - `gen` is reserved.
- **Macros**:
    - Stricter declarative macro matching.
    - `expr` fragment now matches `const` and `_`. Use `expr_2021` for old behavior.
- **Traits**:
    - `impl Trait` capturing rules changed (use `use<..>` to be explicit).
    - `!` type fallback changes.

## New Idioms & Features
- **Async Closures**: `async || {}` supported.
- **Prelude**: `Future` and `IntoFuture` added.
- **Control Flow**: Improved `if let` / `while let` temporary scopes allow cleaner code.
- **Return Types**: `impl Trait` encouraged for return types.
- **Formatting**: `rustfmt` style editions.

## Migration Strategy
1. **Automated**: Use `cargo fix --edition`.
2. **Manual**:
    - Check for `unsafe` blocks in `unsafe fn`.
    - Review `extern` blocks.
    - Check for `static mut` usage.
    - Review `if let` temporary lifetimes if relying on extended drop order.
