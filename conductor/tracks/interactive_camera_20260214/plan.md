# Implementation Plan: Interactive Camera Rendering

## Phase 1: Renderer Foundation
- [ ] Task: Implement `update_camera` method in `Renderer`
    - [ ] Write unit test in `src/renderer/mod.rs` to verify that `update_camera` correctly updates the GPU `camera_buffer`
    - [ ] Implement `update_camera` in `src/renderer/mod.rs`
    - [ ] Verify test passes
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Renderer Foundation' (Protocol in workflow.md)

## Phase 2: Interactive Render Loop
- [ ] Task: Update `render` function signature and basic restart logic
    - [ ] Update `render` signature in `src/renderer/mod.rs` to include `camera_config: &Receiver<CameraConfig>` and `idle: bool`
    - [ ] Modify the render loop to check for camera updates at the start of each iteration
    - [ ] Implement the logic to call `update_camera` and reset `sample` to 0 when an update is received
- [ ] Task: Implement Idle Mode
    - [ ] Add logic at the end of the `samples_per_pixel` loop to enter a polling loop if `idle` is true
    - [ ] Implement polling for `camera_config` and `abort` signals with a 10ms sleep in the idle state
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Interactive Render Loop' (Protocol in workflow.md)

## Phase 3: Integration and Final Verification
- [ ] Task: Verify interactive rendering with an integration test
    - [ ] Create/Update an integration test that simulates camera updates via the receiver and verifies the renderer responds correctly
    - [ ] Run all tests and ensure no regressions
    - [ ] Verify code coverage meets requirements (>90%)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration and Final Verification' (Protocol in workflow.md)
