# Specification: Interactive Camera Rendering

## Overview
Enhance the `Renderer::render` function to allow interactive camera movement during the rendering process. The function will now be able to listen for updated camera configurations and restart the rendering from the first sample whenever an update is received. Additionally, an "idle" mode will allow the renderer to remain active after finishing all samples, waiting for further camera updates.

## Functional Requirements
- **Camera Updates:** The `render` function will accept a `Receiver<CameraConfig>` to listen for updates.
- **Render Restart:** Upon receiving a new `CameraConfig`, the current rendering process must reset to sample 0.
- **Buffer Update:** The `Renderer` will implement a method to update the GPU `camera_buffer` with the new configuration.
- **Idle Mode:** A new boolean parameter `idle` will control whether the `render` function terminates after reaching the maximum samples or enters an idle state.
- **Idle Behavior:** In idle mode, the function will poll for camera updates with a 10ms sleep interval to avoid high CPU usage.
- **Abort Signal:** The function must still respect the `abort` receiver in all states (rendering or idling).

## Non-Functional Requirements
- **Performance:** Polling in idle mode should have minimal CPU impact.
- **Responsiveness:** Camera updates should be processed promptly between sample iterations.

## Acceptance Criteria
- `Renderer::render` signature is updated to include `camera_config_receiver: &Receiver<CameraConfig>` and `idle: bool`.
- Rendering restarts from sample 0 when a new camera configuration is received.
- The GPU `camera_buffer` is correctly updated on camera changes.
- If `idle` is true, the function continues to run after `samples_per_pixel` is reached, responding to new camera updates.
- If `idle` is true, the function only returns when an abort signal is received.

## Out of Scope
- Dynamic scene updates other than the camera.
- Changes to the `PostProcessor` during the idle phase.
