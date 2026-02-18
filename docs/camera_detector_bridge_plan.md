# Camera->Detector Bridge Plan (Hackathon Scope, No Implementation Yet)

## Goal
- Remove avoidable camera frame round-trips between Python and DeepStream in `jetson_full` mode.
- Keep the current runtime contracts unchanged: `PerceptionState -> ActionPlan -> HardwareCommand`.
- Preserve Mac safety/default behavior (dummy backends in `dev` mode).

## Why This Target
- Current path does extra Python-mediated copying and synchronization:
  - Camera capture to NumPy in `pala/hardware/camera_gst.py`
  - NumPy back into DeepStream `appsrc` in `pala/perception/detector/deepstream_backend.py`
- For hackathon value, this is the one performance rewrite area most likely to matter.

## Non-Negotiables
- Do not change infra scripts (`deploy_jetson.sh`, `run_jetson.sh`, `Makefile`, SSH/rsync setup).
- Keep 4-loop model and loop ownership in `pala/main.py`.
- Keep data contracts stable.
- Core must still run on Mac without Jetson hardware.

## Go/No-Go Gate (Do This First)
Proceed only if baseline on Jetson misses demo targets.

1. Collect baseline:
   - `uv run python tools/test_camera_fps.py --mode jetson_full --seconds 30 --threaded`
   - `uv run python tools/test_detector_stats.py --mode jetson_full --detector deepstream --seconds 30`
2. Record:
   - Camera effective FPS and frame-age stats
   - Detector FPS and timeout/warning rate
   - CPU/GPU headroom (`tegrastats`)
3. Go:
   - Detector path is unstable or below demo target.
4. No-go:
   - Current path is already stable at target FPS/latency; spend time on behavior polish/reliability instead.

## Hackathon-Safe Architecture (Minimal Change)
1. Keep Python orchestration and contracts.
2. Replace `appsink -> NumPy -> appsrc` bridge with one direct GStreamer/DeepStream path:
   - `v4l2src` (or `nvarguscamerasrc`) -> conversion -> `nvstreammux` -> `nvinfer`
3. Extract detections in one place and return `List[Detection]` to existing `PerceptionNode`.
4. Keep `detector: deepstream` as fallback; add explicit opt-in for new path (example: `deepstream_direct`).

## Execution Plan
1. Baseline + decision
   - Run the gate commands above and capture numbers in a short note.
2. Design freeze (1 pass)
   - Define the exact Jetson-only detector backend interface and config key.
   - Keep output format identical to current `Detection`.
3. Implement minimal direct pipeline backend
   - New backend module under `pala/perception/detector/`.
   - Initialize pipeline once; avoid frame copies through Python.
   - Surface health/errors into existing debug fields.
4. Integration with runtime
   - Wire backend selection in `_build_detector` only.
   - No loop topology changes.
5. Safety and fallback
   - On pipeline error, fail closed to no detections + clear health flags.
   - Preserve deadman and existing behavior fallback semantics.
6. Validation
   - Re-run camera/detector tools and compare to baseline.
   - Run `uv run python -m pala.main` in `dev` mode to confirm Mac path unaffected.
7. Demo hardening
   - 10-15 minute soak run with logs.
   - Document launch flags and known limitations.

## Acceptance Criteria
- `dev` mode unchanged on Mac.
- `jetson_full` uses direct bridge path when configured.
- Measurable improvement in at least one:
  - Detector FPS
  - End-to-end frame age/jitter
  - CPU usage in perception path
- No regressions in control/hardware safety behavior.

## Time Budget (Hackathon Reality)
- Baseline + decision: 0.5 day
- Direct pipeline backend + integration: 1 day
- Validation + soak + fallback polish: 0.5 day
- Total: ~2 days max. If not stable by then, revert to current backend and ship.

## Risks
- DeepStream/GStreamer debugging can consume time unexpectedly.
- Camera source differences (`v4l2src` vs `nvarguscamerasrc`) may affect stability.
- Python binding behavior (`gi`, `pyds`) can vary by Jetson image.

## Rollback Strategy
- Keep current `deepstream` backend untouched.
- Guard new path behind explicit detector config switch.
- If issues appear, switch config back to existing backend for demo reliability.
