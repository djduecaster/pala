# Legacy → New Structure Porting Plan

## Mode-Aligned Milestones (Planned)
1. **dev**: all dummy backends; focus on behavior/planning quality and evaluation hooks.
2. **jetson_perception**: on hold; needs careful design to avoid confusing closed-loop behavior.
3. **jetson_full**: port control + servo backend; enable real motion with safety checks (near-term focus).

## Legacy Mapping
- `../pala_old/pala_project/src/hardware/camera.py` → `pala/hardware/camera_gst.py`
- `../pala_old/pala_project/src/hardware/servos.py` → `pala/hardware/servo_pca9685.py`
- `../pala_old/pala_project/src/vision/*` → `pala/perception/*`
  - `capture.py` → `frame_source_gst.py`
  - `process.py` → `preprocess.py`
  - `detector_trt10.py` → `detector_trt10.py`
  - `pose_trt10.py` → `pose_trt10.py`
  - `pose_head.py` → `pose_head.py`
  - `perception.py` → `node.py`
- `../pala_old/pala_project/src/kinematics/*` → `pala/control/*`
  - `state.py` → `state.py`
  - `control.py` → `executor.py` (plus primitives)

## Migration Order (Recommended)
1. **Hardware**: port camera + servos into Jetson-specific modules.
2. **Perception**: port preprocess + detector/pose; keep dummy fallbacks.
3. **Control**: port kinematics controller and primitives; wire into executor.
4. **Behavior**: replace heuristic policy with Cosmos planner integration.

## Guardrails
- Keep dummy implementations as defaults for Mac.
- Preserve clean data contracts (`PerceptionState`, `ActionPlan`, `HardwareCommand`).
- Add tests or scripts per module before integrating into the runtime.
