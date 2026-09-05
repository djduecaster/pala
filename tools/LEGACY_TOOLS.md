# Legacy Tool Notes

Last updated: 2026-09-05

## `tools/cosmos_api_test_tool.py` (removed)
- Status: removed.
- Reason: original implementation depended on removed behavior V3 modules (`env_summarizer`, `intent_proposer`, V3 schemas).
- Replacements:
  - `tools/model_provider_probe.py` for generic endpoint/provider compatibility checks.

## Removed planner diagnostic
- `tools/cosmos_planner_smoke.sh` was removed with the planner reset; it targeted the deleted runtime planner path.

## Removed V4 tooling
- `tools/primitive_sim/state_machine.py`, `tools/probe_web/`, and `tools/ft_capture*/` were retired during Session 1 cleanup because they depended on the deleted V4 behavior runtime.
- Their tests and V4 scenario catalog were removed with them; the current runtime is capture-only and hold-only.
- Primitive tuning/playback remains under `tools/primitive_sim/` and uses current typed control primitives.
