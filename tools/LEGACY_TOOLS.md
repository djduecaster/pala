# Legacy Tool Notes

Last updated: 2026-03-01

## `tools/cosmos_api_test_tool.py` (removed)
- Status: removed.
- Reason: original implementation depended on removed behavior V3 modules (`env_summarizer`, `intent_proposer`, V3 schemas).
- Replacements:
  - `tools/model_provider_probe.py` for endpoint/provider compatibility and schema probe checks.
  - `tools/probe_web/` for interactive payload/response inspection.

## `tools/primitive_sim/state_machine.py`
- Status: active Behavior V4 simulator.
- Notes:
  - Uses `ModeFsmV4` + `skills_v4` contracts for deterministic mode stepping and allowed primitive sets.
  - Keeps a few legacy signal aliases (`planner_open_breaker`, `perception_degraded`, `activity_level`, `novelty`, `env_delta`) for UI/backward compatibility only.
