# Bug Log (Active Only)

Last verified: 2026-02-23 (latest sweep)

Conventions:
- Severity: `P1` high, `P2` medium, `P3` low.
- Keep IDs stable.
- This file contains only currently active bugs.

| ID | Severity | Area | Summary | Evidence |
|---|---|---|---|---|
| BUG-2026-02-22-001 | P1 | control | Executor preempts active action without honoring `cancel_current`, causing unintended primitive interruption. | `pala/control/executor.py:149` |
| BUG-2026-02-22-002 | P1 | behavior/remote parsing | `_coerce_text()` inserts newlines when joining content chunks; this can corrupt JSON and cause parser failures. | `pala/behavior/model_clients/response_utils.py:75` |
| BUG-2026-02-22-003 | P2 | hardware/camera | GStreamer camera does not validate `appsink` before use; missing sink can crash on `.emit()`. | `pala/hardware/camera_gst.py:44`, `pala/hardware/camera_gst.py:47` |
| BUG-2026-02-22-004 | P2 | perception | Real-camera transient detector misses provide no fallback person/zone signal, causing behavior instability. | `pala/perception/node.py:76` |
| BUG-2026-02-22-005 | P2 | telemetry/capture | Capture `close()` sets `_closed` before all artifact writes; failures can leave partial output and block retry close. | `tools/telemetry/capture.py:224`, `tools/telemetry/capture.py:249` |
| BUG-2026-02-22-008 | P2 | runtime mode wiring | `jetson_perception` mode still builds a dummy frame source path instead of camera-backed perception, so mode behavior does not match its name. | `pala/main.py:320`, `pala/main.py:323` |
| BUG-2026-02-22-009 | P1 | behavior/parser robustness | A single invalid proposal causes the entire planner response to be rejected, even when other proposals are valid; this amplifies parse-fail streaks unnecessarily. | `pala/behavior/intent_proposer.py:120`, `pala/behavior/intent_proposer.py:123` |
| BUG-2026-02-22-010 | P3 | behavior telemetry | Request-start logs always label `response_format` as `json_schema`, even when provider-specific payload uses another mode (for example Gemini `json_object`). | `pala/behavior/policy.py:376`, `pala/behavior/policy.py:430` |
