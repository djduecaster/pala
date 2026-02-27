# Bug Log (Active Only)

Last verified: 2026-02-27 (repeat sweep 3)

Conventions:
- Severity: `P1` high, `P2` medium, `P3` low.
- Keep IDs stable.
- This file contains only currently active bugs.

| ID | Severity | Area | Summary | Evidence |
|---|---|---|---|---|
| BUG-2026-02-22-001 | P1 | control | Executor preempts active action without honoring `cancel_current`, causing unintended primitive interruption. | `pala/control/executor.py:149` |
| BUG-2026-02-22-002 | P1 | behavior/remote parsing | `_coerce_text()` inserts newlines when joining content chunks; this can corrupt JSON and cause parser failures. | `pala/behavior/model_clients/response_utils.py:78` |
| BUG-2026-02-22-003 | P2 | hardware/camera | GStreamer camera does not validate `appsink` before use; missing sink can crash on `.emit()`. | `pala/hardware/camera_gst.py:44`, `pala/hardware/camera_gst.py:47` |
| BUG-2026-02-22-004 | P2 | perception | Real-camera transient detector misses provide no fallback person/zone signal, causing behavior instability. | `pala/perception/node.py:76` |
| BUG-2026-02-22-005 | P2 | telemetry/capture | Capture `close()` sets `_closed` before all artifact writes; failures can leave partial output and block retry close. | `tools/telemetry/capture.py:224`, `tools/telemetry/capture.py:249` |
| BUG-2026-02-22-008 | P2 | runtime mode wiring | `jetson_perception` mode still builds a dummy frame source path instead of camera-backed perception, so mode behavior does not match its name. | `pala/main.py:320`, `pala/main.py:323` |
| BUG-2026-02-22-009 | P1 | behavior/parser robustness | A single malformed proposal causes the entire planner response to be rejected (schema-first fail), even when other proposals are valid; this amplifies parse-fail streaks unnecessarily. | `pala/behavior/intent_proposer.py:100`, `pala/behavior/intent_proposer.py:103` |
| BUG-2026-02-22-011 | P3 | tools/primitive-sim | Default `move_to` target generation biases/clamps toward lower bounds for positive-only joint ranges, producing unrealistic baseline targets. | `tools/primitive_sim/run.py:229`, `tools/primitive_sim/run.py:230` |
| BUG-2026-02-27-017 | P2 | tools/primitive-sim | Studio `Save All` posts `{primitives: ...}` without required baseline version; backend normalization raises and `/api/baseline` fails with a server error instead of saving. | `tools/primitive_sim/web/app.js:1346`, `tools/primitive_sim/run.py:433`, `tools/primitive_sim/run.py:1043` |
| BUG-2026-02-22-013 | P2 | telemetry/run-report | `--json` returns before strict-mode enforcement, so `--strict --json` exits `0` even when alerts exist, breaking CI/automation expectations. | `tools/telemetry/run_report.py:350` |
| BUG-2026-02-22-014 | P2 | telemetry/doctor | Workspace checks are anchored to `os.getcwd()` instead of repository root; running from non-root directories can produce false missing-path failures. | `tools/telemetry/doctor.py:313` |
| BUG-2026-02-22-015 | P3 | telemetry/query fallback | In-memory query fallback tokenizes via plain `.split()` while indexed/sqlite query parsing uses shell-like tokenization, causing inconsistent results for quoted queries. | `tools/telemetry/mac_viewer.py:1109` |
