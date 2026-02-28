# Bug Log (Active Only)

Last verified: 2026-02-28 (full repo sweep)

Conventions:
- Severity: `P1` high, `P2` medium, `P3` low.
- Keep IDs stable.
- This file contains only currently active bugs.

| ID | Severity | Area | Summary | Evidence |
|---|---|---|---|---|
| BUG-2026-02-22-001 | P2 | control | Executor preempts active action without honoring `cancel_current`, causing unintended primitive interruption. | `pala/control/executor.py:149` |
| BUG-2026-02-22-002 | P2 | behavior/remote parsing | `_coerce_text()` inserts newlines when joining content chunks; this can corrupt JSON and cause parser failures. | `pala/behavior/model_clients/response_utils.py:78` |
| BUG-2026-02-22-004 | P2 | perception | Real-camera transient detector misses provide no fallback person/zone signal, causing behavior instability. | `pala/perception/node.py:76` |
| BUG-2026-02-22-008 | P2 | runtime mode wiring | `jetson_perception` mode still builds a dummy frame source path instead of camera-backed perception, so mode behavior does not match its name. | `pala/main.py:341`, `pala/main.py:343` |
| BUG-2026-02-28-022 | P2 | config/control limits | Config loader accepts reversed joint limits (`min > max`), and control clamp then pins outputs to the lower-expression bound, producing deterministic wrong motion. | `pala/config/load.py:239`, `pala/control/executor.py:313` |
| BUG-2026-02-28-023 | P2 | behavior/control integration | Startup `move_to` targets are hard-coded to 5 joints; non-5-joint configurations produce `move_to target length mismatch` rejections in executor. | `pala/behavior/policy.py:924`, `pala/control/executor.py:182` |
| BUG-2026-02-22-005 | P2 | telemetry/capture | Capture `close()` sets `_closed` before all artifact writes; failures can leave partial output and block retry close. | `tools/telemetry/capture.py:224`, `tools/telemetry/capture.py:249` |
| BUG-2026-02-22-013 | P2 | telemetry/run-report | `--json` returns before strict-mode enforcement, so `--strict --json` exits `0` even when alerts exist, breaking CI/automation expectations. | `tools/telemetry/run_report.py:546`, `tools/telemetry/run_report.py:548` |
| BUG-2026-02-22-014 | P2 | telemetry/doctor | Workspace checks are anchored to `os.getcwd()` instead of repository root; running from non-root directories can produce false missing-path failures. | `tools/telemetry/doctor.py:375` |
| BUG-2026-02-28-024 | P1 | telemetry/case selection | Timeline case `row_id` can collide (`first_event_index=0` falls back to ordinal), so selection/review hotkeys can act on the wrong case. | `tools/telemetry/mac_viewer.py:1208`, `tools/telemetry/mac_viewer.py:476`, `tools/telemetry/mac_viewer.py:2316` |
| BUG-2026-02-28-025 | P2 | telemetry/case detail refresh | Case detail retries are blocked after a transient query failure because refresh gating requires empty `case_detail_note`; stale detail persists until selection changes. | `tools/telemetry/mac_viewer.py:2516`, `tools/telemetry/mac_viewer.py:2529` |
| BUG-2026-02-28-026 | P3 | telemetry/trace extraction | Trace extraction uses `a or b` for latency floats, so valid `0.0` latencies are treated as false and replaced/dropped. | `tools/telemetry/trace_graph.py:178`, `tools/telemetry/trace_graph.py:233`, `tools/telemetry/trace_graph.py:253`, `tools/telemetry/trace_graph.py:273`, `tools/telemetry/trace_graph.py:299` |
| BUG-2026-02-22-011 | P3 | tools/primitive-sim | Default `move_to` target generation biases/clamps toward lower bounds for positive-only joint ranges, producing unrealistic baseline targets. | `tools/primitive_sim/run.py:340`, `tools/primitive_sim/run.py:341` |
| BUG-2026-02-27-017 | P2 | tools/primitive-sim | Studio `Save All` posts `{primitives: ...}` without required baseline version; backend normalization raises and `/api/baseline` fails with a server error instead of saving. | `tools/primitive_sim/web/app.js:1350`, `tools/primitive_sim/run.py:438`, `tools/primitive_sim/run.py:1317` |

Notes:
- `BUG-2026-02-28-021` moved out of active bugs: env-only mode/zone signals are currently an intentional hardening choice while remote env+planner paths are being tuned.
- `BUG-2026-02-27-019` removed from active bugs: embedded-shell navigation now promotes mode links to `window.top`.
- Removed after fix: `BUG-2026-02-22-003`, `BUG-2026-02-22-009`, `BUG-2026-02-27-018`, `BUG-2026-02-28-020`.
