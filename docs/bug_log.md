# Bug Log (Active Only)

Last verified: 2026-08-05 (Phase 1/2 perception and behavior reset)

Conventions:
- Severity: `P1` high, `P2` medium, `P3` low.
- Keep IDs stable.
- This file contains only currently active bugs.

| ID | Severity | Area | Summary | Evidence |
|---|---|---|---|---|
| BUG-2026-02-22-001 | P2 | control | Executor preempts active action without honoring `cancel_current`, causing unintended primitive interruption. | `pala/control/executor.py:149` |
| BUG-2026-02-22-002 | P2 | behavior/remote parsing | `_coerce_text()` inserts newlines when joining content chunks; this can corrupt JSON and cause parser failures. | `pala/behavior/model_clients/response_utils.py:78` |
| BUG-2026-02-28-022 | P2 | config/control limits | Config loader accepts reversed joint limits (`min > max`), and control clamp then pins outputs to the lower-expression bound, producing deterministic wrong motion. | `pala/config/load.py:246`, `pala/control/executor.py:313` |
| BUG-2026-03-01-029 | P2 | hardware/servo | Servo command-length mismatch is not validated; short lists silently skip channels and long lists can raise index errors in the hardware loop. | `pala/hardware/servo_pca9685.py:65` |
| BUG-2026-03-01-030 | P2 | control/style profiles | Non-positive style `rate_scale` values are accepted; this can force zero step size and leave motion primitives running without progress. | `pala/control/executor.py:271`, `pala/control/executor.py:333` |
| BUG-2026-03-01-040 | P2 | config/runtime rates | Loop rates accept zero/negative values; rate limiter coerces them into near-frozen long periods, making loops appear hung. | `pala/config/load.py:230`, `pala/utils/timing.py:8` |
| BUG-2026-03-01-041 | P1 | config/hardware safety | Negative `deadman_timeout_ms` is accepted; hardware deadman check then trips continuously and keeps outputs disabled. | `pala/config/load.py:237`, `pala/main.py:196`, `pala/main.py:203` |
| BUG-2026-02-22-005 | P2 | telemetry/capture | Capture `close()` sets `_closed` before all artifact writes; failures can leave partial output and block retry close. | `tools/telemetry/capture.py:224`, `tools/telemetry/capture.py:230` |
| BUG-2026-02-22-013 | P2 | telemetry/run-report | `--json` returns before strict-mode enforcement, so `--strict --json` exits `0` even when alerts exist, breaking CI/automation expectations. | `tools/telemetry/run_report.py:578`, `tools/telemetry/run_report.py:580`, `tools/telemetry/run_report.py:674` |
| BUG-2026-02-22-014 | P2 | telemetry/doctor | Workspace checks are anchored to `os.getcwd()` instead of repository root; running from non-root directories can produce false missing-path failures. | `tools/telemetry/doctor.py:402` |
| BUG-2026-02-28-024 | P1 | telemetry/case selection | Timeline case `row_id` can collide (`first_event_index=0` falls back to ordinal), so selection/review hotkeys can act on the wrong case. | `tools/telemetry/mac_viewer.py:1194`, `tools/telemetry/mac_viewer.py:1195`, `tools/telemetry/mac_viewer.py:2402` |
| BUG-2026-02-28-025 | P2 | telemetry/case detail refresh | Case detail retries are blocked after a transient query failure because refresh gating requires empty `case_detail_note`; stale detail persists until selection changes. | `tools/telemetry/mac_viewer.py:2597`, `tools/telemetry/mac_viewer.py:2599`, `tools/telemetry/mac_viewer.py:2611` |
| BUG-2026-02-28-026 | P3 | telemetry/trace extraction | Trace extraction uses `a or b` for latency floats, so valid `0.0` latencies are treated as false and replaced/dropped. | `tools/telemetry/trace_graph.py:187`, `tools/telemetry/trace_graph.py:242`, `tools/telemetry/trace_graph.py:262`, `tools/telemetry/trace_graph.py:282`, `tools/telemetry/trace_graph.py:321` |
| BUG-2026-03-01-033 | P2 | telemetry/integrity | Integrity verification trusts report file paths and can validate `../` or absolute paths outside the session root. | `tools/telemetry/integrity.py:120`, `tools/telemetry/integrity.py:124` |
| BUG-2026-03-01-038 | P3 | telemetry/run-report | Positional path handling treats telemetry roots as single sessions (no descent), which can emit false `no_run_artifacts` failures. | `tools/telemetry/run_report.py:31`, `tools/telemetry/run_report.py:34` |
| BUG-2026-02-22-011 | P3 | tools/primitive-sim | Default `move_to` target generation biases/clamps toward lower bounds for positive-only joint ranges, producing unrealistic baseline targets. | `tools/primitive_sim/run.py:329`, `tools/primitive_sim/run.py:338` |
| BUG-2026-02-27-017 | P2 | tools/primitive-sim | Studio `Save All` posts `{primitives: ...}` without required baseline version; backend normalization raises and `/api/baseline` fails with a server error instead of saving. | `tools/primitive_sim/web/app.js:1564`, `tools/primitive_sim/run.py:545`, `tools/primitive_sim/run.py:1392` |
| BUG-2026-03-01-032 | P1 | tools/primitive-sim | Simulation path monkeypatches `time.monotonic` globally; concurrent requests can race and corrupt timing across active simulations. | `tools/primitive_sim/simulate.py:57`, `tools/primitive_sim/run.py:46`, `tools/primitive_sim/web/app.js:1486` |
| BUG-2026-03-01-039 | P2 | tools/primitive-sim | Output path fields accept absolute/escape paths (`..`), allowing writes outside intended workspace output locations. | `tools/primitive_sim/run.py:191`, `tools/primitive_sim/run.py:1412`, `tools/primitive_sim/run.py:1476`, `tools/primitive_sim/run.py:1569` |
| BUG-2026-03-01-034 | P2 | tools/probe-web | Run-detail endpoint does not validate `run_id`; crafted values can trigger path escape semantics against logs storage. | `tools/probe_web/app.py:689`, `tools/probe_web/storage.py:138`, `tools/probe_web/storage.py:141` |
| BUG-2026-03-01-035 | P2 | tools/probe-web | Override payload type-cast failures can raise uncaught exceptions and return server errors instead of user-facing validation errors. | `tools/probe_web/service.py:873`, `pala/behavior/model_clients/response_utils.py:33`, `tools/probe_web/app.py:784` |
| BUG-2026-03-01-036 | P2 | tools/ft-capture | `auto` camera fallback to dummy plus unpaced frame capture can generate runaway frame counts and excessive disk usage. | `tools/ft_capture/capture.py:239`, `tools/ft_capture/capture.py:278`, `pala/hardware/camera.py:21` |
| BUG-2026-03-01-037 | P2 | tools/ft-capture-web | URL building assumes paths are always under dataset root; malformed/symlinked records can raise and break listing render. | `tools/ft_capture_web/service.py:46`, `tools/ft_capture_web/service.py:102` |
| BUG-2026-03-01-045 | P1 | tools/ft-capture | Session directory creation trusts `session_id`; path traversal sequences can escape dataset root and write artifacts outside expected storage. | `tools/ft_capture/__main__.py:78`, `tools/ft_capture/storage.py:72` |
| BUG-2026-03-01-046 | P3 | tools/ft-capture | Generated session IDs use only second precision, so multiple captures started in the same second can collide into one session. | `tools/ft_capture/storage.py:58` |
| BUG-2026-03-01-047 | P3 | tools/model-provider-probe | `text_ping` readiness check matches substring `"READY"` and can false-positive on content like `"NOT READY"` or `"ALREADY"`. | `tools/model_provider_probe.py:349` |

Notes:
- Removed during the Phase 1/2 reset: detector-specific bugs `004`, `008`, `027`, `028`, and `031`; Behavior V4 bugs `042`, `043`, and `044`. The affected runtime paths no longer exist.
- OBE cleanup this sweep: none of the currently active entries were overtaken by events.
- `BUG-2026-02-28-021` moved out of active bugs: env-only mode/zone signals are currently an intentional hardening choice while remote env+planner paths are being tuned.
- `BUG-2026-02-27-019` removed from active bugs: embedded-shell navigation now promotes mode links to `window.top`.
- `BUG-2026-02-28-023` removed from active bugs: V3 startup `move_to` path was removed in Behavior V4 cutover.
- Removed after fix: `BUG-2026-02-22-003`, `BUG-2026-02-22-009`, `BUG-2026-02-27-018`, `BUG-2026-02-28-020`.
