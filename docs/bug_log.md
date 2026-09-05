# Bug Log (Active Only)

Cleanup review: 2026-09-05. Resolved/retired entries below were checked against
the cleanup changes. Retained telemetry and legacy diagnostic findings are
backlog from the earlier review, not newly reproduced hardware results.

Conventions:
- Severity: `P1` high, `P2` medium, `P3` low.
- Keep IDs stable.
- This file contains only currently active bugs.

| ID | Severity | Area | Summary | Evidence |
|---|---|---|---|---|
| BUG-2026-02-22-002 | P2 | behavior/remote parsing | `_coerce_text()` inserts newlines when joining content chunks; this can corrupt JSON and cause parser failures. | `pala/behavior/model_clients/response_utils.py:78` |
| BUG-2026-03-01-030 | P2 | control/style profiles | YAML style scales are now validated. Direct programmatic executor construction can still supply non-positive scales; validate that boundary before exposing it to external behavior inputs. | `pala/control/executor.py:271`, `pala/control/executor.py:333` |
| BUG-2026-02-22-005 | P2 | telemetry/capture | Capture `close()` sets `_closed` before all artifact writes; failures can leave partial output and block retry close. | `tools/telemetry/capture.py:224`, `tools/telemetry/capture.py:230` |
| BUG-2026-02-22-013 | P2 | telemetry/run-report | `--json` returns before strict-mode enforcement, so `--strict --json` exits `0` even when alerts exist, breaking CI/automation expectations. | `tools/telemetry/run_report.py:578`, `tools/telemetry/run_report.py:580`, `tools/telemetry/run_report.py:674` |
| BUG-2026-02-22-014 | P2 | telemetry/doctor | Workspace checks are anchored to `os.getcwd()` instead of repository root; running from non-root directories can produce false missing-path failures. | `tools/telemetry/doctor.py:402` |
| BUG-2026-02-28-024 | P1 | telemetry/case selection | Timeline case `row_id` can collide (`first_event_index=0` falls back to ordinal), so selection/review hotkeys can act on the wrong case. | `tools/telemetry/mac_viewer.py:1194`, `tools/telemetry/mac_viewer.py:1195`, `tools/telemetry/mac_viewer.py:2402` |
| BUG-2026-02-28-025 | P2 | telemetry/case detail refresh | Case detail retries are blocked after a transient query failure because refresh gating requires empty `case_detail_note`; stale detail persists until selection changes. | `tools/telemetry/mac_viewer.py:2597`, `tools/telemetry/mac_viewer.py:2599`, `tools/telemetry/mac_viewer.py:2611` |
| BUG-2026-02-28-026 | P3 | telemetry/trace extraction | Trace extraction uses `a or b` for latency floats, so valid `0.0` latencies are treated as false and replaced/dropped. | `tools/telemetry/trace_graph.py:187`, `tools/telemetry/trace_graph.py:242`, `tools/telemetry/trace_graph.py:262`, `tools/telemetry/trace_graph.py:282`, `tools/telemetry/trace_graph.py:321` |
| BUG-2026-03-01-033 | P2 | telemetry/integrity | Integrity verification trusts report file paths and can validate `../` or absolute paths outside the session root. | `tools/telemetry/integrity.py:120`, `tools/telemetry/integrity.py:124` |
| BUG-2026-03-01-038 | P3 | telemetry/run-report | Positional path handling treats telemetry roots as single sessions (no descent), which can emit false `no_run_artifacts` failures. | `tools/telemetry/run_report.py:31`, `tools/telemetry/run_report.py:34` |
| BUG-2026-02-22-011 | P3 | tools/primitive-sim | Default `move_to` target generation biases/clamps toward lower bounds for positive-only joint ranges, producing unrealistic baseline targets. | `tools/primitive_sim/run.py:329`, `tools/primitive_sim/run.py:338` |
| BUG-2026-03-01-039 | P2 | tools/primitive-sim | Output path fields accept absolute/escape paths (`..`), allowing writes outside intended workspace output locations. | `tools/primitive_sim/run.py:191`, `tools/primitive_sim/run.py:1412`, `tools/primitive_sim/run.py:1476`, `tools/primitive_sim/run.py:1569` |

Notes:
- Removed during the Phase 1/2 reset: detector-specific bugs `004`, `008`, `027`, `028`, and `031`; Behavior V4 bugs `042`, `043`, and `044`. The affected runtime paths no longer exist.
- Session 1 removed obsolete probe-web and fine-tuning tools; their findings `034`, `035`, `036`, `037`, `045`, and `046` are retired with those code paths.
- `BUG-2026-02-28-021` moved out of active bugs: env-only mode/zone signals are currently an intentional hardening choice while remote env+planner paths are being tuned.
- `BUG-2026-02-27-019` removed from active bugs: embedded-shell navigation now promotes mode links to `window.top`.
- `BUG-2026-02-28-023` removed from active bugs: V3 startup `move_to` path was removed in Behavior V4 cutover.
- Removed after fix: `BUG-2026-02-22-003`, `BUG-2026-02-22-009`, `BUG-2026-02-27-018`, `BUG-2026-02-28-020`.

Session 1 resolutions:
- `017`: Studio Save All supplies the schema version before validation; confirmed through the browser using a temporary baseline.
- `047`: provider text probe requires the complete trimmed response to equal READY.
- `001`: clarified existing latest-intent interruption semantics; `cancel_current` is not interpreted. Gesture-level interruption policy remains a separate design task.
- `022`, `040`, `041`: configuration rejects reversed/nonfinite limits, nonpositive/nonfinite loop rates, and nonpositive deadman timeouts.
- `029`: servo input is fully normalized and validated before any channel writes. Bus faults during writes are not transactional.
- `032`: simulator uses a private executor clock instead of patching global monotonic time.
- Gesture tool dry-runs no longer construct a PCA9685 backend. The unsupported `--keep-enabled` flag was removed because shutdown already disables outputs.

Before powered gesture work:
- Reconcile software limits with calibration and current tested positions; the numeric calibration was deliberately preserved.
- Establish physical starting posture; the executor initializes from a zero command estimate.
- Verify physical settling and shutdown behavior. The software deadman is not an independent hardware watchdog.
