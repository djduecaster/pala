# PALA Telemetry Design Plan

Last updated: 2026-02-07

## Objective
Design a Mac-run telemetry experience that streams live data from Jetson for debugging and demos:
- Live video (reduced FPS/resolution acceptable)
- Overlays from detector and runtime decision stream (TODO: updated schema after reset)
- Important runtime variables and live logs
- Sidecar-style implementation, separate from core control safety logic

## Scope And Guardrails
- Keep telemetry optional and failure-isolated from the 4-loop runtime.
- Prefer tooling under `tools/` with minimal changes to `pala/`.
- Do not make core control/hardware behavior depend on telemetry.
- Keep existing JSONL logs as source of truth for reproducibility.

## Required Telemetry Signals
- Video frames: latest camera frame, target 5 to 15 FPS for viewer.
- Perception: bbox/confidence, `fps`, `latency_ms`, detector health.
- Behavior/planner: TODO after architecture reset.
- Control/hardware: command age, deadman state, enable/disable state.
- System: Jetson `tegrastats` summary (CPU/GPU/memory/temps/power).
- Logs: filtered warnings/errors from runtime and DeepStream/GStreamer.

## Proposal A: SSH Sidecar Stream (Recommended)
### Architecture
- Jetson: run `tools/telemetry/jetson_agent.py`.
- Mac: run `tools/telemetry/mac_viewer.py` which starts Jetson agent via `ssh`.
- Transport: single SSH stdout stream with length-prefixed messages.
- Video: JPEG frames at reduced rate.
- Metadata: NDJSON events (perception/action/hardware/system/log).

### Data Flow
1. Jetson agent captures/receives frame samples.
2. Jetson agent tails JSONL logs and polls `tegrastats`.
3. Agent emits multiplexed stream:
   - `frame` messages (JPEG payload + timestamp)
   - `event` messages (JSON metadata payload)
4. Mac viewer decodes and overlays by nearest timestamp.

### Pros
- No open network ports or firewall setup.
- Works with existing SSH workflow and trust model.
- Strong separation from main runtime.
- Simple to package and run from Mac.

### Cons
- Single-client oriented by default.
- Custom framing protocol to maintain.
- Reconnect/resume behavior must be implemented.

### Risk Notes
- SSH bandwidth spikes if frame size/FPS is too high.
- Clock alignment and buffering need explicit handling.

## Proposal B: WebRTC Telemetry Service
### Architecture
- Jetson sidecar publishes:
  - Video track (H.264 or VP8)
  - Data channel (JSON metadata/events)
- Mac consumes in browser UI.
- Signaling service local to Mac or Jetson.

### Pros
- Best interactive UX and low-latency video path.
- Easy multi-viewer support.
- Browser-native rendering and overlays.

### Cons
- Highest implementation complexity.
- Signaling and ICE/STUN setup overhead.
- Larger dependency and operational surface area.

### Risk Notes
- More moving parts than needed for initial debug tooling.
- Harder to keep deterministic in constrained Jetson conditions.

## Proposal C: RTP/SRT Video + Separate Metadata Stream
### Architecture
- Jetson sidecar publishes video using GStreamer RTP or SRT.
- Metadata/logs sent via lightweight TCP/WebSocket/ZeroMQ.
- Mac viewer merges streams by timestamp.

### Pros
- Mature, high-performance video transport.
- Can scale beyond one consumer.
- Good path for robust demo streaming.

### Cons
- Two-channel synchronization complexity.
- More setup burden than SSH sidecar.
- Requires extra ops discipline around ports/processes.

### Risk Notes
- Timestamp drift/jitter can cause overlay mismatch.
- Network and process lifecycle complexity for local development.

## Decision Matrix
| Criterion | Proposal A (SSH Sidecar) | Proposal B (WebRTC) | Proposal C (RTP/SRT + Metadata) |
|---|---|---|---|
| Build Speed | High | Low | Medium |
| Operational Simplicity | High | Low | Medium |
| Multi-Viewer | Low | High | Medium |
| Latency Potential | Medium | High | High |
| Failure Isolation | High | Medium | Medium |
| Fit For Current Repo | High | Medium | Medium |

## Recommended Path
Start with Proposal A, then graduate only if needed:
1. Proposal A for immediate debug/demos with minimal disruption.
2. Add recording/replay hooks once message schema stabilizes.
3. Re-evaluate Proposal B for competition-grade demo polish.

## Proposed Tooling Layout
- `tools/telemetry/README.md`
- `tools/telemetry/jetson_agent.py`
- `tools/telemetry/mac_viewer.py`
- `tools/telemetry/protocol.py`
- `tools/telemetry/overlay.py`
- `tools/telemetry/sources.py`
- `tools/telemetry/sinks.py`

## Minimal Message Schema (Draft)
```json
{
  "type": "event",
  "ts_mono_s": 123.456,
  "source": "perception",
  "payload": {
    "num_detections": 1,
    "used_fallback_bbox": false,
    "primary_person": {"cx": 0.52, "cy": 0.48, "w": 0.21, "h": 0.41},
    "primary_person_conf": 0.87
  }
}
```

```json
{
  "type": "frame",
  "ts_mono_s": 123.460,
  "codec": "jpeg",
  "width": 640,
  "height": 360,
  "bytes_b64": "..."
}
```

## Phased Rollout Plan
### Phase 0: Text Dashboard (No Video)
- Stream perception/action/hardware/log events to a Mac TUI.
- Include filtered DeepStream/GStreamer errors and `tegrastats`.

### Phase 1: Video + Basic Overlays
- Add low-FPS frame stream.
- Overlay primary bbox, primitive, confidence, deadman status.

### Phase 2: Reliability And Recordability
- Add reconnect/backoff and dropped-frame counters.
- Optional session recording (`events.jsonl` + sampled frames).

### Phase 3: Demo Polish
- Improve UI layout, panel toggles, and latency indicators.
- Optional migration path toward WebRTC if needed.

## Acceptance Criteria
- Launch from Mac with one command and no manual Jetson edits.
- Viewer receives live updates within 500 ms median event delay.
- Video remains usable at 5 FPS under nominal Jetson load.
- Telemetry failure does not stop `uv run python -m pala.main`.
- Core runtime still supports log-only mode with no telemetry tooling.

## Open Questions
- Should frames come from camera sidecar capture or from runtime-exported taps?
- Should telemetry schema be versioned from day one?
- Do we need remote access beyond local network/SSH scenarios?
- Is Cosmos explanation text always available or optional at first?
