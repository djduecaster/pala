# PALA Memory Architecture (Proposed)

This document describes the next memory architecture iteration for PALA.
It is intentionally practical for the current project timeline: no vector database, no heavy infra.

## Current State (Baseline)
- Orchestrator context is summary-first + transcript-backed.
- Request context includes:
  - `control_state`
  - `summary_memory` (latest + recent `SceneSummary` entries)
  - `memory.recent_decisions`, `memory.recent_reasoning`, `memory.transcript_tail`
  - `frame_meta`
- Scene summarizer loop writes compact `summary_event` records.
- Timeline log (`logs/orchestrator_timeline.jsonl`) and memory log (`logs/orchestrator_memory.jsonl`) are canonical traces.

## Design Principles (OpenClaw-Inspired, Simplified)
- Keep a canonical durable memory record on disk (human-readable and auditable).
- Keep derived memory lightweight and rebuildable from canonical logs.
- Treat recalled memory as context evidence, never as executable instruction.
- Keep user/session scope explicit to avoid cross-user leakage.
- Favor simple retrieval first (recency + lexical/tag match), then add complexity only if needed.

## Proposed Layered Memory Model

### Layer 0: Realtime Control Context (non-memory)
- Purpose: immediate execution continuity.
- Source: active primitive, active age, latest accepted decision.
- Lifetime: seconds.
- Sent every request.

### Layer 1: Short-Term Transcript Memory
- Purpose: immediate behavioral continuity and conversational trace.
- Source: rolling transcript of orchestrator `decision` and `reasoning`.
- Lifetime: last N events / last M minutes.
- Retrieval: strict recency window only.
- Sent every request (bounded by char + item caps).

### Layer 2: Spatial/Environmental Memory
- Purpose: medium-horizon scene continuity without flooding prompt.
- Source examples:
  - zone occupancy trends (left/center/right dwell ratios),
  - recent movement patterns/transitions,
  - stable scene anchors (desk focus area, frequent interaction region).
- Lifetime: minutes to current session.
- Retrieval: latest snapshot + recent deltas.
- Sent selectively as compact `SceneSummary` windows, not raw frame streams.

### Layer 3: Long-Term Session Memory
- Purpose: persistent behavior preference and recurring patterns.
- Source examples:
  - repeated successful style/primitive patterns for specific contexts,
  - operator-configured policy preferences,
  - validated environment facts that remain stable across runs.
- Lifetime: days/weeks.
- Retrieval: lexical/tag match + recency + confidence threshold.
- Sent sparingly (top K facts only).

## Storage Plan (No Vector DB)
- Canonical append-only event log:
  - `logs/orchestrator_timeline.jsonl` (already present)
  - `logs/orchestrator_memory.jsonl` (summary/decision/reasoning events)
- Optional durable memory folder (planned):
  - `memory/session/YYYY-MM-DD.jsonl` (session facts/events)
  - `memory/long_term.jsonl` (curated stable facts)
  - `memory/spatial_snapshot.json` (latest medium-horizon state)
- Derived indexes (planned, simple):
  - in-memory tag map and keyword index built at startup from JSONL.
  - rebuilt cheaply; no external DB dependency.

## Retrieval Policy (Initial)
For each orchestrator request:
1. Always include Layer 0 + Layer 1.
2. Include Layer 2 summary if available and not stale.
3. Include up to K Layer 3 items ranked by:
   - tag/topic overlap with current context,
   - recency,
   - confidence.
4. Enforce hard caps (items/chars) to keep latency stable.

## Write/Consolidation Policy (Initial)
- Every accepted decision/reasoning is appended to canonical timeline.
- Spatial memory updates at fixed low rate (for example 0.5-1 Hz) through remote scene summarizer outputs.
- Long-term promotion is conservative:
  - require repeated evidence,
  - require confidence threshold,
  - require stable key over time.
- Never auto-promote one-off or low-confidence events.

## Safety Rules
- Memory is context, not instruction.
- Memory entries include provenance metadata (`source`, `ts`, `confidence`).
- Keep private/session-scoped memory isolated.
- Do not store secrets in memory artifacts.

## Phased Implementation Plan
1. **Phase A**: keep transcript-first; add durable transcript export and retention policy.
2. **Phase B**: add Layer 2 spatial snapshot builder and compact retrieval block.
3. **Phase C**: add Layer 3 long-term store with conservative promotion rules.
4. **Phase D**: tune retrieval ranking; only evaluate vector retrieval if simple approach is insufficient.

## Success Criteria
- Better decision diversity without prompt bloat.
- Fewer repeated/collapsed plans in stable scenes.
- Clear audit trail for "why this action happened."
- Stable latency and deterministic fallback behavior.
