from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .trace_graph import TraceRecord


def build_trace_story(trace: Optional[TraceRecord], *, max_events: int = 12) -> List[str]:
    if trace is None:
        return ["No trace selected."]
    lines: List[str] = []
    lines.append(
        f"Trace {trace.trace_id} | req={trace.req_id if trace.req_id is not None else '-'} "
        f"status={trace.status} severity={trace.severity}"
    )
    if trace.duration_ms is not None:
        lines.append(f"Duration: {trace.duration_ms:.0f}ms")
    if trace.summary:
        lines.append(f"Summary: {trace.summary}")
    refs = list(trace.event_refs)[-max(1, int(max_events)) :]
    if not refs:
        lines.append("No events.")
        return lines
    lines.append("Timeline:")
    for idx, ref in enumerate(refs, start=1):
        ts = f"{ref.ts_wall_s:.3f}" if ref.ts_wall_s is not None else "n/a"
        latency = f"{ref.latency_ms:.0f}ms" if ref.latency_ms is not None else "-"
        lines.append(
            f"{idx}. [{ts}] {ref.source} phase={ref.phase or '-'} status={ref.status or '-'} "
            f"sev={ref.severity} lat={latency}"
        )
        if ref.summary:
            lines.append(f"   {ref.summary}")
    return lines


def build_reasoning_story(rows: Sequence[Dict[str, Any]], *, max_rows: int = 10) -> List[str]:
    if not rows:
        return ["No reasoning rows."]
    out: List[str] = []
    for idx, row in enumerate(rows[: max(1, int(max_rows))], start=1):
        req = row.get("req_id")
        phase = row.get("phase")
        status = row.get("status")
        severity = row.get("severity")
        summary = row.get("summary")
        out.append(f"{idx}. req={req if req is not None else '-'} phase={phase or '-'} status={status or '-'} sev={severity or '-'}")
        if summary:
            out.append(f"   {summary}")
    return out
