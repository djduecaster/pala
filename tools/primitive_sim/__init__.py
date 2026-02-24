"""Primitive simulation sidecar tooling."""

from .simulate import (
    SimSegment,
    build_suite_segments,
    load_segments_from_json,
    simulate_segments,
    write_trace_json,
)

__all__ = [
    "SimSegment",
    "build_suite_segments",
    "load_segments_from_json",
    "simulate_segments",
    "write_trace_json",
]
