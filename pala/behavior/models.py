from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SceneObservation:
    ts_mono_s: float
    person_present: bool
    person_conf: Optional[float]
    zone: Optional[str]
    zone_changed: bool
    zone_dwell_s: float
    activity_hint: Optional[str]
    event: str

