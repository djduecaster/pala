from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BehaviorIntent:
    mode: str  # idle|engage|track|assist|reacquire
    target_zone: Optional[str] = None
    confidence: float = 0.5
    reason: str = ""
    allow_interrupt: bool = False
    urgency: str = "low"
    act_now: bool = True
    remote_confidence: float = 0.0
