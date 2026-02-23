from __future__ import annotations

from dataclasses import dataclass

from .types import ProposerResponse


@dataclass
class ComponentHealth:
    state: str = "HEALTHY"
    transport_fail_streak: int = 0
    parse_fail_streak: int = 0
    no_signal_streak: int = 0
    slow_streak: int = 0
    last_latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "transport_fail_streak": self.transport_fail_streak,
            "parse_fail_streak": self.parse_fail_streak,
            "no_signal_streak": self.no_signal_streak,
            "slow_streak": self.slow_streak,
            "last_latency_ms": round(float(self.last_latency_ms), 1),
        }


@dataclass
class PerceptionHealth:
    state: str = "HEALTHY"
    detector_fail_streak: int = 0
    source_fail_streak: int = 0
    stale_frame_streak: int = 0

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "detector_fail_streak": self.detector_fail_streak,
            "source_fail_streak": self.source_fail_streak,
            "stale_frame_streak": self.stale_frame_streak,
        }


class HealthManager:
    """Tracks remote quality and applies circuit-breaker style states."""

    def __init__(self) -> None:
        self.env = ComponentHealth()
        self.planner = ComponentHealth()
        self.perception = PerceptionHealth()
        self._planner_slow_latency_ms = 2500.0
        self._env_slow_latency_ms = 6000.0

    def on_env_result(self, *, status: str, latency_ms: float) -> None:
        health = self.env
        health.last_latency_ms = max(0.0, float(latency_ms))
        if health.last_latency_ms >= self._env_slow_latency_ms:
            health.slow_streak += 1
        else:
            health.slow_streak = 0
        if status == "ok":
            health.transport_fail_streak = 0
            health.parse_fail_streak = 0
            health.no_signal_streak = 0
            health.state = "DEGRADED" if health.slow_streak >= 3 else "HEALTHY"
            return

        if status == "transport_error":
            health.transport_fail_streak += 1
        else:
            health.parse_fail_streak += 1

        if health.transport_fail_streak >= 3 or health.parse_fail_streak >= 3:
            health.state = "OPEN_BREAKER"
        else:
            health.state = "DEGRADED"

    def on_planner_result(self, *, status: str, latency_ms: float, response: ProposerResponse | None) -> None:
        health = self.planner
        health.last_latency_ms = max(0.0, float(latency_ms))
        if health.last_latency_ms >= self._planner_slow_latency_ms:
            health.slow_streak += 1
        else:
            health.slow_streak = 0

        if status == "ok":
            health.transport_fail_streak = 0
            health.parse_fail_streak = 0
            if self._is_no_signal(response):
                health.no_signal_streak += 1
            else:
                health.no_signal_streak = 0
        elif status == "transport_error":
            health.transport_fail_streak += 1
        else:
            health.parse_fail_streak += 1

        if health.transport_fail_streak >= 3 or health.parse_fail_streak >= 3:
            health.state = "OPEN_BREAKER"
        elif health.slow_streak >= 8:
            health.state = "OPEN_BREAKER"
        elif health.no_signal_streak >= 6:
            health.state = "DEGRADED"
        elif health.slow_streak >= 3:
            health.state = "DEGRADED"
        else:
            health.state = "HEALTHY"

    def planner_effective_hz(self, base_hz: float) -> float:
        hz = max(0.05, float(base_hz))
        if self.planner.state == "OPEN_BREAKER":
            return min(hz, 0.15)
        if self.planner.state == "DEGRADED":
            return min(hz, 0.35)
        return hz

    def planner_open_breaker(self) -> bool:
        return self.planner.state == "OPEN_BREAKER"

    def planner_no_signal_streak(self) -> int:
        return int(self.planner.no_signal_streak)

    def on_perception_result(
        self,
        *,
        detector_alive: bool | None,
        source_alive: bool | None,
        stale_frame: bool,
    ) -> None:
        health = self.perception

        if detector_alive is False:
            health.detector_fail_streak += 1
        elif detector_alive is True:
            health.detector_fail_streak = 0

        if source_alive is False:
            health.source_fail_streak += 1
        elif source_alive is True:
            health.source_fail_streak = 0

        if stale_frame:
            health.stale_frame_streak += 1
        else:
            health.stale_frame_streak = 0

        if health.detector_fail_streak >= 3 or health.source_fail_streak >= 3:
            health.state = "OPEN_BREAKER"
        elif health.detector_fail_streak > 0 or health.source_fail_streak > 0 or health.stale_frame_streak >= 4:
            health.state = "DEGRADED"
        else:
            health.state = "HEALTHY"

    def perception_degraded(self) -> bool:
        return self.perception.state in {"DEGRADED", "OPEN_BREAKER"}

    @staticmethod
    def _is_no_signal(response: ProposerResponse | None) -> bool:
        if response is None:
            return True
        if not response.proposals:
            return True
        if all(item.primitive in {"hold", "home"} for item in response.proposals):
            return True
        top = max(response.proposals, key=lambda item: item.score)
        return top.score < 0.15
