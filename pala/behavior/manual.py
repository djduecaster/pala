"""Small manual request gate; plans execute in the existing control loop."""
from __future__ import annotations

import threading

from pala.control.performances import PerformanceLibrary, PerformancePlan, PerformanceStatus


class ManualBehaviorPolicy:
    def __init__(self, library: PerformanceLibrary) -> None:
        self.library = library
        self._lock = threading.Lock()
        self._plan = library.plan("startup")
        self._busy = True
        self._state = "zero"
        self._greeted = False
        self._closing = False
        self._failed = None
        self._finished = False
        self._report = None

    def request(self, name: str) -> tuple[bool, str]:
        with self._lock:
            if name not in {"greet", "attend", "settle", "demo", "shutdown", "notice", "excite"}:
                return False, "Unknown request"
            if name in {"notice", "excite"} and name not in self.library.performances:
                return False, "Performance unavailable in this library"
            if self._closing or self._failed:
                return False, "Shutdown or failure is already in progress"
            if name != "shutdown":
                if self._busy:
                    return False, "Busy; request rejected rather than queued or interrupting"
                if name == "notice" and (self._state != "rest" or self._greeted):
                    return False, "Notice requires unengaged rest"
                if name == "excite" and (self._state != "attention" or not self._greeted):
                    return False, "Excitement requires a completed greeting"
                if name in {"greet", "demo"} and self._greeted:
                    return False, "Already greeted this interaction; settle before greeting again"
                if name == "settle" and self._state == "rest":
                    return False, "Already resting"
                if name == "attend" and self._state == "attention":
                    return False, "Already attentive"
            variant = f"settle_{self._state}"
            plan_name = variant if name == "settle" and variant in self.library.performances else name
            self._plan = self.library.plan(plan_name)
            self._busy = True
            self._report = None
            self._closing = name == "shutdown"
            return True, f"Accepted {name}"

    def step(self, report: PerformanceStatus | None) -> PerformancePlan:
        with self._lock:
            if report is not None and report.plan_id == self._plan.plan_id:
                self._report = report
                if report.status == "failed":
                    self._failed = report.reason or "performance failed"
                elif report.status == "complete" and self._busy:
                    self._busy = False
                    self._state = self._plan.end_state
                    if self._plan.name == "greet":
                        self._greeted = True
                    elif self._plan.name in {"settle", "demo"} or self._plan.name.startswith("settle_"):
                        self._greeted = False
                    if self._closing:
                        self._finished = True
            return self._plan

    def status(self) -> dict[str, object]:
        with self._lock:
            return {"state": self._state, "busy": self._busy, "greeted": self._greeted,
                    "closing": self._closing, "finished": self._finished, "failure": self._failed,
                    "performance": self._plan.name, "plan_id": self._plan.plan_id,
                    "step": self._report.step if self._report else None,
                    "phase": self._report.phase if self._report else None}
