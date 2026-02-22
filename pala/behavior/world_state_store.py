from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import logging
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

DEFAULT_IDENTITY_CORE = (
    "You are PALA, a physical desk companion lamp. "
    "Your purpose is to be helpful, safe, expressive, and socially aware."
)


@dataclass
class EnvironmentSnapshot:
    scene: str
    events: str
    hypotheses: str
    opportunities: str
    uncertainties: str
    summary: str
    delta_score: float
    timestamp_wall_s: float = field(default_factory=time.time)


@dataclass
class DecisionSnapshot:
    primitive: str
    style: str
    confidence: float
    rationale_short: str
    timestamp_wall_s: float = field(default_factory=time.time)


@dataclass
class WorldStateStoreConfig:
    identity_path: str = "memory/identity.md"
    world_state_path: str = "memory/world_state.md"
    session_digest_path: str = "memory/session_digest.md"
    max_events: int = 12
    max_decisions: int = 6
    default_identity_core: str = DEFAULT_IDENTITY_CORE


class WorldStateStore:
    """Compact local behavior memory with markdown persistence."""

    def __init__(self, config: Optional[WorldStateStoreConfig] = None):
        self._cfg = config or WorldStateStoreConfig()
        self._lock = threading.Lock()

        self.identity_core = self._load_identity_core()
        self.latest_env_snapshot: Optional[EnvironmentSnapshot] = None
        self.event_tail: List[Dict[str, Any]] = []
        self.decision_tail: List[DecisionSnapshot] = []
        self.control_state_latest: str = "unknown"
        self.session_digest = self._load_session_digest()
        self.updated_at_wall_s = time.time()

    def update_environment(self, snapshot: EnvironmentSnapshot) -> None:
        with self._lock:
            self.latest_env_snapshot = snapshot
            self.updated_at_wall_s = time.time()
            summary = self._dense_event_summary(snapshot)
            if summary:
                self._append_event_locked(
                    summary,
                    ts_wall_s=snapshot.timestamp_wall_s,
                    source="env_processor",
                )
            self._persist_world_state_locked()

    def append_event(self, text: str, *, ts_wall_s: Optional[float] = None) -> None:
        token = str(text).strip()
        if not token:
            return
        with self._lock:
            self._append_event_locked(token, ts_wall_s=ts_wall_s, source="manual")
            self.updated_at_wall_s = time.time()
            self._persist_world_state_locked()

    def append_decision(self, decision: DecisionSnapshot) -> None:
        with self._lock:
            self.decision_tail.append(decision)
            max_decisions = self._max_decisions()
            if len(self.decision_tail) > max_decisions:
                self.decision_tail = self.decision_tail[-max_decisions:]
            self.updated_at_wall_s = time.time()
            self._persist_world_state_locked()

    def set_control_state(self, control_state: Any) -> None:
        with self._lock:
            formatted = self._format_control_state(control_state)
            if formatted == self.control_state_latest:
                return
            self.control_state_latest = formatted
            self.updated_at_wall_s = time.time()
            self._persist_world_state_locked()

    def rewrite_session_digest(self, digest_text: str) -> None:
        token = str(digest_text).strip()
        with self._lock:
            self.session_digest = token
            self.updated_at_wall_s = time.time()
            self._persist_session_digest_locked()
            self._persist_world_state_locked()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            env = None
            if self.latest_env_snapshot is not None:
                env = {
                    "scene": self.latest_env_snapshot.scene,
                    "events": self.latest_env_snapshot.events,
                    "hypotheses": self.latest_env_snapshot.hypotheses,
                    "opportunities": self.latest_env_snapshot.opportunities,
                    "uncertainties": self.latest_env_snapshot.uncertainties,
                    "summary": self.latest_env_snapshot.summary,
                    "delta_score": self.latest_env_snapshot.delta_score,
                    "timestamp_wall_s": self.latest_env_snapshot.timestamp_wall_s,
                }
            return {
                "identity_core": self.identity_core,
                "latest_env_snapshot": env,
                "event_tail": list(self.event_tail),
                "decision_tail": [
                    {
                        "primitive": d.primitive,
                        "style": d.style,
                        "confidence": d.confidence,
                        "rationale_short": d.rationale_short,
                        "timestamp_wall_s": d.timestamp_wall_s,
                    }
                    for d in self.decision_tail
                ],
                "control_state_latest": self.control_state_latest,
                "session_digest": self.session_digest,
                "updated_at_wall_s": self.updated_at_wall_s,
            }

    def persist(self) -> None:
        with self._lock:
            self._persist_session_digest_locked()
            self._persist_world_state_locked()

    def _append_event_locked(self, text: str, *, ts_wall_s: Optional[float], source: str) -> None:
        ts = float(ts_wall_s if ts_wall_s is not None else time.time())
        self.event_tail.append({"timestamp_wall_s": ts, "summary": text, "source": source})
        max_events = self._max_events()
        if len(self.event_tail) > max_events:
            self.event_tail = self.event_tail[-max_events:]

    def _persist_world_state_locked(self) -> None:
        path = Path(self._cfg.world_state_path)
        self._write_file(path, self._render_world_state_markdown())

    def _persist_session_digest_locked(self) -> None:
        path = Path(self._cfg.session_digest_path)
        self._write_file(path, (self.session_digest.strip() + "\n") if self.session_digest else "")

    def _load_identity_core(self) -> str:
        path = Path(self._cfg.identity_path)
        if not path.exists():
            return self._cfg.default_identity_core
        try:
            text = path.read_text(encoding="utf-8").strip()
            return text if text else self._cfg.default_identity_core
        except OSError as exc:
            logger.warning("world_state_store identity load failed path=%s error=%s", path, exc)
            return self._cfg.default_identity_core

    def _load_session_digest(self) -> str:
        path = Path(self._cfg.session_digest_path)
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("world_state_store session digest load failed path=%s error=%s", path, exc)
            return ""

    def _render_world_state_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# PALA World State")
        lines.append("")
        lines.append(f"Last updated: {self._format_wall_time(self.updated_at_wall_s)}")
        lines.append("")
        lines.append("## Identity Core")
        lines.append("")
        lines.append(self.identity_core.strip() if self.identity_core.strip() else DEFAULT_IDENTITY_CORE)
        lines.append("")

        lines.append("## Latest Environment Snapshot")
        lines.append("")
        if self.latest_env_snapshot is None:
            lines.append("_none_")
        else:
            snap = self.latest_env_snapshot
            lines.append(f"- timestamp: {self._format_wall_time(snap.timestamp_wall_s)}")
            lines.append(f"- delta_score: {snap.delta_score:.2f}")
            lines.append("")
            lines.append("### Scene")
            lines.append(snap.scene.strip() or "_empty_")
            lines.append("")
            lines.append("### Events")
            lines.append(snap.events.strip() or "_empty_")
            lines.append("")
            lines.append("### Hypotheses")
            lines.append(snap.hypotheses.strip() or "_empty_")
            lines.append("")
            lines.append("### Opportunities")
            lines.append(snap.opportunities.strip() or "_empty_")
            lines.append("")
            lines.append("### Uncertainties")
            lines.append(snap.uncertainties.strip() or "_empty_")
            lines.append("")
            lines.append("### Summary")
            lines.append(snap.summary.strip() or "_empty_")
        lines.append("")

        lines.append("## Event Tail")
        lines.append("")
        if not self.event_tail:
            lines.append("_none_")
        else:
            for i, event in enumerate(self.event_tail[-self._max_events() :], start=1):
                ts = float(event.get("timestamp_wall_s", 0.0))
                summary = str(event.get("summary", "")).strip() or "_empty_"
                source = str(event.get("source", "")).strip() or "unknown"
                lines.append(f"{i}. [{self._format_wall_time(ts)}] source={source} {summary}")
        lines.append("")

        lines.append("## Decision Tail")
        lines.append("")
        if not self.decision_tail:
            lines.append("_none_")
        else:
            for i, item in enumerate(self.decision_tail[-self._max_decisions() :], start=1):
                lines.append(
                    f"{i}. [{self._format_wall_time(item.timestamp_wall_s)}] "
                    f"primitive={item.primitive} style={item.style} "
                    f"confidence={item.confidence:.2f} rationale={item.rationale_short}"
                )
        lines.append("")

        lines.append("## Control State")
        lines.append("")
        lines.append(self.control_state_latest.strip() or "_unknown_")
        lines.append("")

        lines.append("## Session Digest")
        lines.append("")
        lines.append(self.session_digest.strip() or "_empty_")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_control_state(control_state: Any) -> str:
        if control_state is None:
            return "unknown"
        if isinstance(control_state, str):
            return control_state
        if isinstance(control_state, Mapping):
            return ", ".join(f"{k}={v}" for k, v in control_state.items())
        active_kind = getattr(control_state, "active_kind", None)
        status = getattr(control_state, "status", None)
        reason = getattr(control_state, "reason", None)
        started = getattr(control_state, "started_monotonic_s", None)
        return f"active_kind={active_kind} status={status} reason={reason} started={started}"

    @staticmethod
    def _format_wall_time(ts_wall_s: float) -> str:
        dt = datetime.fromtimestamp(float(ts_wall_s), tz=timezone.utc)
        return dt.isoformat(timespec="seconds")

    @classmethod
    def _dense_event_summary(cls, snapshot: EnvironmentSnapshot) -> str:
        fields = [
            ("scene", snapshot.scene),
            ("events", snapshot.events),
            ("hypotheses", snapshot.hypotheses),
            ("opportunities", snapshot.opportunities),
            ("uncertainties", snapshot.uncertainties),
            ("summary", snapshot.summary),
        ]
        segments = []
        for key, value in fields:
            token = cls._compact_text(value)
            if token:
                segments.append(f"{key}={token}")
        dense = " | ".join(segments).strip()
        return dense[:1200]

    @staticmethod
    def _compact_text(value: str) -> str:
        token = " ".join(str(value or "").split()).strip()
        return token

    @staticmethod
    def _write_file(path: Path, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            logger.warning("world_state_store write failed path=%s error=%s", path, exc)

    def _max_events(self) -> int:
        try:
            return max(1, int(self._cfg.max_events))
        except (TypeError, ValueError):
            return 1

    def _max_decisions(self) -> int:
        try:
            return max(1, int(self._cfg.max_decisions))
        except (TypeError, ValueError):
            return 1
