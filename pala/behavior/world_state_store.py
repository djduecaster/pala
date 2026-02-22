from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
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
    summary: str
    delta_score: float
    features: Dict[str, Any] = field(default_factory=dict)
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
    """Small persistent behavior memory: identity, env summary, event tail, decision tail."""

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
            summary = self._event_summary_text(snapshot)
            if summary:
                self._append_event_locked(summary, ts_wall_s=snapshot.timestamp_wall_s, source="env_processor")
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
            if len(self.decision_tail) > self._max_decisions():
                self.decision_tail = self.decision_tail[-self._max_decisions() :]
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
                    "summary": self.latest_env_snapshot.summary,
                    "delta_score": self.latest_env_snapshot.delta_score,
                    "features": dict(self.latest_env_snapshot.features),
                    "timestamp_wall_s": self.latest_env_snapshot.timestamp_wall_s,
                }

            decisions = [
                {
                    "primitive": item.primitive,
                    "style": item.style,
                    "confidence": item.confidence,
                    "rationale_short": item.rationale_short,
                    "timestamp_wall_s": item.timestamp_wall_s,
                }
                for item in self.decision_tail
            ]

            return {
                "identity_core": self.identity_core,
                "latest_env_snapshot": env,
                "event_tail": list(self.event_tail),
                "decision_tail": decisions,
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
        if len(self.event_tail) > self._max_events():
            self.event_tail = self.event_tail[-self._max_events() :]

    def _persist_world_state_locked(self) -> None:
        path = Path(self._cfg.world_state_path)
        self._write_file(path, self._render_world_state_markdown())

    def _persist_session_digest_locked(self) -> None:
        path = Path(self._cfg.session_digest_path)
        payload = (self.session_digest.strip() + "\n") if self.session_digest else ""
        self._write_file(path, payload)

    def _load_identity_core(self) -> str:
        path = Path(self._cfg.identity_path)
        if not path.exists():
            return self._cfg.default_identity_core
        try:
            token = path.read_text(encoding="utf-8").strip()
            return token if token else self._cfg.default_identity_core
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
        lines.append(self.identity_core.strip() or DEFAULT_IDENTITY_CORE)
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
            lines.append("### Summary")
            lines.append(snap.summary.strip() or "_empty_")
            lines.append("")
            lines.append("### Features")
            if snap.features:
                lines.append("```json")
                lines.append(json.dumps(snap.features, ensure_ascii=True, indent=2, sort_keys=True))
                lines.append("```")
            else:
                lines.append("_none_")
        lines.append("")

        lines.append("## Event Tail")
        lines.append("")
        if not self.event_tail:
            lines.append("_none_")
        else:
            for idx, item in enumerate(self.event_tail[-self._max_events() :], start=1):
                ts = self._format_wall_time(float(item.get("timestamp_wall_s", 0.0)))
                source = str(item.get("source", "unknown")).strip() or "unknown"
                summary = str(item.get("summary", "")).strip() or "_empty_"
                lines.append(f"{idx}. [{ts}] source={source} {summary}")
        lines.append("")

        lines.append("## Decision Tail")
        lines.append("")
        if not self.decision_tail:
            lines.append("_none_")
        else:
            for idx, item in enumerate(self.decision_tail[-self._max_decisions() :], start=1):
                lines.append(
                    f"{idx}. [{self._format_wall_time(item.timestamp_wall_s)}] "
                    f"primitive={item.primitive} style={item.style} confidence={item.confidence:.2f} "
                    f"rationale={item.rationale_short}"
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
    def _event_summary_text(snapshot: EnvironmentSnapshot) -> str:
        events = WorldStateStore._compact_text(snapshot.events)
        hypotheses = WorldStateStore._compact_text(snapshot.hypotheses)
        summary = WorldStateStore._compact_text(snapshot.summary)
        zone = str(snapshot.features.get("zone_hint", "unknown")) if snapshot.features else "unknown"
        token = (
            f"events={events} | hypotheses={hypotheses} | summary={summary} | "
            f"delta={snapshot.delta_score:.2f} | zone={zone}"
        )
        return token[:1200]

    @staticmethod
    def _compact_text(value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

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
