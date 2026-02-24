from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from ..types import ActionPlan, PerceptionState, to_json_dict


class ContextBuilder:
    def build_env_context(
        self,
        *,
        world_snapshot: Mapping[str, Any],
        current_action: ActionPlan,
        frame_timeline: List[Dict[str, float]],
        mode: str = "idle_presence",
    ) -> Dict[str, Any]:
        latest_env = world_snapshot.get("latest_env_snapshot") or {}
        event_tail = world_snapshot.get("event_tail", [])[-2:]
        recent_events = [
            {
                "t": self._format_ts_seconds(item.get("timestamp_wall_s")),
                "summary": self._short_text(item.get("summary"), max_chars=220),
            }
            for item in event_tail
        ]
        return {
            "mode": mode,
            "current_action": {
                "primitive": current_action.primitive.value,
                "style": current_action.style,
                "confidence": float(current_action.confidence),
            },
            "latest_env_summary": self._short_text(latest_env.get("summary"), max_chars=180),
            "recent_env_events": recent_events,
            "control_state": world_snapshot.get("control_state_latest"),
            "frame_timeline": frame_timeline,
        }

    def build_planner_context(
        self,
        *,
        st: Optional[PerceptionState],
        world_snapshot: Mapping[str, Any],
        current_action: ActionPlan,
        planner_health: Mapping[str, Any],
        mode: str = "idle_presence",
        now_mono_s: float,
        last_commit_mono_s: float,
        no_commit_s: float,
    ) -> Dict[str, Any]:
        latest_env = world_snapshot.get("latest_env_snapshot") or {}
        features = latest_env.get("features") or {}

        zone_hint: Optional[str] = None
        person_conf = None
        if st is not None:
            person_conf = st.primary_person_conf
            if st.debug:
                zone_candidate = str(st.debug.get("zone_hint") or "").strip().lower()
                if zone_candidate in {"left", "center", "right"}:
                    zone_hint = zone_candidate
        if zone_hint is None:
            zone_candidate = str(features.get("zone_hint") or "").strip().lower()
            if zone_candidate in {"left", "center", "right"}:
                zone_hint = zone_candidate

        evidence_ids = ["frame:latest", "env:latest"]
        if zone_hint is not None:
            evidence_ids.append(f"perception:zone:{zone_hint}")

        signals: Dict[str, Any] = {
            "person_conf": person_conf,
            "env_delta": self._as_float(latest_env.get("delta_score"), default=0.0),
            "activity_level": self._as_float(features.get("activity_level"), default=0.0),
            "novelty": self._as_float(features.get("novelty"), default=0.0),
            "person_present": bool(features.get("person_present", False)),
        }
        if zone_hint is not None:
            signals["zone_hint"] = zone_hint

        return {
            "mode": mode,
            "current_action": {
                "primitive": current_action.primitive.value,
                "command": self._command_digest(current_action.command),
                "style": current_action.style,
                "confidence": float(current_action.confidence),
                "age_s": max(0.0, float(now_mono_s - last_commit_mono_s)),
            },
            "signals": signals,
            "latest_env": {
                "scene": self._short_text(latest_env.get("scene"), max_chars=380),
                "summary": self._short_text(latest_env.get("summary"), max_chars=160),
            },
            "control_state": world_snapshot.get("control_state_latest"),
            "planner_health": dict(planner_health),
            "anti_collapse": {
                "no_commit_s": max(0.0, float(no_commit_s)),
            },
            "evidence_index": {
                "available": evidence_ids,
            },
        }

    @staticmethod
    def _short_text(value: Any, *, max_chars: int) -> str:
        token = " ".join(str(value or "").split()).strip()
        if not token:
            return ""
        if len(token) <= max_chars:
            return token
        return token[: max_chars - 3] + "..."

    @staticmethod
    def _format_ts_seconds(ts_wall_s: Any) -> Optional[str]:
        try:
            ts = float(ts_wall_s)
        except (TypeError, ValueError):
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _as_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _command_digest(command: Any) -> Any:
        try:
            token = to_json_dict(command)
        except Exception:  # noqa: BLE001
            token = str(command)
        if isinstance(token, Mapping):
            return {str(k): token[k] for k in token.keys()}
        return token
