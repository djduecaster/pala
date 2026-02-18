from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
import io
import json
import logging
import threading
import time
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from PIL import Image

from ..perception.frame_cache import LatestFrameCache
from ..types import (
    ActionPlan,
    PrimitiveKind,
    HoldCommand,
    BreathCommand,
    GlanceCommand,
    NodCommand,
    OrientToZoneCommand,
    PerceptionState,
)
from .heuristic import HeuristicPlanner
from .protocol import PlannerInterface
from .state_models import OrchestratorDecision, SceneSummary

logger = logging.getLogger(__name__)


@dataclass
class _OrchestratorRequest:
    summary: SceneSummary
    frames: list[np.ndarray]


class AsyncOrchestratorPlanner(PlannerInterface):
    """Async orchestrator with transcript context and latest-only remote requests."""

    def __init__(
        self,
        *,
        frame_cache: LatestFrameCache,
        fallback: Optional[PlannerInterface] = None,
        provider: str = "brev",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "nvidia/cosmos-reason2-2b",
        planner_prompt: Optional[str] = None,
        orchestrator_hz: float = 1.0,
        max_frame_age_ms: int = 500,
        video_window_s: float = 8.0,
        video_max_frames: int = 8,
        video_max_width: int = 320,
        video_jpeg_quality: int = 60,
        request_timeout_ms: int = 5000,
        response_ttl_ms: int = 1500,
        transcript_max_items: int = 80,
    ) -> None:
        self._frame_cache = frame_cache
        self._fallback = fallback or HeuristicPlanner()
        self._provider = str(provider).strip().lower()
        self._chat_url = _normalize_chat_url(base_url)
        self._api_key = api_key
        self._model = str(model or "nvidia/cosmos-reason2-2b")
        self._planner_prompt = (planner_prompt or "").strip()
        self._max_frame_age_ms = max(50, int(max_frame_age_ms))
        self._video_window_s = max(0.1, float(video_window_s))
        self._video_max_frames = max(1, int(video_max_frames))
        self._video_max_width = max(64, int(video_max_width))
        self._video_jpeg_quality = max(1, min(100, int(video_jpeg_quality)))
        self._request_timeout_s = max(0.05, float(request_timeout_ms) / 1000.0)
        self._response_ttl_s = max(0.1, float(response_ttl_ms) / 1000.0)
        self._orchestrator_period_s = 1.0 / max(0.2, float(orchestrator_hz))
        self._remote_enabled = self._provider in {"brev", "cosmos", "openai"} and self._chat_url is not None
        self._request_count = 0
        self._success_count = 0
        self._last_stats_log_s = 0.0

        self._transcript_lock = threading.Lock()
        self._transcript: deque[str] = deque(maxlen=max(10, int(transcript_max_items)))
        self._zone_history: deque[str] = deque(maxlen=6)
        self._frame_history: deque[tuple[float, int, np.ndarray]] = deque()
        self._last_seen_frame_mono_ns: Optional[int] = None

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._pending: Optional[_OrchestratorRequest] = None
        self._last_submit_s = 0.0

        self._latest_decision: Optional[OrchestratorDecision] = None
        self._latest_action: Optional[ActionPlan] = None
        self._latest_action_ts_s: Optional[float] = None
        self._latest_summary: Optional[SceneSummary] = None

        if self._remote_enabled:
            logger.info(
                "orchestrator remote enabled provider=%s url=%s model=%s",
                self._provider,
                self._chat_url,
                self._model,
            )
        else:
            logger.info("orchestrator remote disabled provider=%s base_url=%s", self._provider, self._chat_url)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def plan(self, st: PerceptionState) -> ActionPlan:
        now = time.monotonic()
        frame_age_ms = self._update_frame_history(now)
        summary = self._summarize_state(st, frame_age_ms)
        self._latest_summary = summary

        if (now - self._last_submit_s) >= self._orchestrator_period_s:
            frames = self._sample_frame_history()
            with self._lock:
                self._pending = _OrchestratorRequest(summary=summary, frames=frames)
                self._last_submit_s = now
                self._cond.notify_all()

        with self._lock:
            action = self._latest_action
            action_ts = self._latest_action_ts_s

        if action is not None and action_ts is not None and (now - action_ts) <= self._response_ttl_s:
            return action

        decision = _local_decision(summary)
        return _decision_to_action(decision)

    def snapshot(self) -> tuple[Optional[SceneSummary], list[str], Optional[OrchestratorDecision]]:
        with self._transcript_lock:
            transcript = list(self._transcript)
        return self._latest_summary, transcript, self._latest_decision

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                while not self._stop.is_set() and self._pending is None:
                    self._cond.wait(timeout=0.1)
                if self._stop.is_set():
                    break
                req = self._pending
                self._pending = None

            if req is None:
                continue

            try:
                if self._remote_enabled:
                    decision = self._remote_decision(req)
                    if decision is None:
                        decision = _local_decision(req.summary)
                else:
                    decision = _local_decision(req.summary)
                action = _decision_to_action(decision)
            except Exception as exc:
                logger.warning("orchestrator planning failed: %s", exc)
                decision = _local_decision(req.summary)
                action = _decision_to_action(decision)

            self._append_transcript(
                "decision",
                (
                    f"source={decision.source} intent={decision.intent} style={decision.style} "
                    f"primitive={decision.primitive_hint or 'breath'} target_zone={decision.target_zone or '-'} "
                    f"confidence={decision.confidence:.2f} frames={len(req.frames)} rationale={decision.rationale}"
                ),
            )
            self._latest_decision = decision
            with self._lock:
                self._latest_action = action
                self._latest_action_ts_s = time.monotonic()

            now = time.monotonic()
            if now - self._last_stats_log_s >= 10.0:
                logger.info(
                    "orchestrator stats requests=%d successes=%d source=%s intent=%s style=%s",
                    self._request_count,
                    self._success_count,
                    decision.source,
                    decision.intent,
                    decision.style,
                )
                self._last_stats_log_s = now

    def _update_frame_history(self, now_s: float) -> Optional[float]:
        snap = self._frame_cache.get(max_age_ms=self._max_frame_age_ms)
        if snap is None:
            return None
        frame_age_ms = (time.monotonic_ns() - snap.mono_ns) / 1_000_000.0
        if self._last_seen_frame_mono_ns == snap.mono_ns:
            return frame_age_ms
        self._last_seen_frame_mono_ns = snap.mono_ns
        self._frame_history.append((now_s, snap.mono_ns, np.asarray(snap.frame).copy()))

        cutoff = now_s - self._video_window_s
        while self._frame_history and self._frame_history[0][0] < cutoff:
            self._frame_history.popleft()
        return frame_age_ms

    def _sample_frame_history(self) -> list[np.ndarray]:
        if not self._frame_history:
            return []
        frames = [entry[2] for entry in self._frame_history]
        if len(frames) <= self._video_max_frames:
            return frames
        step = len(frames) / float(self._video_max_frames)
        idxs = [int(i * step) for i in range(self._video_max_frames)]
        return [frames[i] for i in idxs]

    def _summarize_state(self, st: PerceptionState, frame_age_ms: Optional[float]) -> SceneSummary:
        zone = None
        if isinstance(st.debug, dict):
            raw_zone = st.debug.get("zone_hint")
            if isinstance(raw_zone, str) and raw_zone:
                zone = raw_zone
        if zone is not None:
            self._zone_history.append(zone)

        person_present = st.primary_person is not None
        uncertainty_flags: list[str] = []
        conf = st.primary_person_conf
        if person_present and conf is not None and conf < 0.5:
            uncertainty_flags.append("low_person_conf")

        if frame_age_ms is not None and frame_age_ms > 250.0:
            uncertainty_flags.append("stale_frame")

        activity_hint = self._infer_activity_hint(person_present, zone)
        return SceneSummary(
            timestamp_monotonic_s=time.monotonic(),
            person_present=person_present,
            zone_hint=zone,
            primary_person_conf=conf,
            activity_hint=activity_hint,
            uncertainty_flags=uncertainty_flags,
            frame_age_ms=frame_age_ms,
        )

    def _infer_activity_hint(self, person_present: bool, zone: Optional[str]) -> Optional[str]:
        if not person_present:
            return "away"
        if len(self._zone_history) < 3:
            return "engaged"
        recent = list(self._zone_history)[-3:]
        if len(set(recent)) >= 2:
            return "transitioning"
        if zone == "center":
            return "focused_work"
        return "engaged"

    def _remote_decision(self, req: _OrchestratorRequest) -> Optional[OrchestratorDecision]:
        assert self._chat_url is not None
        self._request_count += 1
        payload = self._build_payload(req)
        response = _post_json(
            self._chat_url,
            payload,
            timeout_s=self._request_timeout_s,
            api_key=self._api_key,
        )
        content = _extract_content(response)
        if content is None:
            return None
        decision = _parse_decision_content(content)
        if decision is None:
            return None
        self._success_count += 1
        return decision

    def _build_payload(self, req: _OrchestratorRequest) -> dict[str, Any]:
        with self._transcript_lock:
            transcript_tail = list(self._transcript)[-24:]
        image_data_urls, resized_shapes, total_jpeg_bytes = _encode_frames_to_data_urls(
            req.frames,
            max_width=self._video_max_width,
            jpeg_quality=self._video_jpeg_quality,
        )
        context = {
            "summary": {
                "person_present": req.summary.person_present,
                "zone_hint": req.summary.zone_hint,
                "primary_person_conf": req.summary.primary_person_conf,
                "activity_hint": req.summary.activity_hint,
                "uncertainty_flags": req.summary.uncertainty_flags,
                "frame_age_ms": req.summary.frame_age_ms,
            },
            "transcript": transcript_tail,
            "frames_sent": len(image_data_urls),
            "resized_shapes": resized_shapes,
            "jpeg_total_bytes": total_jpeg_bytes,
        }
        user_text = "Use the provided context and frame sequence to decide next action.\n" + json.dumps(
            context,
            separators=(",", ":"),
            ensure_ascii=True,
        )

        system_prompt = (
            "You are an interaction orchestrator for a social desk robot lamp. "
            "You may receive multiple frames in chronological order. "
            "Return JSON only with keys: intent, style, primitive_hint, target_zone, confidence, rationale. "
            "style must be one of ['calm','curious','focused']. "
            "primitive_hint should be one of ['hold','breath','glance','nod','orient_to_zone'] or null. "
            "target_zone should be one of ['left','center','right'] or null. "
            "confidence in [0,1]. Keep rationale concise."
        )
        if self._planner_prompt:
            system_prompt += f" Operator guidance: {self._planner_prompt}"

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *[{"type": "image_url", "image_url": {"url": u}} for u in image_data_urls],
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 180,
            "stream": False,
        }

    def _append_transcript(self, role: str, content: str) -> None:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        with self._transcript_lock:
            self._transcript.append(f"{timestamp} {role}: {content}")


def _local_decision(summary: SceneSummary) -> OrchestratorDecision:
    if not summary.person_present:
        return OrchestratorDecision(
            intent="idle_presence",
            style="calm",
            primitive_hint="breath",
            target_zone=None,
            confidence=0.6,
            rationale="no person present",
            source="local",
        )

    if summary.activity_hint == "transitioning":
        return OrchestratorDecision(
            intent="track_transition",
            style="curious",
            primitive_hint="glance",
            target_zone=summary.zone_hint,
            confidence=0.7,
            rationale="person motion implies context shift",
            source="local",
        )

    if summary.zone_hint == "center":
        return OrchestratorDecision(
            intent="engaged_focus",
            style="focused",
            primitive_hint="nod",
            target_zone=summary.zone_hint,
            confidence=0.72,
            rationale="stable centered engagement",
            source="local",
        )

    return OrchestratorDecision(
        intent="maintain_presence",
        style="curious",
        primitive_hint="orient_to_zone",
        target_zone=summary.zone_hint,
        confidence=0.66,
        rationale="person present off-center",
        source="local",
    )


def _decision_to_action(decision: OrchestratorDecision) -> ActionPlan:
    hint = (decision.primitive_hint or "").strip().lower()
    if hint == "nod":
        return ActionPlan(
            primitive=PrimitiveKind.NOD,
            command=NodCommand(amp_rad=0.2, duration_s=0.5, cycles=1, rate_rad_s=1.8),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
        )
    if hint == "glance":
        direction = "left"
        if decision.target_zone == "right":
            direction = "right"
        elif decision.target_zone == "left":
            direction = "left"
        elif "right" in decision.rationale:
            direction = "right"
        return ActionPlan(
            primitive=PrimitiveKind.GLANCE,
            command=GlanceCommand(direction=direction, amp_rad=0.24, duration_s=0.55, rate_rad_s=1.6),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
        )
    if hint == "orient_to_zone":
        zone = decision.target_zone if decision.target_zone in {"left", "center", "right"} else "center"
        if zone == "center" and "left" in decision.rationale:
            zone = "left"
        elif zone == "center" and "right" in decision.rationale:
            zone = "right"
        return ActionPlan(
            primitive=PrimitiveKind.ORIENT_TO_ZONE,
            command=OrientToZoneCommand(zone=zone, amp_rad=0.2, rate_rad_s=1.4),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
        )
    if hint == "hold":
        return ActionPlan(
            primitive=PrimitiveKind.HOLD,
            command=HoldCommand(),
            confidence=decision.confidence,
            explanation=f"{decision.source}:{decision.rationale}",
            style=decision.style,
        )
    return ActionPlan(
        primitive=PrimitiveKind.BREATH,
        command=BreathCommand(amp_rad=0.08, period_s=6.5, rate_rad_s=1.0),
        confidence=decision.confidence,
        explanation=f"{decision.source}:{decision.rationale}",
        style=decision.style,
    )


def _encode_frames_to_data_urls(
    frames: list[np.ndarray],
    *,
    max_width: int,
    jpeg_quality: int,
) -> tuple[list[str], list[list[int]], int]:
    urls: list[str] = []
    resized_shapes: list[list[int]] = []
    total_bytes = 0
    if not frames:
        return urls, resized_shapes, total_bytes

    for frame in frames:
        arr = np.asarray(frame)
        img = Image.fromarray(arr)
        if max_width > 0 and img.width > max_width:
            new_h = int(round((max_width / float(img.width)) * img.height))
            img = img.resize((max_width, max(1, new_h)))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
        jpeg = out.getvalue()
        total_bytes += len(jpeg)
        resized_shapes.append([img.height, img.width, 3])
        urls.append("data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"))
    return urls, resized_shapes, total_bytes


def _normalize_chat_url(base_url: Optional[str]) -> Optional[str]:
    if base_url is None:
        return None
    base = str(base_url).strip()
    if not base:
        return None
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if base.endswith("/v1/"):
        return f"{base}chat/completions"
    return f"{base.rstrip('/')}/v1/chat/completions"


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float, api_key: Optional[str]) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    req = urllib_request.Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid JSON response from orchestrator endpoint") from exc


def _extract_content(response: dict[str, Any]) -> Optional[str]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None
    message = choice0.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        joined = "\n".join(parts).strip()
        return joined if joined else None
    return None


def _parse_decision_content(content: str) -> Optional[OrchestratorDecision]:
    cleaned = content.strip()
    data = _parse_json_obj(cleaned)
    if data is None:
        candidate = _extract_first_json_object(cleaned)
        data = None if candidate is None else _parse_json_obj(candidate)
    if data is None or not isinstance(data, dict):
        return None

    if isinstance(data.get("decision"), dict):
        data = data["decision"]

    intent = str(data.get("intent", "")).strip()
    style = str(data.get("style", "calm")).strip().lower()
    primitive_hint = data.get("primitive_hint")
    primitive_hint_str = None if primitive_hint in (None, "") else str(primitive_hint).strip().lower()
    target_zone_raw = data.get("target_zone", data.get("zone_hint"))
    target_zone = None if target_zone_raw in (None, "") else str(target_zone_raw).strip().lower()
    if target_zone not in {None, "left", "center", "right"}:
        target_zone = None
    rationale = str(data.get("rationale", "")).strip()
    if not intent or not rationale:
        return None
    try:
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    if style not in {"calm", "curious", "focused"}:
        style = "calm"
    return OrchestratorDecision(
        intent=intent,
        style=style,
        primitive_hint=primitive_hint_str,
        target_zone=target_zone,
        confidence=confidence,
        rationale=rationale,
        source="remote",
    )


def _parse_json_obj(raw: str) -> Optional[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_first_json_object(raw: str) -> Optional[str]:
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None
