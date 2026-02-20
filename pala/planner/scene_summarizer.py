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
from ..types import PerceptionState
from .state_models import SceneSummary
from .timeline import TimelineWriter

logger = logging.getLogger(__name__)

_SUMMARY_STATES = {"idle_presence", "user_detected", "focused_work", "transition", "uncertain"}


@dataclass
class _SummaryRequest:
    state: PerceptionState
    frames: list[np.ndarray]
    frame_age_ms: Optional[float]


class AsyncSceneSummarizer:
    """Low-rate scene summarizer that emits compact structured world snapshots."""

    def __init__(
        self,
        *,
        frame_cache: LatestFrameCache,
        provider: str = "brev",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "nvidia/cosmos-reason2-2b",
        policy_version: str = "v1",
        summarizer_hz: float = 1.0,
        max_frame_age_ms: int = 500,
        summary_window_s: float = 6.0,
        summary_max_frames: int = 4,
        summary_max_width: int = 320,
        summary_jpeg_quality: int = 55,
        request_timeout_ms: int = 6000,
        request_min_fresh_frames: int = 1,
        timeline: Optional[TimelineWriter] = None,
        memory_manager: Optional[Any] = None,
        history_max_items: int = 32,
    ) -> None:
        self._frame_cache = frame_cache
        self._provider = str(provider).strip().lower()
        self._chat_url = _normalize_chat_url(base_url)
        self._api_key = api_key
        self._model = str(model or "nvidia/cosmos-reason2-2b")
        self._policy_version = str(policy_version or "v1").strip() or "v1"
        self._period_s = 1.0 / max(0.1, float(summarizer_hz))
        self._max_frame_age_ms = max(50, int(max_frame_age_ms))
        self._summary_window_s = max(0.1, float(summary_window_s))
        self._summary_max_frames = max(1, int(summary_max_frames))
        self._summary_max_width = max(64, int(summary_max_width))
        self._summary_jpeg_quality = max(1, min(100, int(summary_jpeg_quality)))
        self._request_timeout_s = max(0.1, float(request_timeout_ms) / 1000.0)
        self._request_min_fresh_frames = max(0, int(request_min_fresh_frames))
        self._timeline = timeline
        self._memory = memory_manager
        self._remote_enabled = self._provider in {"brev", "cosmos", "openai"} and self._chat_url is not None

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._pending: Optional[_SummaryRequest] = None
        self._last_submit_s = 0.0
        self._request_seq = 0

        self._frame_history: deque[tuple[float, int, np.ndarray]] = deque()
        self._last_seen_frame_mono_ns: Optional[int] = None

        self._summary_lock = threading.Lock()
        self._latest_summary: Optional[SceneSummary] = None
        self._history: deque[SceneSummary] = deque(maxlen=max(4, int(history_max_items)))

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def observe(self, st: PerceptionState) -> None:
        now = time.monotonic()
        frame_age_ms = self._update_frame_history(now)
        if (now - self._last_submit_s) < self._period_s:
            return
        frames = self._sample_frame_history()
        if len(frames) < self._request_min_fresh_frames:
            return
        with self._lock:
            self._pending = _SummaryRequest(state=st, frames=frames, frame_age_ms=frame_age_ms)
            self._last_submit_s = now
            self._cond.notify_all()

    def latest_summary(self) -> Optional[SceneSummary]:
        with self._summary_lock:
            return self._latest_summary

    def recent_summaries(self, limit: int) -> list[SceneSummary]:
        n = max(0, int(limit))
        if n == 0:
            return []
        with self._summary_lock:
            return list(self._history)[-n:]

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

            if self._remote_enabled:
                summary = self._remote_summary(req)
            else:
                summary = self._local_summary(req)
            if summary is None:
                continue

            with self._summary_lock:
                self._latest_summary = summary
                self._history.append(summary)
            payload = summary.to_payload()
            if self._memory is not None:
                try:
                    self._memory.append_event("summary_event", payload)
                except Exception:
                    logger.debug("scene summarizer memory append failed", exc_info=True)
            if self._timeline is not None:
                self._timeline.write(
                    "summary_event",
                    {
                        "scene_state": summary.scene_state,
                        "person_present": summary.person_present,
                        "zone_hint": summary.zone_hint,
                        "confidence": summary.confidence,
                        "source": summary.source,
                        "policy_version": self._policy_version,
                    },
                )

    def _update_frame_history(self, now_s: float) -> Optional[float]:
        self._prune_frame_history(now_s)
        snap = self._frame_cache.get(max_age_ms=self._max_frame_age_ms)
        if snap is None:
            return None
        frame_age_ms = (time.monotonic_ns() - snap.mono_ns) / 1_000_000.0
        if self._last_seen_frame_mono_ns == snap.mono_ns:
            return frame_age_ms
        self._last_seen_frame_mono_ns = snap.mono_ns
        self._frame_history.append((now_s, snap.mono_ns, np.asarray(snap.frame).copy()))
        self._prune_frame_history(now_s)
        return frame_age_ms

    def _prune_frame_history(self, now_s: float) -> None:
        cutoff = now_s - self._summary_window_s
        while self._frame_history and self._frame_history[0][0] < cutoff:
            self._frame_history.popleft()

    def _sample_frame_history(self) -> list[np.ndarray]:
        if not self._frame_history:
            return []
        frames = [entry[2] for entry in self._frame_history]
        if len(frames) <= self._summary_max_frames:
            return frames
        if self._summary_max_frames == 1:
            return [frames[-1]]
        n = len(frames)
        k = self._summary_max_frames
        idxs = [int(i * (n - 1) / (k - 1)) for i in range(k - 1)] + [n - 1]
        return [frames[i] for i in idxs]

    def _next_request_id(self) -> int:
        self._request_seq += 1
        return self._request_seq

    def _remote_summary(self, req: _SummaryRequest) -> Optional[SceneSummary]:
        if self._chat_url is None:
            return None
        req_id = self._next_request_id()
        if self._timeline is not None:
            self._timeline.write(
                "summary_request_start",
                {
                    "request_id": req_id,
                    "frames": len(req.frames),
                    "frame_age_ms": req.frame_age_ms,
                    "policy_version": self._policy_version,
                },
            )
        try:
            payload = self._build_payload(req)
            t0 = time.monotonic()
            response = _post_json(self._chat_url, payload, timeout_s=self._request_timeout_s, api_key=self._api_key)
            latency_ms = (time.monotonic() - t0) * 1000.0
            content = _extract_content(response)
            if content is None:
                if self._timeline is not None:
                    self._timeline.write(
                        "summary_request_end",
                        {
                            "request_id": req_id,
                            "status": "no_content",
                            "latency_ms": latency_ms,
                            "policy_version": self._policy_version,
                        },
                    )
                return None
            summary = _parse_summary_content(content)
            if summary is None:
                if self._timeline is not None:
                    self._timeline.write(
                        "summary_request_end",
                        {
                            "request_id": req_id,
                            "status": "parse_fail",
                            "latency_ms": latency_ms,
                            "preview": _preview(content, 300),
                            "policy_version": self._policy_version,
                        },
                    )
                return None
            summary.ts_wall_s = time.time()
            summary.ts_mono_s = time.monotonic()
            summary.source = "remote"
            if self._timeline is not None:
                self._timeline.write(
                    "summary_request_end",
                    {
                        "request_id": req_id,
                        "status": "ok",
                        "latency_ms": latency_ms,
                        "scene_state": summary.scene_state,
                        "zone_hint": summary.zone_hint,
                        "confidence": summary.confidence,
                        "policy_version": self._policy_version,
                    },
                )
            return summary
        except Exception as exc:
            logger.debug("scene summarizer remote request failed: %s", exc)
            if self._timeline is not None:
                self._timeline.write(
                    "summary_request_end",
                    {
                        "request_id": req_id,
                        "status": "error",
                        "error": str(exc),
                        "policy_version": self._policy_version,
                    },
                )
            return None

    def _local_summary(self, req: _SummaryRequest) -> SceneSummary:
        person_present = req.state.primary_person is not None
        zone_hint = None
        if isinstance(req.state.debug, dict):
            raw_zone = req.state.debug.get("zone_hint")
            if isinstance(raw_zone, str) and raw_zone in {"left", "center", "right"}:
                zone_hint = raw_zone
        scene_state = "user_detected" if person_present else "idle_presence"
        activity_hint = None
        if isinstance(req.state.debug, dict):
            raw_activity = req.state.debug.get("activity_hint")
            if isinstance(raw_activity, str):
                activity_hint = raw_activity
        return SceneSummary(
            ts_wall_s=time.time(),
            ts_mono_s=time.monotonic(),
            scene_state=scene_state,
            person_present=person_present,
            zone_hint=zone_hint,
            notable_changes=[],
            activity_hint=activity_hint,
            uncertainty=[],
            confidence=0.35,
            rationale="local summary fallback",
            source="local",
        )

    def _build_payload(self, req: _SummaryRequest) -> dict[str, Any]:
        image_data_urls = _encode_frames_to_data_urls(
            req.frames,
            max_width=self._summary_max_width,
            jpeg_quality=self._summary_jpeg_quality,
        )
        user_text = (
            f"[policy_version={self._policy_version}]\n"
            "You are generating compact scene summaries for an embodied desk lamp.\n"
            "Return JSON only with keys:\n"
            "scene_state,person_present,zone_hint,notable_changes,activity_hint,uncertainty,confidence,rationale.\n"
            "Allowed scene_state values: idle_presence,user_detected,focused_work,transition,uncertain.\n"
            "zone_hint must be left|center|right|null.\n"
            "notable_changes and uncertainty must be short string arrays.\n"
            "confidence must be 0..1.\n"
            "Answer the question using the following format:\n"
            "<think>\n"
            "Your reasoning.\n"
            "</think>\n"
            "Write your final answer immediately after the </think> tag.\n\n"
            f"frame_age_ms={req.frame_age_ms}\n"
            "Summarize current scene for high-level planner continuity."
        )
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image_url", "image_url": {"url": u}} for u in image_data_urls],
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 300,
            "stream": False,
        }


def _encode_frames_to_data_urls(frames: list[np.ndarray], *, max_width: int, jpeg_quality: int) -> list[str]:
    urls: list[str] = []
    for frame in frames:
        arr = np.asarray(frame)
        img = Image.fromarray(arr)
        if max_width > 0 and img.width > max_width:
            new_h = int(round((max_width / float(img.width)) * img.height))
            img = img.resize((max_width, max(1, new_h)))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
        jpeg = out.getvalue()
        urls.append("data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"))
    return urls


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
        raise RuntimeError("invalid JSON response from scene summarizer endpoint") from exc


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


def _parse_summary_content(content: str) -> Optional[SceneSummary]:
    data = _parse_json_obj(_strip_markdown_fences(content.strip()))
    if data is None:
        candidate = _extract_first_json_object(content.strip())
        data = None if candidate is None else _parse_json_obj(candidate)
    if data is None:
        return None

    scene_state = _clean_text(data.get("scene_state"))
    if scene_state is None:
        return None
    scene_state = scene_state.lower()
    if scene_state not in _SUMMARY_STATES:
        return None

    person_present = _coerce_bool(data.get("person_present"))
    if person_present is None:
        return None

    zone_hint = _clean_text(data.get("zone_hint"))
    if zone_hint is not None:
        zone_hint = zone_hint.lower()
        if zone_hint not in {"left", "center", "right"}:
            zone_hint = None

    notable_changes = _coerce_string_list(data.get("notable_changes"))
    uncertainty = _coerce_string_list(data.get("uncertainty"))
    activity_hint = _clean_text(data.get("activity_hint"))
    rationale = _clean_text(data.get("rationale"))
    if rationale is None:
        return None
    try:
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        return None
    confidence = max(0.0, min(1.0, confidence))

    return SceneSummary(
        ts_wall_s=0.0,
        ts_mono_s=0.0,
        scene_state=scene_state,
        person_present=person_present,
        zone_hint=zone_hint,
        notable_changes=notable_changes,
        activity_hint=activity_hint,
        uncertainty=uncertainty,
        confidence=confidence,
        rationale=rationale,
    )


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        token = item.strip()
        if token:
            out.append(token)
    return out


def _clean_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token:
        return None
    if token.lower() in {"none", "null", "n/a", "na"}:
        return None
    return token


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "on"}:
            return True
        if token in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _parse_json_obj(raw: str) -> Optional[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


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


def _preview(raw: str, max_chars: int = 200) -> str:
    text = raw.strip().replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
