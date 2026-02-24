#!/usr/bin/env python3
"""Live camera -> Cosmos image probe for latency and schema characterization."""
from __future__ import annotations

import argparse
import base64
from collections import deque
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image

from pala.behavior.model_clients import extract_message_content, normalize_chat_url, post_chat_json
from pala.config import load_config
from pala.control.primitives import PrimitiveKind
from pala.hardware.camera import DummyCamera
from pala.perception.frame_source import CameraFrameSource, ThreadedFrameSource


@dataclass
class ProbeResult:
    idx: int
    ok_http: bool
    ok_parse: bool
    encode_ms: float
    http_ms: float
    total_ms: float
    status: int
    primitive: Optional[str]
    confidence: Optional[float]
    explanation: Optional[str]
    response_preview: str
    frame_shape: Optional[list[int]]
    resized_shape: Optional[list[int]]
    jpeg_bytes: int
    frames_sent: int
    error: Optional[str]

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


def _parse_action_content(content: str) -> Optional[dict[str, Any]]:
    cleaned = content.strip()
    data = _parse_json_obj(cleaned)
    if data is None:
        candidate = _extract_first_json_object(cleaned)
        data = None if candidate is None else _parse_json_obj(candidate)
    if data is None:
        return None
    if isinstance(data.get("action"), dict):
        data = data["action"]
    if not isinstance(data, dict):
        return None
    return data


def _encode_frame(frame: np.ndarray, *, max_width: int, jpeg_quality: int) -> tuple[bytes, list[int]]:
    arr = np.asarray(frame)
    img = Image.fromarray(arr)
    if max_width > 0 and img.width > max_width:
        new_h = int(round((max_width / float(img.width)) * img.height))
        img = img.resize((max_width, max(1, new_h)))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
    return out.getvalue(), [img.height, img.width, 3]


def _build_payload(
    *,
    task: str,
    question: str,
    model: str,
    planner_prompt: str,
    image_data_urls: list[str],
    frame_shapes: list[list[int]],
    resized_shapes: list[list[int]],
    frame_age_ms: Optional[float],
) -> dict[str, Any]:
    user_context = {
        "source": "cosmos_image_probe",
        "frames_sent": len(image_data_urls),
        "frame_shapes": frame_shapes,
        "resized_shapes": resized_shapes,
        "frame_age_ms": frame_age_ms,
        "planner_prompt": planner_prompt,
    }
    user_text = (
        f"{question}\n"
        f"context={json.dumps(user_context, separators=(',', ':'), ensure_ascii=True)}"
    )

    if task == "describe":
        system_prompt = (
            "You are a vision assistant. You are given one or more frames in chronological order. "
            "Describe what is happening in 1-2 short sentences. "
            "If scene content is unclear, say that directly."
        )
        if planner_prompt:
            system_prompt += f" Operator guidance: {planner_prompt}"
        content = [{"type": "text", "text": user_text}]
        for image_data_url in image_data_urls:
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": 220,
            "stream": False,
        }

    system_prompt = (
        "You are a robotics planner. Return only JSON with keys "
        "primitive, command, confidence, explanation. "
        f"Allowed primitive values: {[k.value for k in PrimitiveKind]}. "
        "Confidence must be 0..1. No markdown or prose."
    )
    if planner_prompt:
        system_prompt += f" Operator guidance: {planner_prompt}"
    content = [{"type": "text", "text": user_text}]
    for image_data_url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 160,
        "stream": False,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _open_source(cfg, mode: str, use_threaded: bool, capture_hz: Optional[float]):
    if mode == "jetson_full":
        from pala.hardware.camera_gst import GStreamerCamera

        camera = GStreamerCamera(
            device=cfg.camera.device,
            width=cfg.camera.width,
            height=cfg.camera.height,
            fps=cfg.camera.fps,
            pipeline=cfg.camera.pipeline,
        )
    else:
        camera = DummyCamera(width=cfg.camera.width, height=cfg.camera.height)

    base_source = CameraFrameSource(camera)
    if use_threaded:
        hz = float(capture_hz) if capture_hz is not None else float(max(1, cfg.camera.fps))
        min_interval_s = 0.0 if hz <= 0 else (1.0 / hz)
        reader = ThreadedFrameSource(base_source, min_interval_s=min_interval_s)
    else:
        reader = base_source
    return reader, base_source, use_threaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Live image probe against Cosmos endpoint")
    parser.add_argument("--config", default="config/robot.yaml", help="Path to robot config")
    parser.add_argument("--mode", default=None, help="Override mode (default: config mode)")
    parser.add_argument("--base-url", default=None, help="Cosmos base URL, e.g. http://<ip>:8000")
    parser.add_argument("--api-key", default=None, help="Optional bearer API key")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--prompt", default=None, help="Planner prompt override")
    parser.add_argument(
        "--task",
        choices=["describe", "action"],
        default="describe",
        help="Probe task: freeform description or action JSON",
    )
    parser.add_argument(
        "--question",
        default="What is happening in this image?",
        help="Question sent with the image",
    )
    parser.add_argument("--count", type=int, default=20, help="Number of requests")
    parser.add_argument("--hz", type=float, default=1.0, help="Request cadence (default 1 Hz)")
    parser.add_argument(
        "--capture-hz",
        type=float,
        default=None,
        help="Capture rate for threaded reader (default: camera fps)",
    )
    parser.add_argument("--timeout-s", type=float, default=None, help="HTTP timeout (seconds)")
    parser.add_argument("--frame-timeout-s", type=float, default=1.0, help="Frame wait timeout for threaded mode")
    parser.add_argument("--max-width", type=int, default=320, help="Max image width before upload")
    parser.add_argument("--jpeg-quality", type=int, default=60, help="JPEG quality for upload")
    parser.add_argument(
        "--video-window-s",
        type=float,
        default=0.0,
        help="If > 0, send a rolling frame sequence covering this many seconds",
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=12,
        help="Cap frames sent per request when --video-window-s > 0",
    )
    parser.add_argument("--unthreaded", action="store_true", help="Disable threaded latest-frame capture")
    parser.add_argument("--out", default="logs/cosmos_image_probe.jsonl", help="JSONL output path (empty disables)")
    parser.add_argument("--print-response", action="store_true", help="Print response preview each request")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.mode
    base_url = args.base_url or os.getenv("PALA_COSMOS_BASE_URL") or cfg.cosmos.base_url
    if not base_url:
        print("cosmos_image_probe: missing base URL (use --base-url or PALA_COSMOS_BASE_URL).")
        return 2
    chat_url = normalize_chat_url(base_url)
    model = args.model or os.getenv("PALA_COSMOS_MODEL") or cfg.cosmos.model
    planner_prompt = args.prompt or os.getenv("PALA_COSMOS_PROMPT") or cfg.cosmos.planner_prompt
    api_key = args.api_key or os.getenv("PALA_COSMOS_API_KEY")
    timeout_s = args.timeout_s if args.timeout_s is not None else max(0.5, cfg.cosmos.request_timeout_ms / 1000.0)
    use_threaded = not args.unthreaded

    reader, base_source, is_threaded = _open_source(
        cfg,
        mode=mode,
        use_threaded=use_threaded,
        capture_hz=args.capture_hz,
    )

    out_fh = None
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_fh = open(args.out, "a", encoding="utf-8")

    encode_ms_vals: list[float] = []
    http_ms_vals: list[float] = []
    total_ms_vals: list[float] = []
    ok_http_count = 0
    ok_parse_count = 0

    period_s = 1.0 / max(0.01, float(args.hz))
    next_due = time.monotonic()
    video_history: deque[tuple[float, np.ndarray]] = deque()
    print(
        f"cosmos_image_probe: mode={mode} url={chat_url} model={model} hz={args.hz:.2f} "
        f"count={args.count} threaded={is_threaded} task={args.task} "
        f"video_window_s={args.video_window_s:.1f}"
    )

    try:
        for idx in range(1, max(1, args.count) + 1):
            now = time.monotonic()
            if now < next_due:
                time.sleep(next_due - now)
            next_due += period_s

            t0 = time.monotonic()
            if is_threaded:
                packet = reader.get_latest(timeout_s=args.frame_timeout_s)
                if packet is None:
                    print(f"[{idx:03d}] frame_timeout")
                    continue
            else:
                packet = reader.get_packet()

            frame = np.asarray(packet.frame)
            frame_shape = list(frame.shape)
            frame_age_ms = (time.monotonic_ns() - packet.mono_ns) / 1_000_000.0
            now_s = time.monotonic()
            video_history.append((now_s, frame))
            if args.video_window_s > 0:
                cutoff = now_s - max(0.1, float(args.video_window_s))
                while video_history and video_history[0][0] < cutoff:
                    video_history.popleft()
            else:
                while len(video_history) > 1:
                    video_history.popleft()

            frames_for_request = [entry[1] for entry in video_history]
            max_frames = max(1, int(args.video_max_frames))
            if len(frames_for_request) > max_frames:
                step = len(frames_for_request) / float(max_frames)
                idxs = [int(i * step) for i in range(max_frames)]
                frames_for_request = [frames_for_request[i] for i in idxs]

            t_enc0 = time.monotonic()
            image_data_urls: list[str] = []
            resized_shapes: list[list[int]] = []
            total_jpeg_bytes = 0
            for req_frame in frames_for_request:
                jpeg_bytes, resized_shape = _encode_frame(
                    req_frame,
                    max_width=max(1, int(args.max_width)),
                    jpeg_quality=max(1, min(100, int(args.jpeg_quality))),
                )
                total_jpeg_bytes += len(jpeg_bytes)
                resized_shapes.append(resized_shape)
                image_data_urls.append("data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii"))
            encode_ms = (time.monotonic() - t_enc0) * 1000.0

            payload = _build_payload(
                task=args.task,
                question=args.question,
                model=model,
                planner_prompt=planner_prompt,
                image_data_urls=image_data_urls,
                frame_shapes=[list(np.asarray(f).shape) for f in frames_for_request],
                resized_shapes=resized_shapes,
                frame_age_ms=frame_age_ms,
            )

            ok_http = False
            ok_parse = False
            primitive = None
            confidence = None
            explanation = None
            response_preview = ""
            error = None
            status = 0

            t_http0 = time.monotonic()
            response = post_chat_json(
                url=chat_url,
                payload=payload,
                timeout_s=timeout_s,
                api_key=api_key,
            )
            if response.ok and isinstance(response.response_json, dict):
                status = response.status_code
                ok_http = True
                content, _reasoning = extract_message_content(response.response_json)
                if content:
                    response_preview = content.strip().replace("\n", "\\n")
                    if args.task == "describe":
                        ok_parse = True
                        explanation = response_preview
                    else:
                        parsed = _parse_action_content(content)
                        if isinstance(parsed, dict):
                            ok_parse = True
                            primitive = str(parsed.get("primitive")) if parsed.get("primitive") is not None else None
                            try:
                                confidence = (
                                    float(parsed.get("confidence"))
                                    if parsed.get("confidence") is not None
                                    else None
                                )
                            except Exception:
                                confidence = None
                            explanation = (
                                str(parsed.get("explanation"))
                                if parsed.get("explanation") is not None
                                else None
                            )
                else:
                    response_preview = "<empty>"
            else:
                error = response.error or "request failed"
                response_preview = error
            http_ms = (time.monotonic() - t_http0) * 1000.0
            total_ms = (time.monotonic() - t0) * 1000.0

            encode_ms_vals.append(encode_ms)
            http_ms_vals.append(http_ms)
            total_ms_vals.append(total_ms)
            if ok_http:
                ok_http_count += 1
            if ok_parse:
                ok_parse_count += 1

            result = ProbeResult(
                idx=idx,
                ok_http=ok_http,
                ok_parse=ok_parse,
                encode_ms=encode_ms,
                http_ms=http_ms,
                total_ms=total_ms,
                status=status,
                primitive=primitive,
                confidence=confidence,
                explanation=explanation,
                response_preview=response_preview[:300],
                frame_shape=frame_shape,
                resized_shape=resized_shapes[-1] if resized_shapes else None,
                jpeg_bytes=total_jpeg_bytes,
                frames_sent=len(image_data_urls),
                error=error,
            )

            if out_fh is not None:
                out_fh.write(json.dumps(result.__dict__, separators=(",", ":"), ensure_ascii=True) + "\n")
                out_fh.flush()

            conf_str = "-" if confidence is None else f"{confidence:.2f}"
            prim_str = primitive or "-"
            parse_str = "ok" if ok_parse else "no"
            http_str = "ok" if ok_http else "err"
            print(
                f"[{idx:03d}] http={http_str} parse={parse_str} total={total_ms:7.1f}ms "
                f"enc={encode_ms:6.1f}ms net={http_ms:7.1f}ms prim={prim_str:<12} conf={conf_str} "
                f"frames={len(image_data_urls):2d} jpeg={total_jpeg_bytes:6d}B"
            )
            if (args.print_response or args.task == "describe") and response_preview:
                print(f"      response={response_preview[:240]}")

    except KeyboardInterrupt:
        print("cosmos_image_probe: interrupted")
    finally:
        if out_fh is not None:
            out_fh.close()
        if is_threaded:
            stats = reader.stats()
            reader.shutdown()
            print(
                "capture_stats:"
                f" captured={stats['captured_count']}"
                f" dropped={stats['dropped_count']}"
                f" last_error={stats['last_error']}"
            )
        else:
            base_source.shutdown()

    n = len(total_ms_vals)
    if n == 0:
        print("cosmos_image_probe: no requests completed")
        return 1

    print("summary:")
    print(f"  requests={n} http_ok={ok_http_count} parse_ok={ok_parse_count}")
    print(
        "  total_ms:"
        f" avg={sum(total_ms_vals)/n:.1f} p50={_percentile(total_ms_vals,50):.1f} p95={_percentile(total_ms_vals,95):.1f}"
    )
    print(
        "  http_ms:"
        f" avg={sum(http_ms_vals)/n:.1f} p50={_percentile(http_ms_vals,50):.1f} p95={_percentile(http_ms_vals,95):.1f}"
    )
    print(
        "  encode_ms:"
        f" avg={sum(encode_ms_vals)/n:.1f} p50={_percentile(encode_ms_vals,50):.1f} p95={_percentile(encode_ms_vals,95):.1f}"
    )
    if args.out:
        print(f"  jsonl={args.out}")
    return 0 if ok_parse_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
