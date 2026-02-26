#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Dict, List, Optional
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pala.behavior.intent_proposer import parse_intent_proposer_response
from pala.behavior.json_parse import parse_json_flexible
from pala.behavior.model_clients import extract_message_content, normalize_chat_url, post_chat_json
from pala.behavior.prompts import build_messages, build_planner_user_text
from pala.behavior.schemas import intent_response_format
from pala.config import load_config


@dataclass
class ProbeSample:
    ok: bool
    parse_ok: bool
    latency_ms: float
    status_code: int
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    error: Optional[str]
    parse_error: Optional[str]
    response_preview: Optional[str]


def _preview_text(value: Optional[str], *, max_chars: int = 180) -> Optional[str]:
    if value is None:
        return None
    token = " ".join(str(value).split()).strip()
    if not token:
        return None
    if len(token) <= max_chars:
        return token
    return token[: max_chars - 3] + "..."


def _response_meta(response_json: Dict[str, Any]) -> tuple[Optional[str], Optional[int], Optional[int]]:
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    choices = response_json.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason_raw = choices[0].get("finish_reason")
        if isinstance(finish_reason_raw, str):
            finish_reason = finish_reason_raw
    usage = response_json.get("usage")
    if isinstance(usage, dict):
        p = usage.get("prompt_tokens")
        c = usage.get("completion_tokens")
        if isinstance(p, int):
            prompt_tokens = p
        if isinstance(c, int):
            completion_tokens = c
    return finish_reason, prompt_tokens, completion_tokens


def _safe_json_parse(content: Optional[str]) -> tuple[bool, Optional[str]]:
    if content is None:
        return False, "missing_message_content"
    obj, err, _stage = parse_json_flexible(content)
    if obj is None:
        return False, err or "json_parse_failed"
    if not isinstance(obj, dict):
        return False, "json_root_not_object"
    return True, None


def _planner_parse(content: Optional[str]) -> tuple[bool, Optional[str]]:
    if content is None:
        return False, "missing_message_content"
    parsed = parse_intent_proposer_response(content)
    if parsed is None:
        return False, "planner_parse_failed"
    if not parsed.response.proposals:
        return False, "planner_no_proposals"
    return True, None


def _run_case(
    *,
    name: str,
    url: str,
    provider: str,
    api_key: Optional[str],
    payload_builder: Callable[[], Dict[str, Any]],
    parse_fn: Callable[[Optional[str]], tuple[bool, Optional[str]]],
    timeout_s: float,
    runs: int,
    sleep_s: float,
    raw_dir: Path,
    verbose: bool,
) -> Dict[str, Any]:
    samples: List[ProbeSample] = []
    for idx in range(runs):
        payload = payload_builder()
        result = post_chat_json(
            url=url,
            payload=payload,
            timeout_s=timeout_s,
            api_key=api_key,
            provider=provider,
        )

        parse_ok = False
        parse_error: Optional[str] = None
        content: Optional[str] = None
        finish_reason: Optional[str] = None
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        if result.ok and result.response_json is not None:
            finish_reason, prompt_tokens, completion_tokens = _response_meta(result.response_json)
            content, _reasoning = extract_message_content(result.response_json)
            parse_ok, parse_error = parse_fn(content)
            raw_path = raw_dir / f"{name}_{idx + 1}.json"
            raw_path.write_text(json.dumps(result.response_json, indent=2), encoding="utf-8")
        else:
            parse_error = result.error or "request_failed"

        sample = ProbeSample(
            ok=bool(result.ok),
            parse_ok=parse_ok,
            latency_ms=round(float(result.latency_ms), 1),
            status_code=int(result.status_code),
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=result.error,
            parse_error=parse_error,
            response_preview=_preview_text(content),
        )
        samples.append(sample)
        if verbose:
            print(
                f"  - {name} [{idx + 1}/{runs}] ok={sample.ok} parse_ok={sample.parse_ok} "
                f"latency_ms={sample.latency_ms} status={sample.status_code} "
                f"finish={sample.finish_reason} err={sample.error or sample.parse_error or '-'}"
            )
        if sleep_s > 0 and idx < (runs - 1):
            time.sleep(float(sleep_s))

    latencies = [s.latency_ms for s in samples]
    p50 = round(statistics.median(latencies), 1) if latencies else None
    p95 = round(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 1) if latencies else None
    return {
        "name": name,
        "runs": runs,
        "ok_count": sum(1 for s in samples if s.ok),
        "parse_ok_count": sum(1 for s in samples if s.parse_ok),
        "latency_ms": {
            "p50": p50,
            "p95": p95,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "samples": [asdict(s) for s in samples],
    }


def _planner_payload(model: str, *, max_tokens: int, provider: str) -> Dict[str, Any]:
    context = {
        "current_action": {
            "primitive": "hold",
            "command": {},
            "style": "calm",
            "confidence": 0.2,
            "age_s": 0.4,
        },
        "signals": {
            "person_conf": 0.7,
            "zone_hint": "left",
            "env_delta": 0.4,
            "activity_level": 0.6,
            "novelty": 0.3,
            "person_present": True,
        },
        "latest_env": {
            "scene": "I see the scene as a desk with a user to my left.",
            "summary": "user present and shifting posture",
        },
        "control_state": "active_kind=hold",
        "planner_health": {"state": "HEALTHY"},
        "anti_collapse": {"no_commit_s": 1.0},
        "evidence_index": {"available": ["frame:latest", "env:latest", "perception:zone:left"]},
    }
    user_text = build_planner_user_text(
        context=context,
        policy_identity="You are PALA.",
        policy_capabilities="Use hold, breath, glance, nod, orient_to_zone.",
        policy_safety="Keep movement safe and calm.",
        policy_style="Use calm by default.",
        planner_prompt="Return concrete next action proposals.",
        max_proposals=3,
    )
    return {
        "model": model,
        "messages": build_messages(user_text=user_text, image_data_urls=[]),
        "temperature": 0.0,
        "top_p": 0.3,
        "presence_penalty": 0.0,
        "max_tokens": int(max_tokens),
        "stream": False,
        "response_format": intent_response_format(),
    }


def _json_schema_payload(model: str, *, max_tokens: int, strict: bool, provider: str) -> Dict[str, Any]:
    provider_token = str(provider or "auto").strip().lower()
    if provider_token == "gemini":
        # Gemini OpenAI-compat is more reliable with json_object than json_schema strict.
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a deterministic JSON generator."},
                {"role": "user", "content": "Return only JSON object: {\"ok\":true,\"note\":\"...\"}"},
            ],
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
            "stream": False,
            "response_format": {"type": "json_object"},
        }

    response_format: Dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "probe_status",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ok", "note"],
                "properties": {
                    "ok": {"type": "boolean"},
                    "note": {"type": "string", "minLength": 1, "maxLength": 80},
                },
            },
        },
    }
    if strict:
        response_format["json_schema"]["strict"] = True
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a deterministic JSON generator."},
            {"role": "user", "content": "Return valid JSON with ok=true and short note."},
        ],
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "stream": False,
        "response_format": response_format,
    }


def _text_payload(model: str, *, max_tokens: int) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with READY only."},
        ],
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "stream": False,
    }


def _mask_secret(token: Optional[str]) -> str:
    if not token:
        return "(empty)"
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe provider endpoint compatibility and latency for PALA.")
    parser.add_argument("--config", default="config/robot.yaml", help="Path to robot config.")
    parser.add_argument("--provider", choices=["auto", "openai", "cosmos", "gemini"], default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--runs", type=int, default=1, help="Requests per probe case.")
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Delay between requests in each case.")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--planner-max-tokens", type=int, default=520)
    parser.add_argument("--no-planner", action="store_true", help="Skip planner-schema probe case.")
    parser.add_argument("--schema-no-strict", action="store_true", help="Use json_schema probe without strict=true.")
    parser.add_argument("--out-root", default="logs/provider_probe")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    cosmos = cfg.cosmos

    provider = (
        args.provider
        or os.getenv("PALA_MODEL_PROVIDER")
        or os.getenv("PALA_COSMOS_PROVIDER")
        or getattr(cosmos, "provider", "auto")
        or "auto"
    )
    base_url = args.base_url or os.getenv("PALA_COSMOS_BASE_URL") or cosmos.base_url
    api_key = args.api_key or os.getenv("PALA_COSMOS_API_KEY")
    model = args.model or os.getenv("PALA_COSMOS_MODEL") or cosmos.model
    if not base_url:
        raise SystemExit("Missing base URL. Set --base-url or PALA_COSMOS_BASE_URL.")
    if not model:
        raise SystemExit("Missing model. Set --model or PALA_COSMOS_MODEL.")

    resolved_chat_url = normalize_chat_url(str(base_url), provider=str(provider))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Provider Probe")
    print(f"- provider: {provider}")
    print(f"- base_url_in: {base_url}")
    print(f"- chat_url_resolved: {resolved_chat_url}")
    print(f"- model: {model}")
    print(f"- api_key: {_mask_secret(api_key)}")
    print(f"- runs_per_case: {max(1, int(args.runs))}")
    print(f"- timeout_s: {float(args.timeout_s)}")

    runs = max(1, int(args.runs))
    results: List[Dict[str, Any]] = []
    results.append(
        _run_case(
            name="text_ping",
            url=str(base_url),
            provider=str(provider),
            api_key=api_key,
            payload_builder=lambda: _text_payload(model=str(model), max_tokens=max(16, int(args.max_tokens))),
            parse_fn=lambda content: (bool(content and "READY" in content.upper()), None if content else "missing_content"),
            timeout_s=float(args.timeout_s),
            runs=runs,
            sleep_s=float(args.sleep_s),
            raw_dir=raw_dir,
            verbose=args.verbose,
        )
    )
    results.append(
        _run_case(
            name="json_schema_probe",
            url=str(base_url),
            provider=str(provider),
            api_key=api_key,
            payload_builder=lambda: _json_schema_payload(
                model=str(model),
                max_tokens=max(64, int(args.max_tokens)),
                strict=not bool(args.schema_no_strict),
                provider=str(provider),
            ),
            parse_fn=_safe_json_parse,
            timeout_s=float(args.timeout_s),
            runs=runs,
            sleep_s=float(args.sleep_s),
            raw_dir=raw_dir,
            verbose=args.verbose,
        )
    )
    if not args.no_planner:
        results.append(
            _run_case(
                name="planner_schema_probe",
                url=str(base_url),
                provider=str(provider),
                api_key=api_key,
                payload_builder=lambda: _planner_payload(
                    model=str(model),
                    max_tokens=max(320, int(args.planner_max_tokens)),
                    provider=str(provider),
                ),
                parse_fn=_planner_parse,
                timeout_s=float(args.timeout_s),
                runs=runs,
                sleep_s=float(args.sleep_s),
                raw_dir=raw_dir,
                verbose=args.verbose,
            )
        )

    report = {
        "run_id": run_id,
        "provider": provider,
        "base_url": str(base_url),
        "chat_url_resolved": resolved_chat_url,
        "model": str(model),
        "timeout_s": float(args.timeout_s),
        "runs_per_case": runs,
        "sleep_s": float(args.sleep_s),
        "cases": results,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nResults")
    quota_hits = 0
    for case in results:
        quota_hits += sum(1 for sample in case.get("samples", []) if int(sample.get("status_code") or 0) == 429)
        print(
            f"- {case['name']}: ok={case['ok_count']}/{case['runs']} "
            f"parse_ok={case['parse_ok_count']}/{case['runs']} "
            f"latency_ms(p50/p95/max)="
            f"{case['latency_ms']['p50']}/{case['latency_ms']['p95']}/{case['latency_ms']['max']}"
        )
    if quota_hits > 0:
        print(
            f"\nNote: observed {quota_hits} HTTP 429 responses. "
            "Lower --runs, add --sleep-s, or use a key/project with higher quota."
        )
    print(f"\nSaved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
