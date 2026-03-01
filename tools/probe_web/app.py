from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pala.behavior.decision_schema_v4 import behavior_decision_response_format
from pala.behavior.prompts import build_behavior_v4_user_text

from .defaults import resolve_defaults
from .models import BehaviorProbeRun
from .service import (
    ProbeFsmSimulator,
    ProbeInputError,
    load_run_for_ui,
    normalize_params,
    parse_force_mode,
    parse_signals_form,
    run_behavior_probe,
)
from .storage import ensure_logs_root, list_recent_runs


def _scenario_presets(defaults: Any) -> List[Dict[str, Any]]:
    return [
        {
            "id": "social_greeting",
            "title": "Social Greeting",
            "description": "Boot -> social interaction with a confident nearby person.",
            "script": [
                {"op": "reset", "now_mono_s": 0.0},
                {
                    "op": "step",
                    "advance_s": 2.0,
                    "signals": {
                        "startup_complete": True,
                        "person_present": True,
                        "person_conf": 0.85,
                        "search_requested": False,
                        "search_complete": False,
                        "task_active": False,
                        "health_degraded": False,
                    },
                },
                {
                    "op": "step",
                    "advance_s": 1.4,
                    "signals": {
                        "startup_complete": True,
                        "person_present": True,
                        "person_conf": 0.85,
                        "search_requested": False,
                        "search_complete": False,
                        "task_active": False,
                        "health_degraded": False,
                    },
                },
            ],
            "form_overrides": {
                "planner_prompt": "Acknowledge the user and keep motion warm, clear, and non-aggressive.",
                "policy_style": "Prefer curious style for social engagement; keep movements smooth and safe.",
                "temperature": "0.0",
                "top_p": "1.0",
                "max_tokens": str(max(1000, int(defaults.max_tokens))),
            },
            "fsm_form_values": {
                "fsm_advance_s": "1.4",
                "person_present": "true",
                "person_conf": "0.85",
                "search_requested": "false",
                "search_complete": "false",
                "task_active": "false",
                "startup_complete": "true",
                "health_degraded": "false",
                "force_mode": "",
                "force_mode_reason": "manual_override",
                "now_mono_s": "0",
            },
        },
        {
            "id": "search_assist",
            "title": "Search Assist",
            "description": "Boot -> idle -> search_assist with active search request.",
            "script": [
                {"op": "reset", "now_mono_s": 0.0},
                {
                    "op": "step",
                    "advance_s": 2.0,
                    "signals": {
                        "startup_complete": True,
                        "person_present": True,
                        "person_conf": 0.70,
                        "search_requested": True,
                        "search_complete": False,
                        "task_active": False,
                        "health_degraded": False,
                    },
                },
                {
                    "op": "step",
                    "advance_s": 1.5,
                    "signals": {
                        "startup_complete": True,
                        "person_present": True,
                        "person_conf": 0.70,
                        "search_requested": True,
                        "search_complete": False,
                        "task_active": False,
                        "health_degraded": False,
                    },
                },
            ],
            "form_overrides": {
                "planner_prompt": "Focus on search-oriented behaviors and clear directional actions.",
                "policy_style": "Prefer focused style while search is active.",
                "temperature": "0.0",
                "top_p": "1.0",
                "max_tokens": str(max(1000, int(defaults.max_tokens))),
            },
            "fsm_form_values": {
                "fsm_advance_s": "1.5",
                "person_present": "true",
                "person_conf": "0.70",
                "search_requested": "true",
                "search_complete": "false",
                "task_active": "false",
                "startup_complete": "true",
                "health_degraded": "false",
                "force_mode": "",
                "force_mode_reason": "manual_override",
                "now_mono_s": "0",
            },
        },
        {
            "id": "recover_reset",
            "title": "Recover Reset",
            "description": "Force degraded health path into recover_reset mode.",
            "script": [
                {"op": "reset", "now_mono_s": 0.0},
                {
                    "op": "step",
                    "advance_s": 0.4,
                    "signals": {
                        "startup_complete": True,
                        "person_present": False,
                        "person_conf": 0.0,
                        "search_requested": False,
                        "search_complete": False,
                        "task_active": False,
                        "health_degraded": True,
                    },
                },
            ],
            "form_overrides": {
                "planner_prompt": "Prioritize conservative recovery behavior and safe reset actions.",
                "policy_style": "Use calm style and avoid sudden motion during degraded health.",
                "temperature": "0.0",
                "top_p": "1.0",
                "max_tokens": str(max(1000, int(defaults.max_tokens))),
            },
            "fsm_form_values": {
                "fsm_advance_s": "0.4",
                "person_present": "false",
                "person_conf": "0.0",
                "search_requested": "false",
                "search_complete": "false",
                "task_active": "false",
                "startup_complete": "true",
                "health_degraded": "true",
                "force_mode": "",
                "force_mode_reason": "manual_override",
                "now_mono_s": "0",
            },
        },
    ]


def _apply_scenario_preset(*, simulator: ProbeFsmSimulator, preset: Dict[str, Any]) -> Dict[str, Any]:
    state = simulator.fsm_snapshot()
    script = preset.get("script")
    if not isinstance(script, list):
        return state
    for step in script:
        if not isinstance(step, dict):
            continue
        op = str(step.get("op", "")).strip().lower()
        if op == "reset":
            now = float(step.get("now_mono_s", 0.0))
            state = simulator.reset(now_mono_s=max(0.0, now))
            continue
        if op == "step":
            signals_raw = step.get("signals")
            if not isinstance(signals_raw, dict):
                continue
            signals = parse_signals_form(signals_raw)
            advance_s = max(0.0, float(step.get("advance_s", 0.0)))
            state = simulator.step(signals=signals, advance_s=advance_s)
            continue
        if op == "force":
            mode_token = str(step.get("mode", "")).strip()
            next_mode = parse_force_mode(mode_token)
            if next_mode is None:
                continue
            reason = str(step.get("reason", "preset_force")).strip() or "preset_force"
            advance_s = max(0.0, float(step.get("advance_s", 0.0)))
            state = simulator.force_mode(next_mode=next_mode, reason=reason, advance_s=advance_s)
    return state


def _preview_override_defaults(*, defaults: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    user_text = build_behavior_v4_user_text(
        context=context,
        policy_identity=defaults.policy_identity,
        policy_capabilities=defaults.policy_capabilities,
        policy_safety=defaults.policy_safety,
        policy_style=defaults.policy_style,
        planner_prompt=defaults.planner_prompt,
    )
    payload = {
        "model": defaults.model,
        "messages": [
            {"role": "system", "content": defaults.system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ],
        "temperature": defaults.temperature,
        "top_p": defaults.top_p,
        "presence_penalty": defaults.presence_penalty,
        "max_tokens": defaults.max_tokens,
        "stream": False,
        "response_format": behavior_decision_response_format(),
    }
    return {
        "context_override_json": json.dumps(context, ensure_ascii=True, indent=2),
        "user_text_override": user_text,
        "payload_override_json": json.dumps(payload, ensure_ascii=True, indent=2),
    }


def _default_form(defaults: Any, *, preview_overrides: Dict[str, str]) -> Dict[str, Any]:
    return {
        "provider": defaults.provider,
        "model": defaults.model,
        "base_url": defaults.base_url,
        "system_prompt": defaults.system_prompt,
        "timeout_s": defaults.timeout_s,
        "max_tokens": defaults.max_tokens,
        "temperature": defaults.temperature,
        "top_p": defaults.top_p,
        "presence_penalty": defaults.presence_penalty,
        "frame_max_width": defaults.frame_max_width,
        "frame_jpeg_quality": defaults.frame_jpeg_quality,
        "policy_identity": defaults.policy_identity,
        "policy_capabilities": defaults.policy_capabilities,
        "policy_safety": defaults.policy_safety,
        "policy_style": defaults.policy_style,
        "planner_prompt": defaults.planner_prompt,
        "context_override_json": preview_overrides.get("context_override_json", defaults.context_override_json),
        "user_text_override": preview_overrides.get("user_text_override", defaults.user_text_override),
        "payload_override_json": preview_overrides.get("payload_override_json", defaults.payload_override_json),
        "inter_frame_ms": defaults.inter_frame_ms,
        "packet_view_mode": defaults.packet_view_mode,
        "fsm_advance_s": "0.5",
        "person_present": "false",
        "person_conf": "0.0",
        "search_requested": "false",
        "search_complete": "false",
        "task_active": "false",
        "startup_complete": "true",
        "health_degraded": "false",
        "force_mode": "",
        "force_mode_reason": "manual_override",
        "now_mono_s": "0",
        "compare_run_a_id": "",
        "compare_run_b_id": "",
    }


def _template_context(
    *,
    request: Request,
    defaults: Any,
    recent_runs: List[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    fsm_state: Dict[str, Any],
    preview_overrides: Dict[str, str],
    scenario_presets: Optional[List[Dict[str, Any]]] = None,
    preset_status: Optional[str] = None,
    compare_result: Optional[Dict[str, Any]] = None,
    compare_error_message: Optional[str] = None,
    error_message: Optional[str] = None,
    form_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    values = _default_form(defaults, preview_overrides=preview_overrides)
    if isinstance(form_values, dict):
        values.update(form_values)
    return {
        "request": request,
        "defaults": defaults,
        "recent_runs": recent_runs,
        "result": result,
        "fsm_state": fsm_state,
        "scenario_presets": scenario_presets or [],
        "preset_status": preset_status,
        "compare_result": compare_result,
        "compare_error_message": compare_error_message,
        "error_message": error_message,
        "form_values": values,
    }


def _run_to_view(run: BehaviorProbeRun) -> Dict[str, Any]:
    out = run.to_dict()
    out["summary"] = {
        "run_id": run.run_id,
        "created_at_utc": run.created_at_utc,
        "mode": run.mode,
        "parse_ok": run.parse_ok,
        "http_status": run.response_meta.get("http_status"),
        "latency_ms": run.response_meta.get("latency_ms"),
        "guard_reason": (run.guard_result or {}).get("reason"),
    }
    return out


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compare_runs(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_meta = left.get("response_meta") if isinstance(left.get("response_meta"), dict) else {}
    right_meta = right.get("response_meta") if isinstance(right.get("response_meta"), dict) else {}
    left_guard = left.get("guard_result") if isinstance(left.get("guard_result"), dict) else {}
    right_guard = right.get("guard_result") if isinstance(right.get("guard_result"), dict) else {}
    left_final = left.get("final_action") if isinstance(left.get("final_action"), dict) else {}
    right_final = right.get("final_action") if isinstance(right.get("final_action"), dict) else {}

    latency_a = _to_float(left_meta.get("latency_ms"))
    latency_b = _to_float(right_meta.get("latency_ms"))
    total_a = _to_int(left_meta.get("total_tokens"))
    total_b = _to_int(right_meta.get("total_tokens"))

    return {
        "left": {
            "run_id": left.get("run_id"),
            "created_at_utc": left.get("created_at_utc"),
            "mode": left.get("mode"),
            "parse_ok": left.get("parse_ok"),
            "parse_stage": left.get("parse_stage"),
            "parse_error": left.get("parse_error"),
            "guard_reason": left_guard.get("reason"),
            "guard_used_fallback": left_guard.get("used_fallback"),
            "final_primitive": left_final.get("primitive"),
            "response_meta": left_meta,
            "parsed_output": left.get("parsed_output"),
            "effective_context": (left.get("effective_inputs") or {}).get("context_effective"),
        },
        "right": {
            "run_id": right.get("run_id"),
            "created_at_utc": right.get("created_at_utc"),
            "mode": right.get("mode"),
            "parse_ok": right.get("parse_ok"),
            "parse_stage": right.get("parse_stage"),
            "parse_error": right.get("parse_error"),
            "guard_reason": right_guard.get("reason"),
            "guard_used_fallback": right_guard.get("used_fallback"),
            "final_primitive": right_final.get("primitive"),
            "response_meta": right_meta,
            "parsed_output": right.get("parsed_output"),
            "effective_context": (right.get("effective_inputs") or {}).get("context_effective"),
        },
        "highlights": [
            {
                "label": "Parse Outcome",
                "left": "ok" if bool(left.get("parse_ok")) else "fail",
                "right": "ok" if bool(right.get("parse_ok")) else "fail",
                "delta": "same" if bool(left.get("parse_ok")) == bool(right.get("parse_ok")) else "changed",
            },
            {
                "label": "Latency (ms)",
                "left": latency_a,
                "right": latency_b,
                "delta": None if latency_a is None or latency_b is None else round(latency_b - latency_a, 1),
            },
            {
                "label": "Total Tokens",
                "left": total_a,
                "right": total_b,
                "delta": None if total_a is None or total_b is None else (total_b - total_a),
            },
            {
                "label": "Guard Reason",
                "left": left_guard.get("reason"),
                "right": right_guard.get("reason"),
                "delta": "same"
                if str(left_guard.get("reason") or "") == str(right_guard.get("reason") or "")
                else "changed",
            },
            {
                "label": "Final Primitive",
                "left": left_final.get("primitive"),
                "right": right_final.get("primitive"),
                "delta": "same"
                if str(left_final.get("primitive") or "") == str(right_final.get("primitive") or "")
                else "changed",
            },
        ],
    }


def create_app(*, logs_root: Path | str = Path("logs/probe_web_v4")) -> FastAPI:
    module_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(module_dir / "templates"))

    logs_root_path = ensure_logs_root(Path(logs_root))
    defaults_init = resolve_defaults()
    simulator = ProbeFsmSimulator(defaults=defaults_init)

    app = FastAPI(title="PALA Probe Web", version="0.2.0")
    app.mount("/probe-web-static", StaticFiles(directory=str(module_dir / "static")), name="probe_web_static")
    app.mount("/probe-logs", StaticFiles(directory=str(logs_root_path)), name="probe_logs")

    def _preview(defaults: Any) -> Dict[str, str]:
        return _preview_override_defaults(defaults=defaults, context=simulator.build_context())

    def _presets(defaults: Any) -> List[Dict[str, Any]]:
        return _scenario_presets(defaults)

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        defaults = resolve_defaults()
        recent_runs = list_recent_runs(logs_root_path)
        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent_runs,
            result=None,
            fsm_state=simulator.fsm_snapshot(),
            preview_overrides=_preview(defaults),
            scenario_presets=_presets(defaults),
        )
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.get("/runs/recent", response_class=HTMLResponse)
    async def recent_runs(request: Request) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent,
            result=None,
            fsm_state=simulator.fsm_snapshot(),
            preview_overrides=_preview(defaults),
            scenario_presets=_presets(defaults),
        )
        return templates.TemplateResponse(request, "partials/recent_runs.html", ctx)

    @app.get("/status/api-key", response_class=HTMLResponse)
    async def api_key_status(request: Request) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent,
            result=None,
            fsm_state=simulator.fsm_snapshot(),
            preview_overrides=_preview(defaults),
            scenario_presets=_presets(defaults),
        )
        return templates.TemplateResponse(request, "partials/api_key_status.html", ctx)

    @app.get("/probe/v4/overrides/defaults", response_class=HTMLResponse)
    async def override_defaults(request: Request) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent,
            result=None,
            fsm_state=simulator.fsm_snapshot(),
            preview_overrides=_preview(defaults),
            scenario_presets=_presets(defaults),
        )
        return templates.TemplateResponse(request, "partials/override_fields.html", ctx)

    @app.post("/probe/v4/presets/apply", response_class=HTMLResponse)
    async def apply_preset(request: Request, preset_id: str = Form("")) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        catalog = _presets(defaults)
        selected = None
        for item in catalog:
            if str(item.get("id", "")).strip() == str(preset_id).strip():
                selected = item
                break

        if selected is None:
            ctx = _template_context(
                request=request,
                defaults=defaults,
                recent_runs=recent,
                result=None,
                fsm_state=simulator.fsm_snapshot(),
                preview_overrides=_preview(defaults),
                scenario_presets=catalog,
                preset_status=f"Unknown preset '{preset_id}'.",
            )
            return templates.TemplateResponse(request, "partials/scenario_presets.html", ctx)

        state = _apply_scenario_preset(simulator=simulator, preset=selected)
        preview = _preview(defaults)
        form_values = {}
        form_values.update(selected.get("form_overrides") or {})
        form_values.update(selected.get("fsm_form_values") or {})
        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent,
            result=None,
            fsm_state=state,
            preview_overrides=preview,
            scenario_presets=catalog,
            preset_status=f"Applied preset: {selected.get('title', selected.get('id', 'preset'))}",
            form_values=form_values,
        )
        return templates.TemplateResponse(request, "partials/preset_apply.html", ctx)

    @app.get("/probe/v4/fsm/meta", response_class=HTMLResponse)
    async def fsm_meta(request: Request) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent,
            result=None,
            fsm_state=simulator.fsm_snapshot(),
            preview_overrides=_preview(defaults),
            scenario_presets=_presets(defaults),
        )
        return templates.TemplateResponse(request, "partials/fsm_panel.html", ctx)

    @app.post("/probe/v4/fsm/reset", response_class=HTMLResponse)
    async def fsm_reset(request: Request, now_mono_s: str = Form("0")) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        try:
            now = float(now_mono_s)
        except ValueError:
            now = 0.0
        state = simulator.reset(now_mono_s=max(0.0, now))
        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent,
            result=None,
            fsm_state=state,
            preview_overrides=_preview(defaults),
            scenario_presets=_presets(defaults),
            form_values={"now_mono_s": str(max(0.0, now))},
        )
        return templates.TemplateResponse(request, "partials/fsm_panel.html", ctx)

    @app.post("/probe/v4/fsm/step", response_class=HTMLResponse)
    async def fsm_step(
        request: Request,
        fsm_advance_s: str = Form("0.5"),
        person_present: str = Form("false"),
        person_conf: str = Form("0.0"),
        search_requested: str = Form("false"),
        search_complete: str = Form("false"),
        task_active: str = Form("false"),
        startup_complete: str = Form("true"),
        health_degraded: str = Form("false"),
        force_mode: str = Form(""),
        force_mode_reason: str = Form("manual_override"),
    ) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)

        raw = {
            "person_present": person_present,
            "person_conf": person_conf,
            "search_requested": search_requested,
            "search_complete": search_complete,
            "task_active": task_active,
            "startup_complete": startup_complete,
            "health_degraded": health_degraded,
        }
        try:
            advance_s = float(fsm_advance_s)
        except ValueError:
            advance_s = 0.5

        maybe_force = parse_force_mode(force_mode)
        if maybe_force is not None:
            state = simulator.force_mode(
                next_mode=maybe_force,
                reason=(str(force_mode_reason or "manual_override").strip() or "manual_override"),
                advance_s=max(0.0, advance_s),
            )
        else:
            signals = parse_signals_form(raw)
            state = simulator.step(signals=signals, advance_s=max(0.0, advance_s))

        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent,
            result=None,
            fsm_state=state,
            preview_overrides=_preview(defaults),
            scenario_presets=_presets(defaults),
            form_values={
                "fsm_advance_s": fsm_advance_s,
                "person_present": person_present,
                "person_conf": person_conf,
                "search_requested": search_requested,
                "search_complete": search_complete,
                "task_active": task_active,
                "startup_complete": startup_complete,
                "health_degraded": health_degraded,
                "force_mode": force_mode,
                "force_mode_reason": force_mode_reason,
            },
        )
        return templates.TemplateResponse(request, "partials/fsm_panel.html", ctx)

    @app.post("/compare/runs", response_class=HTMLResponse)
    async def compare_runs(request: Request, compare_run_a_id: str = Form(""), compare_run_b_id: str = Form("")) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        run_a_id = str(compare_run_a_id or "").strip()
        run_b_id = str(compare_run_b_id or "").strip()

        compare_result = None
        compare_error_message = None
        if not run_a_id or not run_b_id:
            compare_error_message = "Select two runs to compare."
        elif run_a_id == run_b_id:
            compare_error_message = "Select two different runs for a useful comparison."
        else:
            loaded_a = load_run_for_ui(logs_root_path, run_a_id)
            loaded_b = load_run_for_ui(logs_root_path, run_b_id)
            if loaded_a is None or loaded_b is None:
                compare_error_message = "One or both selected runs could not be loaded."
            else:
                compare_result = _compare_runs(loaded_a, loaded_b)

        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent,
            result=None,
            fsm_state=simulator.fsm_snapshot(),
            preview_overrides=_preview(defaults),
            scenario_presets=_presets(defaults),
            compare_result=compare_result,
            compare_error_message=compare_error_message,
            form_values={
                "compare_run_a_id": run_a_id,
                "compare_run_b_id": run_b_id,
            },
        )
        return templates.TemplateResponse(request, "partials/compare_panel.html", ctx)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        defaults = resolve_defaults()
        loaded = load_run_for_ui(logs_root_path, run_id)
        if loaded is None:
            raise HTTPException(status_code=404, detail="Run not found")
        recent = list_recent_runs(logs_root_path)
        params = loaded.get("params") if isinstance(loaded.get("params"), dict) else {}
        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent,
            result=loaded,
            fsm_state=simulator.fsm_snapshot(),
            preview_overrides=_preview(defaults),
            scenario_presets=_presets(defaults),
            form_values=params,
        )
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.post("/probe/v4/run", response_class=HTMLResponse)
    async def probe_v4(
        request: Request,
        provider: str = Form(""),
        model: str = Form(""),
        base_url: str = Form(""),
        system_prompt: str = Form(""),
        timeout_s: str = Form(""),
        max_tokens: str = Form(""),
        temperature: str = Form(""),
        top_p: str = Form(""),
        presence_penalty: str = Form(""),
        frame_max_width: str = Form(""),
        frame_jpeg_quality: str = Form(""),
        policy_identity: str = Form(""),
        policy_capabilities: str = Form(""),
        policy_safety: str = Form(""),
        policy_style: str = Form(""),
        planner_prompt: str = Form(""),
        context_override_json: str = Form(""),
        user_text_override: str = Form(""),
        payload_override_json: str = Form(""),
        inter_frame_ms: str = Form(""),
        packet_view_mode: str = Form("expanded"),
        image_order: str = Form(""),
        images: List[UploadFile] = File(default=[]),
    ) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        raw_form = {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "system_prompt": system_prompt,
            "timeout_s": timeout_s,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frame_max_width": frame_max_width,
            "frame_jpeg_quality": frame_jpeg_quality,
            "policy_identity": policy_identity,
            "policy_capabilities": policy_capabilities,
            "policy_safety": policy_safety,
            "policy_style": policy_style,
            "planner_prompt": planner_prompt,
            "context_override_json": context_override_json,
            "user_text_override": user_text_override,
            "payload_override_json": payload_override_json,
            "inter_frame_ms": inter_frame_ms,
            "packet_view_mode": packet_view_mode,
        }

        try:
            params = normalize_params(raw_form, defaults=defaults)
            run = await run_behavior_probe(
                params=params,
                upload_files=images,
                image_order=image_order,
                logs_root=logs_root_path,
                simulator=simulator,
            )
            result = _run_to_view(run)
            recent = list_recent_runs(logs_root_path)
            ctx = _template_context(
                request=request,
                defaults=defaults,
                recent_runs=recent,
                result=result,
                fsm_state=simulator.fsm_snapshot(),
                preview_overrides=_preview(defaults),
                scenario_presets=_presets(defaults),
                form_values=asdict(params),
            )
            return templates.TemplateResponse(request, "partials/result_panel.html", ctx)
        except ProbeInputError as exc:
            ctx = _template_context(
                request=request,
                defaults=defaults,
                recent_runs=recent,
                result=None,
                fsm_state=simulator.fsm_snapshot(),
                preview_overrides=_preview(defaults),
                scenario_presets=_presets(defaults),
                error_message=str(exc),
                form_values=raw_form,
            )
            return templates.TemplateResponse(request, "partials/result_panel.html", ctx)

    return app


app = create_app()
