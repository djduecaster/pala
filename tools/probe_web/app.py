from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .defaults import resolve_defaults
from .models import EnvProbeRun
from .service import (
    ProbeInputError,
    load_run_for_ui,
    normalize_chain_params,
    normalize_params,
    run_env_then_preview_planner,
    run_planner_from_prepared_env,
    run_env_planner_probe,
    run_env_probe,
)
from .storage import ensure_logs_root, list_recent_runs


def _template_context(
    *,
    request: Request,
    defaults: Any,
    recent_runs: List[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    error_message: Optional[str] = None,
    form_values: Optional[Dict[str, Any]] = None,
    active_mode: str = "env",
) -> Dict[str, Any]:
    default_form = {
        "probe_mode": active_mode,
        "run_action": "run_env" if active_mode == "env" else "run_both",
        "prepared_env_run_id": "",
        "provider": defaults.provider,
        "model": defaults.model,
        "base_url": defaults.base_url,
        "system_prompt": defaults.system_prompt,
        "env_contract": defaults.env_contract,
        "policy_identity": defaults.policy_identity,
        "policy_capabilities": defaults.policy_capabilities,
        "policy_safety": defaults.policy_safety,
        "policy_style": defaults.policy_style,
        "timeout_s": defaults.timeout_s,
        "env_max_tokens": defaults.env_max_tokens,
        "temperature": defaults.temperature,
        "top_p": defaults.top_p,
        "presence_penalty": defaults.presence_penalty,
        "planner_prompt_override": defaults.planner_prompt_override,
        "planner_prompt": defaults.planner_prompt,
        "planner_max_proposals": defaults.planner_max_proposals,
        "planner_use_env_context": "true" if defaults.planner_use_env_context else "false",
        "planner_max_tokens": defaults.planner_max_tokens,
        "planner_temperature": defaults.planner_temperature,
        "planner_top_p": defaults.planner_top_p,
        "planner_presence_penalty": defaults.planner_presence_penalty,
        "planner_system_prompt": defaults.planner_system_prompt,
        "planner_image_indices": defaults.planner_image_indices,
        "planner_context_override_json": defaults.planner_context_override_json,
        "planner_user_text_override": defaults.planner_user_text_override,
        "planner_payload_override_json": defaults.planner_payload_override_json,
        "inter_frame_ms": defaults.inter_frame_ms,
        "packet_view_mode": defaults.packet_view_mode,
    }
    if isinstance(form_values, dict):
        default_form.update(form_values)

    resolved_mode = str(default_form.get("probe_mode", active_mode)).strip().lower()
    if resolved_mode not in {"env", "env_planner"}:
        resolved_mode = "env"

    return {
        "request": request,
        "defaults": defaults,
        "recent_runs": recent_runs,
        "result": result,
        "error_message": error_message,
        "form_values": default_form,
        "active_mode": resolved_mode,
    }


def _run_to_view(run: EnvProbeRun) -> Dict[str, Any]:
    out = run.to_dict()
    out.update(
        {
        "run_id": run.run_id,
        "created_at_utc": run.created_at_utc,
        "mode": run.mode,
        "chain_status": run.chain_status,
        "run_config": run.params,
        "inputs_manifest": run.images,
        "packet_compact": run.packet_compact,
        "packet_expanded": run.packet_expanded,
        "message_structure": run.message_structure,
        "response_meta": run.response_meta,
        "raw_content": run.raw_content,
        "reasoning_content": run.reasoning_content,
        "parse_ok": run.parse_ok,
        "parse_stage": run.parse_stage,
        "parse_error": run.parse_error,
        "parsed_output": run.parsed_output,
        "summary": {
            "run_id": run.run_id,
            "created_at_utc": run.created_at_utc,
            "mode": run.mode,
            "chain_status": run.chain_status,
            "parse_ok": run.parse_ok,
            "http_status": run.response_meta.get("http_status"),
            "latency_ms": run.response_meta.get("latency_ms"),
        },
        }
    )
    if not isinstance(out.get("planner_phase"), dict):
        out["planner_phase"] = {
            "executed": False,
            "parse_ok": False,
            "parse_stage": "skipped",
            "parse_error": out.get("planner_skipped_reason"),
            "parsed_output": None,
            "response_meta": {},
            "raw_content": None,
            "reasoning_content": None,
            "packet_compact": [],
            "packet_expanded": [],
            "message_structure": [],
            "request_payload_redacted": {},
        }
    return out


def create_app(*, logs_root: Path | str = Path("logs/probe_web")) -> FastAPI:
    module_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(module_dir / "templates"))

    logs_root_path = ensure_logs_root(Path(logs_root))

    app = FastAPI(title="PALA Probe Web", version="0.1.0")
    app.mount("/probe-web-static", StaticFiles(directory=str(module_dir / "static")), name="probe_web_static")
    app.mount("/probe-logs", StaticFiles(directory=str(logs_root_path)), name="probe_logs")

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
            active_mode="env",
        )
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.get("/runs/recent", response_class=HTMLResponse)
    async def recent_runs(request: Request) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        ctx = _template_context(request=request, defaults=defaults, recent_runs=recent, result=None, active_mode="env")
        return templates.TemplateResponse(request, "partials/recent_runs.html", ctx)

    @app.get("/status/api-key", response_class=HTMLResponse)
    async def api_key_status(request: Request) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        ctx = _template_context(request=request, defaults=defaults, recent_runs=recent, result=None, active_mode="env")
        return templates.TemplateResponse(request, "partials/api_key_status.html", ctx)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        defaults = resolve_defaults()
        loaded = load_run_for_ui(logs_root_path, run_id)
        if loaded is None:
            raise HTTPException(status_code=404, detail="Run not found")
        recent = list_recent_runs(logs_root_path)
        mode = str(loaded.get("mode", "env")).strip().lower()
        if mode not in {"env", "env_planner", "env_planner_preview", "env_planner_from_prepared"}:
            mode = "env"
        active_mode = "env_planner" if mode.startswith("env_planner") else "env"
        prepared_env_run_id = ""
        if mode == "env_planner_preview":
            prepared_env_run_id = str(loaded.get("run_id") or "")
        elif mode == "env_planner_from_prepared":
            prepared_env_run_id = str((loaded.get("params") or {}).get("source_env_run_id") or "")
        ctx = _template_context(
            request=request,
            defaults=defaults,
            recent_runs=recent,
            result=loaded,
            active_mode=active_mode,
            form_values={"probe_mode": active_mode, "prepared_env_run_id": prepared_env_run_id},
        )
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.post("/probe/env", response_class=HTMLResponse)
    async def probe_env(
        request: Request,
        probe_mode: str = Form("env"),
        run_action: str = Form("run_env"),
        prepared_env_run_id: str = Form(""),
        provider: str = Form(""),
        model: str = Form(""),
        base_url: str = Form(""),
        system_prompt: str = Form(""),
        env_contract: str = Form(""),
        policy_identity: str = Form(""),
        timeout_s: str = Form(""),
        env_max_tokens: str = Form(""),
        temperature: str = Form(""),
        top_p: str = Form(""),
        presence_penalty: str = Form(""),
        planner_prompt_override: str = Form(""),
        inter_frame_ms: str = Form(""),
        packet_view_mode: str = Form("compact"),
        image_order: str = Form(""),
        images: List[UploadFile] = File(default=[]),
    ) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        raw_form = {
            "probe_mode": "env",
            "run_action": "run_env",
            "prepared_env_run_id": "",
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "system_prompt": system_prompt,
            "env_contract": env_contract,
            "policy_identity": policy_identity,
            "timeout_s": timeout_s,
            "env_max_tokens": env_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "planner_prompt_override": planner_prompt_override,
            "inter_frame_ms": inter_frame_ms,
            "packet_view_mode": packet_view_mode,
        }

        try:
            params = normalize_params(raw_form, defaults=defaults)
            run = await run_env_probe(
                params=params,
                upload_files=images,
                image_order=image_order,
                logs_root=logs_root_path,
            )
            result = _run_to_view(run)
            recent = list_recent_runs(logs_root_path)
            ctx = _template_context(
                request=request,
                defaults=defaults,
                recent_runs=recent,
                result=result,
                form_values={**asdict(params), "probe_mode": "env", "run_action": "run_env", "prepared_env_run_id": ""},
                active_mode="env",
            )
            return templates.TemplateResponse(request, "partials/result_panel.html", ctx)
        except ProbeInputError as exc:
            ctx = _template_context(
                request=request,
                defaults=defaults,
                recent_runs=recent,
                result=None,
                error_message=str(exc),
                form_values=raw_form,
                active_mode="env",
            )
            return templates.TemplateResponse(request, "partials/result_panel.html", ctx)

    @app.post("/probe/env-planner", response_class=HTMLResponse)
    async def probe_env_planner(
        request: Request,
        probe_mode: str = Form("env_planner"),
        run_action: str = Form("run_both"),
        prepared_env_run_id: str = Form(""),
        provider: str = Form(""),
        model: str = Form(""),
        base_url: str = Form(""),
        system_prompt: str = Form(""),
        env_contract: str = Form(""),
        policy_identity: str = Form(""),
        policy_capabilities: str = Form(""),
        policy_safety: str = Form(""),
        policy_style: str = Form(""),
        timeout_s: str = Form(""),
        env_max_tokens: str = Form(""),
        temperature: str = Form(""),
        top_p: str = Form(""),
        presence_penalty: str = Form(""),
        planner_prompt_override: str = Form(""),
        planner_prompt: str = Form(""),
        planner_max_proposals: str = Form(""),
        planner_use_env_context: str = Form(""),
        planner_max_tokens: str = Form(""),
        planner_temperature: str = Form(""),
        planner_top_p: str = Form(""),
        planner_presence_penalty: str = Form(""),
        planner_system_prompt: str = Form(""),
        planner_image_indices: str = Form(""),
        planner_context_override_json: str = Form(""),
        planner_user_text_override: str = Form(""),
        planner_payload_override_json: str = Form(""),
        inter_frame_ms: str = Form(""),
        packet_view_mode: str = Form("compact"),
        image_order: str = Form(""),
        images: List[UploadFile] = File(default=[]),
    ) -> HTMLResponse:
        defaults = resolve_defaults()
        recent = list_recent_runs(logs_root_path)
        raw_form = {
            "probe_mode": "env_planner",
            "run_action": run_action,
            "prepared_env_run_id": prepared_env_run_id,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "system_prompt": system_prompt,
            "env_contract": env_contract,
            "policy_identity": policy_identity,
            "policy_capabilities": policy_capabilities,
            "policy_safety": policy_safety,
            "policy_style": policy_style,
            "timeout_s": timeout_s,
            "env_max_tokens": env_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "planner_prompt_override": planner_prompt_override,
            "planner_prompt": planner_prompt,
            "planner_max_proposals": planner_max_proposals,
            "planner_use_env_context": planner_use_env_context,
            "planner_max_tokens": planner_max_tokens,
            "planner_temperature": planner_temperature,
            "planner_top_p": planner_top_p,
            "planner_presence_penalty": planner_presence_penalty,
            "planner_system_prompt": planner_system_prompt,
            "planner_image_indices": planner_image_indices,
            "planner_context_override_json": planner_context_override_json,
            "planner_user_text_override": planner_user_text_override,
            "planner_payload_override_json": planner_payload_override_json,
            "inter_frame_ms": inter_frame_ms,
            "packet_view_mode": packet_view_mode,
        }

        try:
            params = normalize_chain_params(raw_form, defaults=defaults)
            action = str(run_action or "run_both").strip().lower()
            if action == "preview":
                run = await run_env_then_preview_planner(
                    params=params,
                    upload_files=images,
                    image_order=image_order,
                    logs_root=logs_root_path,
                )
                next_prepared_run_id = run.run_id
            elif action == "run_planner_only":
                prepared_id = str(prepared_env_run_id or "").strip()
                if not prepared_id:
                    raise ProbeInputError("No prepared env run selected. Run preview first.")
                run = await run_planner_from_prepared_env(
                    params=params,
                    source_env_run_id=prepared_id,
                    logs_root=logs_root_path,
                )
                next_prepared_run_id = prepared_id
            else:
                run = await run_env_planner_probe(
                    params=params,
                    upload_files=images,
                    image_order=image_order,
                    logs_root=logs_root_path,
                )
                next_prepared_run_id = ""
            result = _run_to_view(run)
            recent = list_recent_runs(logs_root_path)
            ctx = _template_context(
                request=request,
                defaults=defaults,
                recent_runs=recent,
                result=result,
                form_values={
                    **asdict(params),
                    "probe_mode": "env_planner",
                    "run_action": "run_both",
                    "prepared_env_run_id": next_prepared_run_id,
                },
                active_mode="env_planner",
            )
            return templates.TemplateResponse(request, "partials/result_panel.html", ctx)
        except ProbeInputError as exc:
            ctx = _template_context(
                request=request,
                defaults=defaults,
                recent_runs=recent,
                result=None,
                error_message=str(exc),
                form_values=raw_form,
                active_mode="env_planner",
            )
            return templates.TemplateResponse(request, "partials/result_panel.html", ctx)

    return app


app = create_app()
