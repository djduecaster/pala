from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from pala.behavior.schemas import intent_response_format
from tools.probe_web import defaults as defaults_mod
from tools.probe_web.app import create_app
from tools.probe_web.service import parse_image_order


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (64, 48), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _fake_env_response_json() -> dict:
    content = {
        "schema_version": "pala.env_summary.v1",
        "scene": "I see the scene as a desk with a laptop.",
        "events": "I notice slight movement near the keyboard.",
        "hypotheses": "I infer someone is interacting with the desk.",
        "summary_short": "Desk scene with minor activity.",
        "delta_score": 0.42,
        "features": {
            "person_present": True,
            "zone_hint": "center",
            "activity_level": 0.38,
            "novelty": 0.31,
        },
    }
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(content, separators=(",", ":")),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "total_tokens": 168,
        },
    }


def _fake_planner_response_json() -> dict:
    content = {
        "schema_version": "pala.intent_proposals.v2",
        "proposals": [
            {
                "intent": "scan_environment",
                "primitive": "orient_to_zone",
                "command": {"zone": "center", "amp_rad": 0.24, "rate_rad_s": 1.3},
                "style": "calm",
                "score": 0.84,
                "confidence": 0.76,
                "urgency": 0.33,
                "risk": "low",
                "allow_interrupt": True,
                "rationale_short": "Maintain awareness around center activity.",
                "evidence": ["frame:latest", "env:latest"],
            }
        ],
    }
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(content, separators=(",", ":")),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 111,
            "completion_tokens": 37,
            "total_tokens": 148,
        },
    }


def _fake_env_invalid_parse_response_json() -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "not-json-env-response",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 45,
            "completion_tokens": 11,
            "total_tokens": 56,
        },
    }


def _fake_env_truncated_response_json() -> dict:
    return {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "{\"schema_version\":\"pala.env_summary",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 111,
            "completion_tokens": 9,
            "total_tokens": 333,
        },
    }


def test_resolve_defaults_uses_gemini_preset(monkeypatch):
    monkeypatch.setenv("PALA_COSMOS_MODEL", "")
    monkeypatch.setenv("PALA_COSMOS_BASE_URL", "")
    monkeypatch.delenv("PALA_COSMOS_API_KEY", raising=False)
    monkeypatch.delenv("PALA_ENV_FILE", raising=False)
    monkeypatch.setattr(defaults_mod, "load_config", lambda _path: SimpleNamespace(cosmos=SimpleNamespace()))

    out = defaults_mod.resolve_defaults()
    assert out.provider == "gemini"
    assert out.model == "gemini-2.5-flash"
    assert out.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert out.env_max_tokens >= 1000


def test_parse_image_order_validation():
    assert parse_image_order("", file_count=4) == [0, 1, 2, 3]
    assert parse_image_order("2,0,1,3", file_count=4) == [2, 0, 1, 3]


def test_probe_env_happy_path(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(
        service_mod,
        "post_chat_json",
        lambda **_kwargs: SimpleNamespace(
            ok=True,
            status_code=200,
            latency_ms=87.4,
            error=None,
            response_json=_fake_env_response_json(),
        ),
    )

    app = create_app(logs_root=tmp_path / "probe_web")
    client = TestClient(app)

    files = [
        ("images", ("a.jpg", _image_bytes((255, 0, 0)), "image/jpeg")),
        ("images", ("b.jpg", _image_bytes((0, 255, 0)), "image/jpeg")),
        ("images", ("c.jpg", _image_bytes((0, 0, 255)), "image/jpeg")),
        ("images", ("d.jpg", _image_bytes((220, 120, 30)), "image/jpeg")),
    ]
    data = {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "system_prompt": "SYSTEM OVERRIDE",
        "env_contract": "CONTRACT OVERRIDE",
        "policy_identity": "IDENTITY OVERRIDE",
        "timeout_s": "20",
        "env_max_tokens": "600",
        "temperature": "0",
        "top_p": "0.3",
        "presence_penalty": "0",
        "planner_prompt_override": "",
        "inter_frame_ms": "250",
        "packet_view_mode": "compact",
        "image_order": "3,1,2,0",
    }
    resp = client.post("/probe/env", data=data, files=files)
    assert resp.status_code == 200
    assert "Run " in resp.text
    assert "Parsed Output" in resp.text

    index_path = tmp_path / "probe_web" / "recent_index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert isinstance(index, list)
    assert len(index) >= 1
    run_id = index[0]["run_id"]

    summary_path = tmp_path / "probe_web" / run_id / "summary.json"
    assert summary_path.exists()
    run_config = json.loads((tmp_path / "probe_web" / run_id / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["system_prompt"] == "SYSTEM OVERRIDE"
    assert run_config["env_contract"] == "CONTRACT OVERRIDE"
    assert run_config["policy_identity"] == "IDENTITY OVERRIDE"

    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    assert run_id in detail.text


def test_probe_env_missing_api_key_error(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: None)

    app = create_app(logs_root=tmp_path / "probe_web")
    client = TestClient(app)

    files = [
        ("images", ("a.jpg", _image_bytes((255, 0, 0)), "image/jpeg")),
        ("images", ("b.jpg", _image_bytes((0, 255, 0)), "image/jpeg")),
        ("images", ("c.jpg", _image_bytes((0, 0, 255)), "image/jpeg")),
        ("images", ("d.jpg", _image_bytes((220, 120, 30)), "image/jpeg")),
    ]
    resp = client.post("/probe/env", data={"image_order": "0,1,2,3"}, files=files)
    assert resp.status_code == 200
    assert "Missing API key" in resp.text


def test_parse_image_order_rejects_duplicate():
    try:
        parse_image_order("0,1,1,3", file_count=4)
    except Exception as exc:
        assert "exactly once" in str(exc)
    else:
        raise AssertionError("Expected parse_image_order to reject duplicate index")


def test_api_key_status_refresh_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("PALA_COSMOS_API_KEY", "abc123")
    app = create_app(logs_root=tmp_path / "probe_web")
    client = TestClient(app)

    resp = client.get("/status/api-key")
    assert resp.status_code == 200
    assert "API key source" in resp.text
    assert "detected" in resp.text


def test_resolve_api_key_falls_back_to_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "home" / ".config" / "pala" / "env.sh"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("export PALA_COSMOS_API_KEY='from-file-key'\n", encoding="utf-8")

    monkeypatch.delenv("PALA_COSMOS_API_KEY", raising=False)
    monkeypatch.setenv("PALA_ENV_FILE", str(env_file))
    monkeypatch.setattr(defaults_mod, "load_config", lambda _path: SimpleNamespace(cosmos=SimpleNamespace()))

    key = defaults_mod.resolve_api_key()
    defaults = defaults_mod.resolve_defaults()
    assert key == "from-file-key"
    assert defaults.has_api_key is True
    assert "fallback" in defaults.api_key_source


def test_resolve_api_key_prefers_process_env_over_file(tmp_path, monkeypatch):
    env_file = tmp_path / "home" / ".config" / "pala" / "env.sh"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("export PALA_COSMOS_API_KEY='from-file-key'\n", encoding="utf-8")

    monkeypatch.setenv("PALA_ENV_FILE", str(env_file))
    monkeypatch.setenv("PALA_COSMOS_API_KEY", "from-process-env")

    key = defaults_mod.resolve_api_key()
    _token, source = defaults_mod.resolve_api_key_info()
    assert key == "from-process-env"
    assert "process env" in source


def test_probe_env_planner_happy_path(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")

    def _fake_post_chat_json(**kwargs):
        payload = kwargs.get("payload", {})
        schema_name = (
            payload.get("response_format", {})
            .get("json_schema", {})
            .get("name")
        )
        if schema_name == "pala_env_summary_v1":
            return SimpleNamespace(
                ok=True,
                status_code=200,
                latency_ms=82.5,
                error=None,
                response_json=_fake_env_response_json(),
            )
        if schema_name == "pala_intent_proposals_v2":
            return SimpleNamespace(
                ok=True,
                status_code=200,
                latency_ms=91.2,
                error=None,
                response_json=_fake_planner_response_json(),
            )
        raise AssertionError(f"unexpected schema_name: {schema_name}")

    monkeypatch.setattr(service_mod, "post_chat_json", _fake_post_chat_json)

    app = create_app(logs_root=tmp_path / "probe_web")
    client = TestClient(app)

    files = [
        ("images", ("a.jpg", _image_bytes((255, 0, 0)), "image/jpeg")),
        ("images", ("b.jpg", _image_bytes((0, 255, 0)), "image/jpeg")),
        ("images", ("c.jpg", _image_bytes((0, 0, 255)), "image/jpeg")),
        ("images", ("d.jpg", _image_bytes((220, 120, 30)), "image/jpeg")),
    ]
    data = {
        "probe_mode": "env_planner",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "system_prompt": "SYSTEM OVERRIDE",
        "env_contract": "CONTRACT OVERRIDE",
        "policy_identity": "IDENTITY OVERRIDE",
        "policy_capabilities": "CAPABILITIES OVERRIDE",
        "policy_safety": "SAFETY OVERRIDE",
        "policy_style": "STYLE OVERRIDE",
        "timeout_s": "20",
        "env_max_tokens": "600",
        "temperature": "0",
        "top_p": "0.3",
        "presence_penalty": "0",
        "planner_prompt_override": "",
        "planner_prompt": "PLANNER OVERRIDE",
        "planner_max_proposals": "2",
        "planner_use_env_context": "true",
        "planner_max_tokens": "500",
        "planner_temperature": "0.1",
        "planner_top_p": "0.5",
        "planner_presence_penalty": "0",
        "inter_frame_ms": "1000",
        "packet_view_mode": "compact",
        "image_order": "0,1,2,3",
    }
    resp = client.post("/probe/env-planner", data=data, files=files)
    assert resp.status_code == 200
    assert "Planner Phase" in resp.text
    assert "Effective Inputs" in resp.text
    assert "Planner Effective Inputs" in resp.text
    assert "Request Payload (Redacted)" in resp.text
    assert "Planner Request Packet (Redacted)" in resp.text

    index = json.loads((tmp_path / "probe_web" / "recent_index.json").read_text(encoding="utf-8"))
    run_id = index[0]["run_id"]
    assert index[0]["mode"] == "env_planner"
    assert index[0]["chain_status"] == "ok"
    assert index[0]["planner_executed"] is True
    assert index[0]["planner_parse_ok"] is True

    run_dir = tmp_path / "probe_web" / run_id
    assert (run_dir / "run_full.json").exists()
    assert (run_dir / "planner_packet_view.json").exists()
    assert (run_dir / "planner_parsed_output.json").exists()
    assert (run_dir / "effective_inputs.json").exists()

    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    assert "Env + Planner" in detail.text


def test_probe_env_planner_skips_on_env_parse_fail(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")

    def _fake_post_chat_json(**kwargs):
        payload = kwargs.get("payload", {})
        schema_name = (
            payload.get("response_format", {})
            .get("json_schema", {})
            .get("name")
        )
        if schema_name == "pala_env_summary_v1":
            return SimpleNamespace(
                ok=True,
                status_code=200,
                latency_ms=61.1,
                error=None,
                response_json=_fake_env_invalid_parse_response_json(),
            )
        raise AssertionError("planner call should be skipped when env parse fails")

    monkeypatch.setattr(service_mod, "post_chat_json", _fake_post_chat_json)

    app = create_app(logs_root=tmp_path / "probe_web")
    client = TestClient(app)
    files = [
        ("images", ("a.jpg", _image_bytes((255, 0, 0)), "image/jpeg")),
        ("images", ("b.jpg", _image_bytes((0, 255, 0)), "image/jpeg")),
        ("images", ("c.jpg", _image_bytes((0, 0, 255)), "image/jpeg")),
        ("images", ("d.jpg", _image_bytes((220, 120, 30)), "image/jpeg")),
    ]
    resp = client.post("/probe/env-planner", data={"image_order": "0,1,2,3"}, files=files)
    assert resp.status_code == 200
    assert "Planner step skipped" in resp.text

    index = json.loads((tmp_path / "probe_web" / "recent_index.json").read_text(encoding="utf-8"))
    assert index[0]["mode"] == "env_planner"
    assert index[0]["chain_status"] == "partial_env_parse_fail"
    assert index[0]["planner_executed"] is False


def test_probe_env_parse_fail_reports_truncation_hint(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(
        service_mod,
        "post_chat_json",
        lambda **_kwargs: SimpleNamespace(
            ok=True,
            status_code=200,
            latency_ms=45.0,
            error=None,
            response_json=_fake_env_truncated_response_json(),
        ),
    )

    app = create_app(logs_root=tmp_path / "probe_web")
    client = TestClient(app)
    files = [
        ("images", ("a.jpg", _image_bytes((255, 0, 0)), "image/jpeg")),
        ("images", ("b.jpg", _image_bytes((0, 255, 0)), "image/jpeg")),
        ("images", ("c.jpg", _image_bytes((0, 0, 255)), "image/jpeg")),
        ("images", ("d.jpg", _image_bytes((220, 120, 30)), "image/jpeg")),
    ]
    resp = client.post("/probe/env", data={"image_order": "0,1,2,3"}, files=files)
    assert resp.status_code == 200
    assert "truncated_response" in resp.text

    index = json.loads((tmp_path / "probe_web" / "recent_index.json").read_text(encoding="utf-8"))
    run_id = index[0]["run_id"]
    summary = json.loads((tmp_path / "probe_web" / run_id / "summary.json").read_text(encoding="utf-8"))
    assert "truncated_response" in (summary.get("parse_error") or "")


def test_probe_env_planner_allows_payload_override(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")
    seen = {"planner_payload": None}

    def _fake_post_chat_json(**kwargs):
        payload = kwargs.get("payload", {})
        schema_name = (
            payload.get("response_format", {})
            .get("json_schema", {})
            .get("name")
        )
        if schema_name == "pala_env_summary_v1":
            return SimpleNamespace(
                ok=True,
                status_code=200,
                latency_ms=72.2,
                error=None,
                response_json=_fake_env_response_json(),
            )
        seen["planner_payload"] = payload
        return SimpleNamespace(
            ok=True,
            status_code=200,
            latency_ms=88.6,
            error=None,
            response_json=_fake_planner_response_json(),
        )

    monkeypatch.setattr(service_mod, "post_chat_json", _fake_post_chat_json)

    app = create_app(logs_root=tmp_path / "probe_web")
    client = TestClient(app)
    files = [
        ("images", ("a.jpg", _image_bytes((255, 0, 0)), "image/jpeg")),
        ("images", ("b.jpg", _image_bytes((0, 255, 0)), "image/jpeg")),
        ("images", ("c.jpg", _image_bytes((0, 0, 255)), "image/jpeg")),
        ("images", ("d.jpg", _image_bytes((220, 120, 30)), "image/jpeg")),
    ]
    planner_override_payload = {
        "model": "override-model",
        "messages": [
            {"role": "system", "content": "OVERRIDE SYSTEM"},
            {"role": "user", "content": [{"type": "text", "text": "OVERRIDE USER"}]},
        ],
        "temperature": 0.77,
        "top_p": 0.66,
        "presence_penalty": 0.22,
        "max_tokens": 333,
        "stream": False,
        "response_format": intent_response_format(),
    }
    resp = client.post(
        "/probe/env-planner",
        data={
            "image_order": "0,1,2,3",
            "planner_payload_override_json": json.dumps(planner_override_payload),
        },
        files=files,
    )
    assert resp.status_code == 200
    assert "Planner Request Packet (Redacted)" in resp.text
    assert seen["planner_payload"] is not None
    assert seen["planner_payload"]["model"] == "override-model"


def test_probe_env_preview_then_run_planner_from_prepared(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")
    calls = {"env": 0, "planner": 0}

    def _fake_post_chat_json(**kwargs):
        payload = kwargs.get("payload", {})
        schema_name = (
            payload.get("response_format", {})
            .get("json_schema", {})
            .get("name")
        )
        if schema_name == "pala_env_summary_v1":
            calls["env"] += 1
            return SimpleNamespace(
                ok=True,
                status_code=200,
                latency_ms=50.0,
                error=None,
                response_json=_fake_env_response_json(),
            )
        if schema_name == "pala_intent_proposals_v2":
            calls["planner"] += 1
            return SimpleNamespace(
                ok=True,
                status_code=200,
                latency_ms=66.0,
                error=None,
                response_json=_fake_planner_response_json(),
            )
        raise AssertionError(f"unexpected schema: {schema_name}")

    monkeypatch.setattr(service_mod, "post_chat_json", _fake_post_chat_json)

    app = create_app(logs_root=tmp_path / "probe_web")
    client = TestClient(app)

    files = [
        ("images", ("a.jpg", _image_bytes((255, 0, 0)), "image/jpeg")),
        ("images", ("b.jpg", _image_bytes((0, 255, 0)), "image/jpeg")),
        ("images", ("c.jpg", _image_bytes((0, 0, 255)), "image/jpeg")),
        ("images", ("d.jpg", _image_bytes((220, 120, 30)), "image/jpeg")),
    ]

    preview_resp = client.post(
        "/probe/env-planner",
        data={"run_action": "preview", "image_order": "0,1,2,3"},
        files=files,
    )
    assert preview_resp.status_code == 200
    assert "id=\"prepared-env-run-id\"" in preview_resp.text
    assert "Planner step skipped: awaiting_manual_run" in preview_resp.text
    assert calls["env"] == 1
    assert calls["planner"] == 0

    recent_index = json.loads((tmp_path / "probe_web" / "recent_index.json").read_text(encoding="utf-8"))
    prepared_id = recent_index[0]["run_id"]
    assert recent_index[0]["mode"] == "env_planner_preview"
    assert recent_index[0]["chain_status"] == "planner_pending"

    planner_resp = client.post(
        "/probe/env-planner",
        data={
            "run_action": "run_planner_only",
            "prepared_env_run_id": prepared_id,
            "planner_image_indices": "4",
        },
    )
    assert planner_resp.status_code == 200
    assert "mode=env_planner_from_prepared" in planner_resp.text
    assert calls["planner"] == 1

    recent_index = json.loads((tmp_path / "probe_web" / "recent_index.json").read_text(encoding="utf-8"))
    assert recent_index[0]["mode"] == "env_planner_from_prepared"
    assert recent_index[0]["planner_executed"] is True


def test_index_contains_help_overlay(tmp_path):
    app = create_app(logs_root=tmp_path / "probe_web")
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "Input/Output Help" in resp.text
    assert "Env Processor (pala.env_summary.v1)" in resp.text
    assert "Intent Proposer / Planner (pala.intent_proposals.v2)" in resp.text
    assert "Run Env, Preview Planner Inputs" in resp.text
    assert "Run Planner From Prepared Env" in resp.text
    assert "Run Env + Planner" in resp.text
