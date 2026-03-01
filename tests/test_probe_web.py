from __future__ import annotations

import io
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from tools.probe_web import defaults as defaults_mod
from tools.probe_web.app import create_app
from tools.probe_web.service import parse_image_order


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (96, 72), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _decision_payload(**overrides):
    payload = {
        "schema_version": "pala.behavior_decision.v1",
        "mode": "social_interact",
        "mood": "curious",
        "skill": "greet_user",
        "action": {
            "primitive": "orient_to_zone",
            "command": {"zone": "center", "amp_rad": 0.2, "rate_rad_s": 1.1},
            "style": "curious",
        },
        "confidence": 0.74,
        "rationale_short": "User likely present; orient calmly toward center.",
        "mode_transition": "stay",
    }
    payload.update(overrides)
    return payload


def _fake_behavior_response_json(payload: dict | None = None) -> dict:
    content = payload or _decision_payload()
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
            "prompt_tokens": 132,
            "completion_tokens": 48,
            "total_tokens": 180,
        },
    }


def _fake_behavior_invalid_parse_response_json() -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "not-json"},
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "total_tokens": 24,
        },
    }


def _fake_behavior_truncated_response_json() -> dict:
    return {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "{\"schema_version\":\"pala.behavior_decision",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 111,
            "completion_tokens": 9,
            "total_tokens": 333,
        },
    }


def _four_files() -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("images", ("a.jpg", _image_bytes((255, 0, 0)), "image/jpeg")),
        ("images", ("b.jpg", _image_bytes((0, 255, 0)), "image/jpeg")),
        ("images", ("c.jpg", _image_bytes((0, 0, 255)), "image/jpeg")),
        ("images", ("d.jpg", _image_bytes((220, 120, 30)), "image/jpeg")),
    ]


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
    assert out.max_tokens >= 1000


def test_parse_image_order_validation():
    assert parse_image_order("", file_count=4) == [0, 1, 2, 3]
    assert parse_image_order("2,0,1,3", file_count=4) == [2, 0, 1, 3]


def test_parse_image_order_rejects_duplicate():
    try:
        parse_image_order("0,1,1,3", file_count=4)
    except Exception as exc:  # noqa: BLE001
        assert "exactly once" in str(exc)
    else:
        raise AssertionError("Expected parse_image_order to reject duplicate index")


def test_probe_v4_happy_path(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")

    def _fake_post_chat_json(**kwargs):
        payload = kwargs.get("payload", {})
        schema_name = payload.get("response_format", {}).get("json_schema", {}).get("name")
        assert schema_name == "pala_behavior_decision_v1"
        return SimpleNamespace(
            ok=True,
            status_code=200,
            latency_ms=76.0,
            error=None,
            response_json=_fake_behavior_response_json(),
        )

    monkeypatch.setattr(service_mod, "post_chat_json", _fake_post_chat_json)

    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)

    resp = client.post(
        "/probe/v4/run",
        data={
            "image_order": "3,1,2,0",
            "system_prompt": "SYSTEM OVERRIDE",
            "policy_identity": "IDENTITY OVERRIDE",
            "policy_capabilities": "CAPS OVERRIDE",
            "policy_safety": "SAFETY OVERRIDE",
            "policy_style": "STYLE OVERRIDE",
            "planner_prompt": "PROMPT OVERRIDE",
            "max_tokens": "900",
        },
        files=_four_files(),
    )
    assert resp.status_code == 200
    assert "Run " in resp.text
    assert "Guard Result" in resp.text

    index_path = tmp_path / "probe_web_v4" / "recent_index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    run_id = index[0]["run_id"]

    run_dir = tmp_path / "probe_web_v4" / run_id
    assert (run_dir / "run_full.json").exists()
    assert (run_dir / "effective_inputs.json").exists()
    assert (run_dir / "fsm_before.json").exists()
    assert (run_dir / "fsm_after.json").exists()

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["system_prompt"] == "SYSTEM OVERRIDE"
    assert run_config["policy_identity"] == "IDENTITY OVERRIDE"

    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    assert run_id in detail.text
    assert "Behavior V4 Packet Builder" in detail.text


def test_probe_v4_missing_api_key_error(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: None)

    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)

    resp = client.post("/probe/v4/run", data={"image_order": "0,1,2,3"}, files=_four_files())
    assert resp.status_code == 200
    assert "Missing API key" in resp.text


def test_api_key_status_refresh_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("PALA_COSMOS_API_KEY", "abc123")
    app = create_app(logs_root=tmp_path / "probe_web_v4")
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


def test_probe_v4_parse_fail_reports_truncation_hint(tmp_path, monkeypatch):
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
            response_json=_fake_behavior_truncated_response_json(),
        ),
    )

    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)

    resp = client.post("/probe/v4/run", data={"image_order": "0,1,2,3"}, files=_four_files())
    assert resp.status_code == 200
    assert "truncated_response" in resp.text

    index = json.loads((tmp_path / "probe_web_v4" / "recent_index.json").read_text(encoding="utf-8"))
    run_id = index[0]["run_id"]
    summary = json.loads((tmp_path / "probe_web_v4" / run_id / "summary.json").read_text(encoding="utf-8"))
    assert "truncated_response" in (summary.get("parse_error") or "")


def test_probe_v4_parse_fail_schema_version_hint(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")
    bad = _decision_payload(schema_version="pala.behavior_decision.v2")
    monkeypatch.setattr(
        service_mod,
        "post_chat_json",
        lambda **_kwargs: SimpleNamespace(
            ok=True,
            status_code=200,
            latency_ms=45.0,
            error=None,
            response_json=_fake_behavior_response_json(payload=bad),
        ),
    )

    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)
    resp = client.post("/probe/v4/run", data={"image_order": "0,1,2,3"}, files=_four_files())
    assert resp.status_code == 200
    assert "check_schema_version_or_click_sync_override_defaults" in resp.text


def test_probe_v4_guard_fallback_when_skill_not_allowed(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")
    bad_decision = _decision_payload(mode="social_interact", skill="greet_user", mode_transition="stay")
    monkeypatch.setattr(
        service_mod,
        "post_chat_json",
        lambda **_kwargs: SimpleNamespace(
            ok=True,
            status_code=200,
            latency_ms=62.0,
            error=None,
            response_json=_fake_behavior_response_json(payload=bad_decision),
        ),
    )

    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)

    resp = client.post("/probe/v4/run", data={"image_order": "0,1,2,3"}, files=_four_files())
    assert resp.status_code == 200
    assert "Guard Result" in resp.text

    index = json.loads((tmp_path / "probe_web_v4" / "recent_index.json").read_text(encoding="utf-8"))
    assert index[0]["guard_used_fallback"] is True
    assert index[0]["guard_reason"] in {
        "mode_not_current",
        "min_action_dwell",
        "skill_not_allowed",
        "primitive_not_allowed",
    }


def test_probe_v4_allows_payload_override(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")
    seen = {"payload": None}

    def _fake_post_chat_json(**kwargs):
        seen["payload"] = kwargs.get("payload")
        return SimpleNamespace(
            ok=True,
            status_code=200,
            latency_ms=88.6,
            error=None,
            response_json=_fake_behavior_response_json(),
        )

    monkeypatch.setattr(service_mod, "post_chat_json", _fake_post_chat_json)

    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)

    payload_override = {
        "model": "override-model",
        "messages": [
            {"role": "system", "content": "OVERRIDE SYSTEM"},
            {"role": "user", "content": [{"type": "text", "text": "OVERRIDE USER"}]},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pala_behavior_decision_v1",
                "strict": True,
                "schema": {
                    "type": "object",
                    "required": ["schema_version"],
                    "properties": {"schema_version": {"type": "string"}},
                },
            },
        },
        "max_tokens": 222,
        "temperature": 0.4,
        "top_p": 0.8,
        "presence_penalty": 0.1,
        "stream": False,
    }

    resp = client.post(
        "/probe/v4/run",
        data={
            "image_order": "0,1,2,3",
            "payload_override_json": json.dumps(payload_override),
        },
        files=_four_files(),
    )
    assert resp.status_code == 200
    assert seen["payload"] is not None
    assert seen["payload"]["model"] == "override-model"


def test_fsm_meta_reset_step_routes(tmp_path):
    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)

    meta = client.get("/probe/v4/fsm/meta")
    assert meta.status_code == 200
    assert "Mode FSM Simulator" not in meta.text
    assert "boot_awaken" in meta.text

    reset = client.post("/probe/v4/fsm/reset", data={"now_mono_s": "2.0"})
    assert reset.status_code == 200
    assert "2.0" in reset.text

    step = client.post(
        "/probe/v4/fsm/step",
        data={
            "fsm_advance_s": "1.5",
            "startup_complete": "true",
            "person_present": "false",
            "person_conf": "0.0",
            "search_requested": "false",
            "search_complete": "false",
            "task_active": "false",
            "health_degraded": "false",
        },
    )
    assert step.status_code == 200
    assert "idle_presence" in step.text


def test_index_contains_help_overlay(tmp_path):
    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "Input/Output Help" in resp.text
    assert "Behavior V4 Model Packet" in resp.text
    assert "Mode FSM Simulator" in resp.text
    assert "Override Fields" in resp.text
    assert "Sync Override Defaults From FSM State" in resp.text
    assert "Scenario Presets" in resp.text
    assert "Run Compare" in resp.text
    assert "id=\"run-status\"" in resp.text
    assert "id=\"json-validate-btn\"" in resp.text


def test_override_defaults_endpoint(tmp_path):
    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)

    resp = client.get("/probe/v4/overrides/defaults")
    assert resp.status_code == 200
    assert "name=\"context_override_json\"" in resp.text
    assert "name=\"user_text_override\"" in resp.text
    assert "name=\"payload_override_json\"" in resp.text


def test_apply_scenario_preset_endpoint_updates_fsm(tmp_path):
    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)

    resp = client.post("/probe/v4/presets/apply", data={"preset_id": "social_greeting"})
    assert resp.status_code == 200
    assert "Applied preset: Social Greeting" in resp.text
    assert "hx-swap-oob" in resp.text

    meta = client.get("/probe/v4/fsm/meta")
    assert meta.status_code == 200
    assert "social_interact" in meta.text


def test_compare_runs_endpoint(tmp_path, monkeypatch):
    from tools.probe_web import service as service_mod

    monkeypatch.setattr(service_mod, "resolve_api_key", lambda: "test-key")
    call_state = {"n": 0}

    def _fake_post_chat_json(**_kwargs):
        call_state["n"] += 1
        return SimpleNamespace(
            ok=True,
            status_code=200,
            latency_ms=60.0 + (call_state["n"] * 10.0),
            error=None,
            response_json=_fake_behavior_response_json(),
        )

    monkeypatch.setattr(service_mod, "post_chat_json", _fake_post_chat_json)

    app = create_app(logs_root=tmp_path / "probe_web_v4")
    client = TestClient(app)

    run1 = client.post("/probe/v4/run", data={"image_order": "0,1,2,3"}, files=_four_files())
    assert run1.status_code == 200
    run2 = client.post("/probe/v4/run", data={"image_order": "0,1,2,3"}, files=_four_files())
    assert run2.status_code == 200

    recent = json.loads((tmp_path / "probe_web_v4" / "recent_index.json").read_text(encoding="utf-8"))
    assert len(recent) >= 2
    run_a_id = recent[0]["run_id"]
    run_b_id = recent[1]["run_id"]

    resp = client.post("/compare/runs", data={"compare_run_a_id": run_a_id, "compare_run_b_id": run_b_id})
    assert resp.status_code == 200
    assert "Comparison Highlights" in resp.text
    assert run_a_id in resp.text
    assert run_b_id in resp.text
