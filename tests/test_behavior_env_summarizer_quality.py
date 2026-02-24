from __future__ import annotations

from jsonschema import ValidationError

from pala.behavior import env_summarizer as env_mod


def test_canonicalize_env_payload_feature_fallback_and_zone_inference():
    payload = env_mod._canonicalize_env_payload(  # noqa: SLF001
        {
            "scene": {"person_present": True, "zone_hint": "left", "activity_level": 0.6, "novelty": 0.7},
            "events": "user moved to my right",
            "summary_short": "user moved to my right",
            "delta_score": 0.9,
            "zone_hint": "sideways",
        }
    )
    assert payload is not None
    assert payload["features"]["person_present"] is True
    assert payload["features"]["zone_hint"] in {"left", "right"}
    assert payload["delta_score"] == 0.9


def test_canonicalize_env_payload_uses_scene_dict_when_features_invalid():
    payload = env_mod._canonicalize_env_payload(  # noqa: SLF001
        {
            "scene": {"person_present": False, "zone_hint": "center", "activity_level": 0.1, "novelty": 0.2},
            "features": "invalid",
            "summary": "person in front of me",
            "changes": "minimal movement",
        }
    )
    assert payload is not None
    assert payload["features"]["zone_hint"] == "center"
    assert payload["summary_short"] == "person in front of me"


def test_env_summarizer_helper_functions_cover_path_and_truncation():
    assert env_mod._infer_zone_hint_from_text("", None) == "unknown"  # noqa: SLF001
    assert env_mod._infer_zone_hint_from_text("user on my left then right") == "left"  # noqa: SLF001
    assert env_mod._short_text("x" * 20, max_len=10) == "xxxxxxx..."  # noqa: SLF001

    exc = ValidationError("boom")
    assert env_mod._json_path(exc) == "$"  # noqa: SLF001

    exc_idx = ValidationError("bad")
    exc_idx.relative_path.append("features")
    exc_idx.relative_path.append(0)
    assert env_mod._json_path(exc_idx) == "$.features[0]"  # noqa: SLF001
