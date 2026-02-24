from __future__ import annotations

import json

import pala.behavior.intent_proposer as intent_mod


def _proposal(**overrides):
    item = {
        "intent": "track_user",
        "primitive": "orient_to_zone",
        "command": {"zone": "left", "amp_rad": 0.25, "rate_rad_s": 1.2},
        "style": "focused",
        "score": 0.7,
        "confidence": 0.75,
        "urgency": 0.5,
        "risk": "low",
        "allow_interrupt": True,
        "evidence": ["frame:latest"],
        "rationale_short": "track movement",
    }
    item.update(overrides)
    return item


def test_parser_rejects_non_object_payloads_after_json_parse():
    parsed, err, stage = intent_mod._parse_intent_proposer_response_with_error("42")  # noqa: SLF001
    assert parsed is None
    assert err == "proposals_missing_or_empty"
    assert stage == "raw"


def test_canonicalize_payload_handles_wrapped_lists_and_invalid_shapes():
    assert intent_mod._canonicalize_payload("bad-shape") is None  # noqa: SLF001
    wrapped = intent_mod._canonicalize_payload({"pala.intent_proposals.v2": [{"intent": "track_user"}]})  # noqa: SLF001
    assert wrapped == {
        "schema_version": "pala.intent_proposals.v2",
        "notes_short": "",
        "proposals": [{"intent": "track_user"}],
    }


def test_parse_proposal_rejects_invalid_required_fields():
    assert intent_mod._parse_proposal("not-a-dict") is None  # noqa: SLF001
    assert intent_mod._parse_proposal(_proposal(intent="bad_intent")) is None  # noqa: SLF001
    assert intent_mod._parse_proposal(_proposal(primitive="spin")) is None  # noqa: SLF001
    assert intent_mod._parse_proposal(_proposal(style="bold")) is None  # noqa: SLF001
    assert intent_mod._parse_proposal(_proposal(risk="critical")) is None  # noqa: SLF001
    assert intent_mod._parse_proposal(_proposal(command="invalid")) is None  # noqa: SLF001
    assert intent_mod._parse_proposal(_proposal(evidence="frame:latest")) is None  # noqa: SLF001
    assert intent_mod._parse_proposal(_proposal(rationale_short="   ")) is None  # noqa: SLF001
    assert intent_mod._parse_proposal(_proposal(allow_interrupt="yes")) is None  # noqa: SLF001


def test_parse_proposal_strips_blank_evidence_tokens():
    parsed = intent_mod._parse_proposal(_proposal(evidence=["", "   ", "frame:latest"]))  # noqa: SLF001
    assert parsed is not None
    assert parsed.evidence == ["frame:latest"]


def test_normalize_command_handles_nod_and_invalid_variants():
    nod = intent_mod._normalize_command(  # noqa: SLF001
        "nod",
        {"amp_rad": 1.5, "duration_s": 0.0, "cycles": 99, "rate_rad_s": 0.0},
    )
    assert nod == {
        "amp_rad": 0.6,
        "duration_s": 0.1,
        "cycles": 3,
        "rate_rad_s": 0.2,
    }
    assert intent_mod._normalize_command("glance", {"direction": "forward"}) is None  # noqa: SLF001
    assert intent_mod._normalize_command("unknown_primitive", {}) is None  # noqa: SLF001


def test_parser_defensive_paths_when_schema_validation_is_bypassed(monkeypatch):
    monkeypatch.setattr(intent_mod, "validate", lambda instance, schema: None)

    empty_raw = json.dumps({"schema_version": "pala.intent_proposals.v2", "proposals": []})
    parsed_empty, err_empty, _ = intent_mod._parse_intent_proposer_response_with_error(empty_raw)  # noqa: SLF001
    assert parsed_empty is None
    assert err_empty == "proposals_missing_or_empty"

    all_invalid_raw = json.dumps({"schema_version": "pala.intent_proposals.v2", "proposals": [{}, {}, {}]})
    parsed_invalid, err_invalid, _ = intent_mod._parse_intent_proposer_response_with_error(all_invalid_raw)  # noqa: SLF001
    assert parsed_invalid is None
    assert err_invalid == "all_proposals_invalid"
