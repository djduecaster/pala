from __future__ import annotations

import json

from pala.behavior import IntentProposer, parse_intent_proposer_response


def _proposal(**overrides):
    item = {
        "intent": "track_user",
        "primitive": "orient_to_zone",
        "command": {"zone": "left", "amp_rad": 0.2, "rate_rad_s": 1.1},
        "style": "focused",
        "score": 0.8,
        "confidence": 0.7,
        "urgency": 0.5,
        "risk": "low",
        "allow_interrupt": True,
        "evidence": ["frame:latest"],
        "rationale_short": "track user movement",
    }
    item.update(overrides)
    return item


def test_parser_requires_schema_and_strict_fields():
    raw = json.dumps({"notes_short": "x"})
    parsed = parse_intent_proposer_response(raw)
    assert parsed is None


def test_parser_accepts_wrapped_v2_and_drops_invalid_items():
    wrapped = json.dumps(
        {
            "pala.intent_proposals.v2": {
                "proposals": [
                    _proposal(),
                    _proposal(
                        primitive="glance",
                        intent="scan_environment",
                        command={"direction": "right", "amp_rad": 0.2, "duration_s": 0.4, "rate_rad_s": 1.4},
                        style="curious",
                        rationale_short="scan right",
                    ),
                    _proposal(
                        primitive="orient_to_zone",
                        command={"direction": "forward"},
                        rationale_short="invalid zone",
                    ),
                ]
            }
        }
    )
    parsed = parse_intent_proposer_response(wrapped)
    assert parsed is not None
    assert len(parsed.response.proposals) == 2


def test_parser_handles_fenced_json_and_records_stage():
    raw = (
        "```json\n"
        + json.dumps(
            {
                "schema_version": "pala.intent_proposals.v2",
                "proposals": [
                    _proposal(),
                    _proposal(
                        primitive="glance",
                        intent="scan_environment",
                        command={"direction": "left", "amp_rad": 0.2, "duration_s": 0.4, "rate_rad_s": 1.2},
                        style="curious",
                        rationale_short="scan left",
                    ),
                    _proposal(
                        primitive="breath",
                        intent="idle_presence",
                        command={"amp_rad": 0.08, "period_s": 7.0, "rate_rad_s": 1.0},
                        style="calm",
                        rationale_short="fallback breathe",
                    ),
                ],
            }
        )
        + "\n```"
    )
    parsed = parse_intent_proposer_response(raw)
    assert parsed is not None
    assert parsed.parse_stage in {"defenced", "extracted", "raw"}


def test_bookkeeping_records_parse_error():
    proposer = IntentProposer()
    proposer.submit_or_replace({"tick": 1})
    assert proposer.complete_request("{bad json") is None
    assert (proposer.last_parse_error or "").startswith("json_decode:")
