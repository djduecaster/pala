from __future__ import annotations

from typing import Any, Dict


INTENT_PROPOSALS_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "pala.intent_proposals.v2",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "proposals"],
    "properties": {
        "schema_version": {"type": "string", "const": "pala.intent_proposals.v2"},
        "notes_short": {"type": "string", "maxLength": 280},
        "proposals": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"$ref": "#/$defs/Proposal"},
        },
    },
    "$defs": {
        "Proposal": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "intent",
                "primitive",
                "command",
                "style",
                "score",
                "confidence",
                "urgency",
                "risk",
                "allow_interrupt",
                "rationale_short",
                "evidence",
            ],
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "idle_presence",
                        "acknowledge_presence",
                        "track_user",
                        "scan_environment",
                        "react_to_change",
                        "reset_pose",
                        "affirmation",
                    ],
                },
                "primitive": {
                    "type": "string",
                    "enum": ["hold", "home", "breath", "glance", "nod", "orient_to_zone"],
                },
                "command": {"type": "object"},
                "style": {"type": "string", "enum": ["calm", "curious", "focused"]},
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "urgency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                "allow_interrupt": {"type": "boolean"},
                "min_dwell_ms": {"type": "integer", "minimum": 0, "maximum": 30000},
                "max_duration_ms": {"type": "integer", "minimum": 50, "maximum": 60000},
                "activity_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "novelty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rationale_short": {"type": "string", "minLength": 1, "maxLength": 220},
                "evidence": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 64},
                },
            },
        }
    },
}


ENV_SUMMARY_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PALA Env Summary v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "scene",
        "events",
        "hypotheses",
        "summary_short",
        "delta_score",
        "features",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "pala.env_summary.v1"},
        "scene": {"type": "string", "minLength": 1, "maxLength": 360},
        "events": {"type": "string", "minLength": 1, "maxLength": 220},
        "hypotheses": {"type": "string", "minLength": 1, "maxLength": 220},
        "summary_short": {"type": "string", "minLength": 1, "maxLength": 120},
        "delta_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "features": {
            "type": "object",
            "additionalProperties": False,
            "required": ["person_present", "zone_hint", "activity_level", "novelty"],
            "properties": {
                "person_present": {"type": "boolean"},
                "zone_hint": {"type": "string", "enum": ["left", "center", "right", "unknown"]},
                "activity_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "novelty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
    },
}


def intent_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pala_intent_proposals_v2",
            "strict": True,
            "schema": INTENT_PROPOSALS_SCHEMA,
        },
    }


def env_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pala_env_summary_v1",
            "strict": True,
            "schema": ENV_SUMMARY_SCHEMA,
        },
    }
