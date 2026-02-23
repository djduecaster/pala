from __future__ import annotations

from typing import Any, Dict, Optional


INTENT_PROPOSALS_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PALA Intent Proposals v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "proposals"],
    "properties": {
        "schema_version": {"type": "string", "const": "pala.intent_proposals.v1"},
        "notes_short": {"type": "string", "maxLength": 280},
        "proposals": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {"$ref": "#/$defs/Proposal"},
        },
    },
    "$defs": {
        "Intent": {
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
        "Primitive": {
            "type": "string",
            "enum": ["hold", "home", "breath", "glance", "nod", "orient_to_zone"],
        },
        "Style": {"type": "string", "enum": ["calm", "curious", "focused"]},
        "Risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "Zone": {"type": "string", "enum": ["left", "center", "right"]},
        "Direction": {"type": "string", "enum": ["left", "right", "up", "down"]},
        "HoldCommand": {"type": "object", "additionalProperties": False, "properties": {}},
        "HomeCommand": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"rate_rad_s": {"type": "number", "minimum": 0.2, "maximum": 5.0}},
        },
        "BreathCommand": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "amp_rad": {"type": "number", "minimum": 0.0, "maximum": 0.35},
                "period_s": {"type": "number", "minimum": 1.5, "maximum": 20.0},
                "rate_rad_s": {"type": "number", "minimum": 0.2, "maximum": 5.0},
            },
        },
        "GlanceCommand": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "direction": {"$ref": "#/$defs/Direction"},
                "amp_rad": {"type": "number", "minimum": 0.0, "maximum": 0.8},
                "duration_s": {"type": "number", "minimum": 0.1, "maximum": 2.0},
                "rate_rad_s": {"type": "number", "minimum": 0.2, "maximum": 5.0},
            },
        },
        "NodCommand": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "amp_rad": {"type": "number", "minimum": 0.0, "maximum": 0.6},
                "duration_s": {"type": "number", "minimum": 0.1, "maximum": 2.0},
                "cycles": {"type": "integer", "minimum": 1, "maximum": 3},
                "rate_rad_s": {"type": "number", "minimum": 0.2, "maximum": 5.0},
            },
        },
        "OrientToZoneCommand": {
            "type": "object",
            "additionalProperties": False,
            "required": ["zone"],
            "properties": {
                "zone": {"$ref": "#/$defs/Zone"},
                "amp_rad": {"type": "number", "minimum": 0.0, "maximum": 0.8},
                "rate_rad_s": {"type": "number", "minimum": 0.2, "maximum": 5.0},
            },
        },
        "EvidenceId": {"type": "string", "minLength": 1, "maxLength": 64},
        "ProposalBase": {
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
                "evidence",
                "rationale_short",
            ],
            "properties": {
                "intent": {"$ref": "#/$defs/Intent"},
                "primitive": {"$ref": "#/$defs/Primitive"},
                "command": {"type": "object"},
                "style": {"$ref": "#/$defs/Style"},
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "urgency": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "risk": {"$ref": "#/$defs/Risk"},
                "allow_interrupt": {"type": "boolean"},
                "min_dwell_ms": {"type": "integer", "minimum": 0, "maximum": 30000},
                "max_duration_ms": {"type": "integer", "minimum": 0, "maximum": 60000},
                "evidence": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 256},
                },
                "rationale_short": {"type": "string", "minLength": 1, "maxLength": 220},
            },
        },
        "Proposal": {
            "allOf": [
                {"$ref": "#/$defs/ProposalBase"},
                {
                    "oneOf": [
                        {
                            "properties": {
                                "primitive": {"const": "hold"},
                                "command": {"$ref": "#/$defs/HoldCommand"},
                            }
                        },
                        {
                            "properties": {
                                "primitive": {"const": "home"},
                                "command": {"$ref": "#/$defs/HomeCommand"},
                            }
                        },
                        {
                            "properties": {
                                "primitive": {"const": "breath"},
                                "command": {"$ref": "#/$defs/BreathCommand"},
                            }
                        },
                        {
                            "properties": {
                                "primitive": {"const": "glance"},
                                "command": {"$ref": "#/$defs/GlanceCommand"},
                            }
                        },
                        {
                            "properties": {
                                "primitive": {"const": "nod"},
                                "command": {"$ref": "#/$defs/NodCommand"},
                            }
                        },
                        {
                            "properties": {
                                "primitive": {"const": "orient_to_zone"},
                                "command": {"$ref": "#/$defs/OrientToZoneCommand"},
                            }
                        },
                    ]
                },
            ]
        },
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


def intent_response_format(*, provider: Optional[str] = None) -> Dict[str, Any]:
    if _use_json_object_mode(provider):
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pala_intent_proposals_v1",
            "strict": True,
            "schema": INTENT_PROPOSALS_SCHEMA,
        },
    }


def env_response_format(*, provider: Optional[str] = None) -> Dict[str, Any]:
    if _use_json_object_mode(provider):
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pala_env_summary_v1",
            "strict": True,
            "schema": ENV_SUMMARY_SCHEMA,
        },
    }


def _use_json_object_mode(provider: Optional[str]) -> bool:
    token = str(provider or "").strip().lower()
    return token == "gemini"
