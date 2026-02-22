from __future__ import annotations

import json
from typing import Any, Dict, List

from ..types import PrimitiveKind


SYSTEM_PROMPT = "You are a helpful assistant."


def build_env_user_text(*, context: Dict[str, Any], policy_identity: str) -> str:
    contract = (
        "You are the PALA environment processor.\n"
        "Use visual evidence from the provided frames to describe the scene and recent changes.\n"
        "Frames are provided in strict chronological order (oldest to newest) from one short rolling video window.\n"
        "Treat them as one temporal sequence of the same scene unless there is explicit evidence of a hard scene cut.\n"
        "Do not propose actions.\n"
        "Return exactly one instance of each required tag with plain text content:\n"
        "<scene></scene>\n"
        "<events></events>\n"
        "<hypotheses></hypotheses>\n"
        "Optional tags (only when useful):\n"
        "<delta_score></delta_score>\n"
        "<summary></summary>\n"
        "Rules:\n"
        "- scene: dense scene description (people, objects, layout, spatial relations).\n"
        "- events: what changed from earlier frames to later frames.\n"
        "- hypotheses: likely user activity/intent grounded in visible evidence.\n"
        "- Describe continuity through time, not separate image variations.\n"
        "- delta_score: numeric 0.0 to 1.0.\n"
        "- summary: one concise sentence.\n"
        "- Never output placeholder text, markdown fences, or format instructions.\n"
        "Optional: <think></think>\n"
    )
    ctx = json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    return f"{contract}\nidentity_scope={policy_identity}\ncontext_json={ctx}"


def build_planner_user_text(
    *,
    context: Dict[str, Any],
    policy_identity: str,
    policy_capabilities: str,
    policy_safety: str,
    policy_style: str,
    planner_prompt: str,
) -> str:
    primitives = [kind.value for kind in PrimitiveKind]
    decision_schema = {
        "act_now": True,
        "primitive": "<one_of_allowed_primitives_or_null>",
        "command": {},
        "style": "calm",
        "confidence": 0.0,
    }
    contract_lines = [
        "You are the PALA planner. Decide what lamp action to take now.",
        "Return exactly one instance of each required tag:",
        "<decision_json></decision_json>",
        "<rationale_short></rationale_short>",
        "Optional: <think></think>",
        "decision_json content must be valid JSON object and use this shape:",
        json.dumps(decision_schema, separators=(",", ":"), ensure_ascii=True),
        f"Allowed primitive values: {primitives}",
        "If no action should be taken now, set act_now=false and primitive=null.",
        'If primitive is "orient_to_zone", command must include zone as one of: left, center, right.',
        'For orient_to_zone, do not use keys like "target", "target_zone", "direction", or "rate". Use "zone", "amp_rad", "rate_rad_s".',
        "Only set act_now=true when initiating or changing behavior.",
        "If the lamp is already doing the same primitive/command/style and scene change is minor, set act_now=false and primitive=null.",
        "Use home only for deliberate reset; do not repeat home without new evidence.",
        "If uncertain, prefer act_now=false over repetitive actions.",
        "Do not output markdown fences or malformed tags.",
    ]
    ctx = json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    policy = {
        "identity": policy_identity,
        "capabilities": policy_capabilities,
        "safety": policy_safety,
        "style": policy_style,
        "planner_prompt": planner_prompt,
    }
    policy_text = json.dumps(policy, separators=(",", ":"), ensure_ascii=True)
    return "\n".join(contract_lines) + f"\npolicy_json={policy_text}\ncontext_json={ctx}"


def build_messages(*, user_text: str, image_data_urls: List[str]) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    for url in image_data_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
