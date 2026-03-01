from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = (
    "You are a deterministic JSON generator for embodied robot behavior. "
    "Return valid JSON only in message content. "
    "No markdown, no code fences, no tags, no prose outside JSON. "
    "First non-whitespace character must be '{' or '['."
)

_MODE_GUIDANCE: Dict[str, List[str]] = {
    "boot_awaken": [
        "BOOT_AWAKEN: complete wake-up expression, then settle to stable observation.",
        "Prefer gentle orient, nod, breath; avoid abrupt transitions.",
    ],
    "idle_presence": [
        "IDLE_PRESENCE: keep calm aliveness with sparse accents.",
        "Prefer breath/hold, with occasional glance or slow orient.",
    ],
    "social_interact": [
        "SOCIAL_INTERACT: acknowledge user presence and maintain readable attention.",
        "Prioritize greet_user/social_ack style actions with stable zone orientation.",
    ],
    "search_assist": [
        "SEARCH_ASSIST: perform deliberate search and confirmation behavior.",
        "Prefer expressive_search or point_and_hold depending on confidence.",
    ],
    "task_lighting": [
        "TASK_LIGHTING: optimize supportive lighting posture for active task.",
        "Prefer focused orient/hold adjustments and avoid unnecessary motion.",
    ],
    "return_home": [
        "RETURN_HOME: converge back to neutral home posture smoothly.",
        "Use home/hold actions; do not add extra expressive motion.",
    ],
    "recover_reset": [
        "RECOVER_RESET: fail-closed and stable; prioritize safe, low-risk behavior.",
        "Use hold/home/breath only until health recovers.",
    ],
}


def env_contract_lines() -> List[str]:
    return [
        "You are the PALA environment summarizer.",
        "The camera view is my view as the lamp.",
        "Use egocentric framing (left/center/right relative to me).",
        "Analyze the provided frame sequence as a short timeline.",
        "Return JSON only matching schema `pala.env_summary.v1`.",
        "Top-level keys must be exactly: schema_version,scene,events,hypotheses,summary_short,delta_score,features.",
        "features keys must be exactly: person_present,zone_hint,activity_level,novelty.",
        "scene must start with 'I see the scene as'.",
        "events must start with 'I notice'.",
        "hypotheses must start with 'I infer'.",
        "Keep string fields concise and factual.",
        "Do not include recommendations or action proposals.",
    ]


def build_env_user_text(*, context: Dict[str, Any], policy_identity: str, contract_override: str | None = None) -> str:
    if isinstance(contract_override, str) and contract_override.strip():
        contract_text = contract_override.strip()
    else:
        contract_text = "\n".join(env_contract_lines())
    ctx_json = json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    return contract_text + f"\nidentity_scope={policy_identity}\ncontext_json={ctx_json}"


def build_planner_user_text(
    *,
    context: Dict[str, Any],
    policy_identity: str,
    policy_capabilities: str,
    policy_safety: str,
    policy_style: str,
    planner_prompt: str,
    max_proposals: int,
) -> str:
    proposal_count = max(1, int(max_proposals))
    contract = [
        "You are the PALA intent proposer.",
        "The camera view is my view as the lamp.",
        "Return JSON only matching schema `pala.intent_proposals.v2`.",
        "Return compact minified JSON on a single line.",
        f"Return exactly {proposal_count} proposals ranked best to worst.",
        "Every proposal must include all required fields from the schema.",
        "Do not duplicate primitive values across all proposals unless no safe alternative exists.",
        "If person_present is true in context, do not return all-idle proposals.",
        "Use the current mode in context_json to shape your proposal choices.",
        "Allowed intents: idle_presence, acknowledge_presence, track_user, scan_environment, react_to_change, reset_pose, affirmation.",
        "Allowed primitives: hold, home, breath, glance, nod, orient_to_zone.",
        "For orient_to_zone, command.zone must be left|center|right.",
        "For glance, command.direction must be left|right|up|down.",
        "Use terse rationale_short and compact evidence.",
        "Do not output unknown enum values.",
    ]
    policy = {
        "identity": policy_identity,
        "capabilities": policy_capabilities,
        "safety": policy_safety,
        "style": policy_style,
        "planner_prompt": planner_prompt,
    }
    policy_json = json.dumps(policy, separators=(",", ":"), ensure_ascii=True)
    ctx_json = json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    return "\n".join(contract) + f"\npolicy_json={policy_json}\ncontext_json={ctx_json}"


def build_behavior_v4_user_text(
    *,
    context: Dict[str, Any],
    policy_identity: str,
    policy_capabilities: str,
    policy_safety: str,
    policy_style: str,
    planner_prompt: str,
    mode_guidance: Optional[List[str]] = None,
) -> str:
    contract = [
        "You are the PALA behavior planner for a desk companion lamp.",
        "The camera view is my view as the lamp.",
        "Return JSON only matching schema `pala.behavior_decision.v1`.",
        "No markdown, no code fences, no tags.",
        "Always provide keys: schema_version,mode,mood,skill,action,confidence,rationale_short,mode_transition.",
        "Choose one best next action for the current mode and active skill.",
        "Set mode_transition to either 'stay' or 'to_<mode>'.",
        "Keep rationale_short concise and concrete.",
        "Only use primitives allowed by schema and provided mode context.",
        "Use calm motion unless context strongly supports curious/focused.",
        "When uncertain, choose low-risk stable behavior (hold or gentle breath).",
    ]
    if mode_guidance:
        contract.extend([str(line).strip() for line in mode_guidance if str(line).strip()])
    policy = {
        "identity": policy_identity,
        "capabilities": policy_capabilities,
        "safety": policy_safety,
        "style": policy_style,
        "planner_prompt": planner_prompt,
    }
    policy_json = json.dumps(policy, separators=(",", ":"), ensure_ascii=True)
    ctx_json = json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    return "\n".join(contract) + f"\npolicy_json={policy_json}\ncontext_json={ctx_json}"


def build_messages(*, user_text: str, image_data_urls: List[str]) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    for url in image_data_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def behavior_v4_mode_guidance(mode: str, active_skill: str) -> List[str]:
    mode_token = str(mode or "").strip().lower()
    skill_token = str(active_skill or "").strip().lower()
    lines = list(_MODE_GUIDANCE.get(mode_token, []))
    if skill_token:
        lines.append(f"Current skill focus: {skill_token}.")
    return lines
