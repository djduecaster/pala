from __future__ import annotations

import json
from typing import Any, Dict, List


SYSTEM_PROMPT = (
    "You are a deterministic JSON generator for embodied robot behavior. "
    "Return valid JSON only in message content. "
    "No markdown, no code fences, no tags, no prose outside JSON. "
    "First non-whitespace character must be '{' or '['."
)


def build_env_user_text(*, context: Dict[str, Any], policy_identity: str) -> str:
    contract = [
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
    ctx_json = json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    return "\n".join(contract) + f"\nidentity_scope={policy_identity}\ncontext_json={ctx_json}"


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


def build_messages(*, user_text: str, image_data_urls: List[str]) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    for url in image_data_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
