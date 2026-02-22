from __future__ import annotations

import json
from typing import Any, Dict, List


SYSTEM_PROMPT = "You are a helpful assistant."


def build_env_user_text(*, context: Dict[str, Any], policy_identity: str) -> str:
    contract = [
        "You are the PALA environment processor.",
        "Analyze the provided frame sequence (oldest to newest) as one continuous timeline.",
        "Return JSON only matching schema `pala.env_summary.v1`.",
        "Do not output markdown or code fences.",
        "Do not propose actions.",
        "Output exactly these top-level keys: schema_version,scene,events,hypotheses,summary_short,delta_score,features.",
        "features must contain exactly: person_present,zone_hint,activity_level,novelty.",
        "Do not emit alternate keys like description, changes, or inference.",
        "scene describes layout, relevant objects, and spatial relationships.",
        "events focuses on what changed over time in this short window.",
        "hypotheses infers likely user activity from visible evidence.",
        "summary_short is one concise sentence.",
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
        "Return JSON only matching schema `pala.intent_proposals.v1`.",
        "Do not output markdown or code fences.",
        "Output one compact single-line JSON object.",
        f"Return exactly {proposal_count} ranked proposal(s).",
        "Return one concrete, safe next action proposal each cycle.",
        "Avoid indecision language; do not output no-op style text.",
        "Each proposal must include: intent, primitive, command.",
        "Optional keys: style, score, confidence, urgency, risk, allow_interrupt, evidence, rationale_short.",
        "Keep responses terse: evidence <= 1 short token, rationale_short <= 48 chars.",
        "Allowed intents: idle_presence, acknowledge_presence, track_user, scan_environment, react_to_change, reset_pose, affirmation.",
        "Allowed primitives: hold, home, breath, glance, nod, orient_to_zone.",
        "Use home only for explicit reset_pose intent; otherwise prefer breath, glance, or orient_to_zone.",
        "Do not use keys such as likelihood, probability, description, details, or id.",
        "Types are strict: command is an object, evidence is an array of strings, risk is one of low|medium|high, urgency is a number 0..1.",
        "rationale_short must be concise (max 80 chars).",
        "For primitive=hold, set command to {} (an empty object).",
        "Use evidence IDs from context when possible.",
        'For orient_to_zone, command must include zone in {"left","center","right"}.',
        'For breath, command may include amp_rad, period_s, rate_rad_s.',
        'For glance, command may include direction, amp_rad, duration_s, rate_rad_s.',
        'For nod, command may include amp_rad, duration_s, cycles, rate_rad_s.',
        "Output only the JSON object.",
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
