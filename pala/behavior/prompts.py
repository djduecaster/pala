from __future__ import annotations

import json
from typing import Any, Dict, List


SYSTEM_PROMPT = (
    "You are a deterministic JSON generator for robot behavior planning. "
    "Return only valid JSON in message content. "
    "Do not output markdown, backticks, XML tags, or prose outside JSON. "
    "The first non-whitespace character must be '{' or '['."
)


def build_env_user_text(*, context: Dict[str, Any], policy_identity: str) -> str:
    contract = [
        "You are the PALA environment processor.",
        "The camera view is my view. I am the lamp observing this scene.",
        "Analyze the provided frame sequence (oldest to newest) as one continuous timeline.",
        "Return JSON only matching schema `pala.env_summary.v1`.",
        "Output one compact single-line JSON object. No markdown or fences.",
        "Do not include newline characters in any string value.",
        'Do not include quoted speech or embedded quote characters (\" or \') in string values.',
        "Do not propose actions.",
        "Output exactly these top-level keys: schema_version,scene,events,hypotheses,summary_short,delta_score,features.",
        "features must contain exactly: person_present,zone_hint,activity_level,novelty.",
        'zone_hint must be one of {"left","center","right"} when person_present=true; use "unknown" only if no person is visible.',
        "scene should start with 'I see the scene as ...' and describe layout/relative positions from my viewpoint.",
        "events should start with 'I notice ...' and focus on short-window changes.",
        "hypotheses should start with 'I infer ...' and infer likely user activity from visible evidence.",
        "Keep it terse: scene <= 180 chars, events <= 120 chars, hypotheses <= 120 chars.",
        "summary_short is one concise sentence <= 80 chars.",
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
        "The camera view is my view. I am the lamp and must choose my next body action.",
        "Return JSON only matching schema `pala.intent_proposals.v1`.",
        "Output one compact single-line JSON object. No markdown or fences.",
        f"Return exactly {proposal_count} ranked proposal(s), best first.",
        "If returning more than one proposal, ensure the last proposal is a safe fallback (hold or low-amplitude breath/glance).",
        "At least one proposal must be non-idle (primitive must be one of: orient_to_zone, glance, nod, home).",
        "Return concrete action proposals for what I should do next now.",
        "Use egocentric language internally: person relative to me (left/center/right, closer/farther to me).",
        "Each proposal must include: intent, primitive, command, style, score, confidence, urgency, risk, allow_interrupt, evidence, rationale_short.",
        "Do not repeat the same primitive across all proposals unless no other safe option exists.",
        "Keep responses terse: evidence <= 1 short token, rationale_short <= 48 chars.",
        "Allowed intents: idle_presence, acknowledge_presence, track_user, scan_environment, react_to_change, reset_pose, affirmation.",
        "Allowed primitives: hold, home, breath, glance, nod, orient_to_zone.",
        "Use home only for explicit reset_pose intent.",
        "Types: command object, evidence array of strings, risk low|medium|high, urgency 0..1.",
        "For primitive=hold, set command to {} (an empty object).",
        "Use evidence IDs from context when possible and avoid extra keys.",
        "If zone_hint is missing in context_json, infer a zone from the image; never output zone='unknown'.",
        'For orient_to_zone, command must include zone in {"left","center","right"}.',
        'For glance, command.direction must be one of {"left","right","up","down"}; never use "center".',
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
