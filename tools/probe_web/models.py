from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProbeDefaults:
    provider: str
    model: str
    base_url: str
    system_prompt: str
    timeout_s: float
    max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    frame_max_width: int
    frame_jpeg_quality: int
    policy_identity: str
    policy_capabilities: str
    policy_safety: str
    policy_style: str
    planner_prompt: str
    context_override_json: str
    user_text_override: str
    payload_override_json: str
    inter_frame_ms: float
    packet_view_mode: str
    min_action_dwell_s: float
    stale_after_s: float
    min_mode_dwell_s: float
    engage_person_conf: float
    disengage_person_conf: float
    api_key_source: str
    has_api_key: bool


@dataclass(frozen=True)
class BehaviorProbeParams:
    provider: str
    model: str
    base_url: str
    system_prompt: str
    timeout_s: float
    max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    frame_max_width: int
    frame_jpeg_quality: int
    policy_identity: str
    policy_capabilities: str
    policy_safety: str
    policy_style: str
    planner_prompt: str
    context_override_json: str
    user_text_override: str
    payload_override_json: str
    inter_frame_ms: float
    packet_view_mode: str


@dataclass(frozen=True)
class PreparedImage:
    filename: str
    content_type: str
    original_width: int
    original_height: int
    encoded_width: int
    encoded_height: int
    jpeg_bytes: int
    data_url: str


@dataclass
class BehaviorProbeRun:
    run_id: str
    created_at_utc: str
    params: Dict[str, Any]
    images: List[Dict[str, Any]]
    packet_compact: List[Dict[str, Any]]
    packet_expanded: List[Dict[str, Any]]
    message_structure: List[Dict[str, Any]]
    request_payload_redacted: Dict[str, Any]
    response_meta: Dict[str, Any]
    raw_content: Optional[str]
    reasoning_content: Optional[str]
    parse_ok: bool
    parse_stage: str
    parse_error: Optional[str]
    parsed_output: Optional[Dict[str, Any]]
    guard_result: Optional[Dict[str, Any]]
    final_action: Optional[Dict[str, Any]]
    mode: str = "behavior_v4"
    effective_inputs: Optional[Dict[str, Any]] = None
    fsm_before: Optional[Dict[str, Any]] = None
    fsm_after: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
