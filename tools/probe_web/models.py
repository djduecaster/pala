from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProbeDefaults:
    provider: str
    model: str
    base_url: str
    system_prompt: str
    env_contract: str
    policy_identity: str
    policy_capabilities: str
    policy_safety: str
    policy_style: str
    timeout_s: float
    env_max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    planner_prompt: str
    planner_max_proposals: int
    planner_use_env_context: bool
    planner_max_tokens: int
    planner_temperature: float
    planner_top_p: float
    planner_presence_penalty: float
    planner_system_prompt: str
    planner_image_indices: str
    planner_context_override_json: str
    planner_user_text_override: str
    planner_payload_override_json: str
    planner_prompt_override: str
    inter_frame_ms: float
    packet_view_mode: str
    api_key_source: str
    has_api_key: bool


@dataclass(frozen=True)
class EnvProbeParams:
    provider: str
    model: str
    base_url: str
    system_prompt: str
    env_contract: str
    policy_identity: str
    timeout_s: float
    env_max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    planner_prompt_override: str
    inter_frame_ms: float
    packet_view_mode: str


@dataclass(frozen=True)
class EnvPlannerProbeParams(EnvProbeParams):
    planner_prompt: str
    policy_capabilities: str
    policy_safety: str
    policy_style: str
    planner_max_proposals: int
    planner_use_env_context: bool
    planner_max_tokens: int
    planner_temperature: float
    planner_top_p: float
    planner_presence_penalty: float
    planner_system_prompt: str
    planner_image_indices: str
    planner_context_override_json: str
    planner_user_text_override: str
    planner_payload_override_json: str


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
class EnvProbeRun:
    run_id: str
    created_at_utc: str
    params: Dict[str, Any]
    images: List[Dict[str, Any]]
    packet_compact: Dict[str, Any]
    packet_expanded: Dict[str, Any]
    message_structure: List[Dict[str, Any]]
    request_payload_redacted: Dict[str, Any]
    response_meta: Dict[str, Any]
    raw_content: Optional[str]
    reasoning_content: Optional[str]
    parse_ok: bool
    parse_stage: str
    parse_error: Optional[str]
    parsed_output: Optional[Dict[str, Any]]
    mode: str = "env"
    chain_status: str = "env_only"
    planner_skipped_reason: Optional[str] = None
    effective_inputs: Optional[Dict[str, Any]] = None
    planner_phase: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
