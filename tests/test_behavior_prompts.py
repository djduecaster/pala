from __future__ import annotations

import json

from pala.behavior.prompts import SYSTEM_PROMPT, build_env_user_text, build_messages, build_planner_user_text
from pala.types import PrimitiveKind


def test_build_env_user_text_contains_contract_and_serialized_context():
    context = {"recent_events": [{"summary": "user moved"}], "person_conf": 0.8}
    text = build_env_user_text(context=context, policy_identity="PALA observer")

    required_tags = [
        "<scene></scene>",
        "<events></events>",
        "<hypotheses></hypotheses>",
        "<delta_score></delta_score>",
        "<summary></summary>",
    ]
    for tag in required_tags:
        assert tag in text
    assert "Do not propose actions." in text
    assert "Never output placeholder text" in text
    assert "identity_scope=PALA observer" in text

    context_json = text.split("context_json=", 1)[1]
    assert json.loads(context_json) == context


def test_build_planner_user_text_contains_contract_and_all_primitives():
    context = {"current_action": "hold"}
    text = build_planner_user_text(
        context=context,
        policy_identity="PALA planner",
        policy_capabilities="caps",
        policy_safety="safety",
        policy_style="style",
        planner_prompt="prompt",
    )

    assert "<decision_json></decision_json>" in text
    assert "<rationale_short></rationale_short>" in text
    assert "If no action should be taken now" in text
    assert "Use home only for deliberate reset" in text
    for kind in PrimitiveKind:
        assert kind.value in text

    tail = text.split("policy_json=", 1)[1]
    policy_json, context_json = tail.split("\ncontext_json=", 1)
    policy = json.loads(policy_json)
    assert policy["identity"] == "PALA planner"
    assert policy["planner_prompt"] == "prompt"
    assert json.loads(context_json) == context


def test_build_messages_orders_media_before_text():
    messages = build_messages(
        user_text="hello",
        image_data_urls=["data:image/jpeg;base64,a", "data:image/jpeg;base64,b"],
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert isinstance(messages[0]["content"], str)
    assert messages[0]["content"] == SYSTEM_PROMPT
    user = messages[1]["content"]
    assert user[0]["type"] == "image_url"
    assert user[1]["type"] == "image_url"
    assert user[0]["image_url"]["url"] == "data:image/jpeg;base64,a"
    assert user[1]["image_url"]["url"] == "data:image/jpeg;base64,b"
    assert user[2]["type"] == "text"
    assert user[2]["text"] == "hello"


def test_build_messages_without_images_contains_text_only():
    messages = build_messages(user_text="only text", image_data_urls=[])
    assert len(messages[1]["content"]) == 1
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][0]["text"] == "only text"
