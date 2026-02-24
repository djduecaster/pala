from __future__ import annotations

import json

from pala.behavior.prompts import SYSTEM_PROMPT, build_env_user_text, build_messages, build_planner_user_text


def test_build_env_user_text_requires_json_contract():
    context = {"recent_events": [{"summary": "user moved"}], "person_conf": 0.8}
    text = build_env_user_text(context=context, policy_identity="PALA observer")

    assert "The camera view is my view as the lamp." in text
    assert "Return JSON only matching schema `pala.env_summary.v1`" in text
    assert "Do not include recommendations or action proposals." in text
    assert "scene must start with 'I see the scene as'." in text
    assert "identity_scope=PALA observer" in text

    context_json = text.split("context_json=", 1)[1]
    assert json.loads(context_json) == context


def test_build_planner_user_text_requires_ranked_proposals():
    context = {"current_action": "hold"}
    text = build_planner_user_text(
        context=context,
        policy_identity="PALA planner",
        policy_capabilities="caps",
        policy_safety="safety",
        policy_style="style",
        planner_prompt="prompt",
        max_proposals=3,
    )

    assert "Return JSON only matching schema `pala.intent_proposals.v2`" in text
    assert "Return exactly 3 proposals ranked best to worst." in text
    assert "The camera view is my view as the lamp." in text
    assert "Use the current mode in context_json to shape your proposal choices." in text

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
    assert messages[0]["content"] == SYSTEM_PROMPT

    user = messages[1]["content"]
    assert user[0]["type"] == "image_url"
    assert user[1]["type"] == "image_url"
    assert user[2]["type"] == "text"
    assert user[2]["text"] == "hello"
