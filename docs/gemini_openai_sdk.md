# Gemini via OpenAI SDK (Short Guide)

## Compatibility status

Your pasted guidance is directionally correct:

- Gemini supports OpenAI-compatible APIs.
- The common migration is effectively three knobs:
  - `api_key=GEMINI_API_KEY`
  - `base_url=https://generativelanguage.googleapis.com/v1beta/openai/`
  - `model=gemini-*`

## Minimal Python example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

resp = client.chat.completions.create(
    model="gemini-3-flash-preview",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## PALA routing support

PALA transport now supports provider-aware endpoint routing:

- `provider=auto` (default) infers Gemini from `generativelanguage.googleapis.com`.
- `provider=gemini` routes base URLs to `.../v1beta/openai/chat/completions`.
- `provider=openai`/`provider=cosmos` routes to `.../v1/chat/completions`.

You can set provider with either:

- `cosmos.provider` in `config/robot.yaml`
- env var `PALA_MODEL_PROVIDER` (or `PALA_COSMOS_PROVIDER`)

## Notes that match your pasted guidance

- `reasoning_effort` is supported on Gemini compatibility endpoints.
- Gemini-specific controls can be passed via `extra_body` (for example `google.thinking_config`).
- Streaming and tool calling are supported on compatibility endpoints.

## Practical next step for PALA

Add provider-aware URL routing in transport:

- Cosmos/OpenAI-style: `.../v1/chat/completions`
- Gemini OpenAI-compat: `.../v1beta/openai/chat/completions`

Then run the same probe/runtime tests for A/B:

1. latency (p50/p90)
2. schema compliance
3. behavior quality

## New helper tools in this repo

Provider probe:

```bash
uv run python tools/model_provider_probe.py --runs 1 --verbose
```

If you are on Gemini free tier, keep `--runs 1` (or add `--sleep-s`) to avoid quick 429 quota hits.

Env wizard (interactive):

```bash
bash tools/pala_env_wizard.sh
source ~/.config/pala/env.sh
```

## References

- Gemini OpenAI compatibility docs: https://ai.google.dev/gemini-api/docs/openai
- Vertex AI OpenAI compatibility docs: https://cloud.google.com/vertex-ai/generative-ai/docs/start/openai
