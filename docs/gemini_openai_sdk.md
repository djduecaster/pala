# Model transport diagnostics

The runtime remains hold-only. Model probes run independently and do not
activate planning or actuate the lamp. The next Gemini behavior contract will
be designed after the gesture workshop.

The surviving transport is under `pala/behavior/model_clients/`. It supports
provider-aware URL normalization and chat requests. `tools/model_provider_probe.py`
checks basic text and JSON responses; it no longer uses deleted V4 prompts,
skills, or decision schemas.

```bash
uv run python tools/model_provider_probe.py --help
```

For an explicitly configured model endpoint, use `--provider`, `--base-url`,
and `--model`, or the existing `PALA_MODEL_PROVIDER`, `PALA_COSMOS_BASE_URL`,
and `PALA_COSMOS_MODEL` variables. The probe reads credentials from
`PALA_COSMOS_API_KEY`; keep them in the local/Jetson secret environment, never
in shell examples committed to Git. Running the probe sends requests to the
configured service. A transport result does not establish behavior quality.

Image capture and image/video diagnostics remain in
`tools/capture_cosmos_api_inputs.py`, `tools/cosmos_image_probe.py`, and
`tools/cosmos_video_probe.py`. Their historical names remain for now. Inspect
each tool's help before use; camera source and endpoint must be explicit.

The `cosmos` configuration block is retained for these diagnostic tools.
`pala.main` does not consume its enabled flag to construct a planner. The old
runtime planner smoke script was removed because its assertions referenced
behavior that no longer exists.

Before future API integration, check the selected provider's current model
and parameter documentation. Historical model IDs and quota assumptions are
not a current recommendation. See [Google's compatibility reference](https://ai.google.dev/gemini-api/docs/openai).
