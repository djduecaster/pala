# Fine-Tune Capture Tooling

Dedicated data collection and labeling workflow for Cosmos intent-proposer fine-tuning.

## Components

- Jetson CLI capture: `tools.ft_capture`
- Mac web review/annotation: `tools.ft_capture_web`

## Scenario Catalog

Default catalog:

- `config/ft_scenarios.yaml`

Validate:

```bash
uv run python -m tools.ft_capture validate-catalog --catalog config/ft_scenarios.yaml
```

List scenarios:

```bash
uv run python -m tools.ft_capture list-scenarios --catalog config/ft_scenarios.yaml
```

## Capture (Jetson)

```bash
uv run python -m tools.ft_capture capture \
  --catalog config/ft_scenarios.yaml \
  --scenario walk_into_frame_ack \
  --takes 3 \
  --countdown-s 5 \
  --duration-s 5 \
  --sample-fps 1 \
  --camera-source gst \
  --out-root logs/ft_capture
```

Notes:

- `ffmpeg` must be installed on Jetson to generate `clip.mp4`.
- Capture stores full raw frames and 1 FPS sampled frames for review/export.

## Review (Mac)

```bash
uv run python -m tools.ft_capture_web \
  --dataset-root logs/ft_capture \
  --catalog config/ft_scenarios.yaml \
  --host 127.0.0.1 \
  --port 8790
```

Open:

- `http://127.0.0.1:8790`

## Export

```bash
uv run python -m tools.ft_capture export \
  --catalog config/ft_scenarios.yaml \
  --dataset-root logs/ft_capture \
  --out logs/ft_capture_exports/latest
```

Outputs:

- `dataset_openai.jsonl`
- `dataset_index.jsonl`
- `export_manifest.json`
