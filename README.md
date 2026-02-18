# PALA: Programmable Autonomous Lamp Assistant

## Overview
PALA is a physical AI project that runs on an NVIDIA Jetson and uses a fast deploy/run loop from a Mac dev machine. The repo is structured for rapid iteration: edit locally, sync to Jetson, and run on-device with a single command.

PALA uses an IKEA NYMANE lamp, modified to be actuated with 5 degrees of freedom. With custom designed 3d printed servo mounts, a PCA9685 servo hat, and a collection of COTS servos, PALA serves as a minimum viable development platform for at-home functional robotics features.

## NVIDIA Competition Focus
This project is designed to score well on the competition criteria by emphasizing:
- **Quality of Ideas**: a clear, compelling application of Cosmos Reason for robotics and video analytics agents.
- **Technical Implementation**: readable, reproducible software with explicit inputs/outputs and evaluation steps.
- **Design**: intuitive runtime modes and a clean operator workflow.
- **Impact**: practical, testable behaviors that advance physical AI in the home.

## Runtime Modes (Planned)
We will run PALA in three explicit modes to enable careful, staged porting:
- `dev`: fully dummy perception + dummy hardware for safe iteration on planning/behavior.
- `jetson_perception`: reserved for a later milestone; details TBD to avoid confusing closed-loop behavior.
- `jetson_full`: full perception + control + actuation on Jetson hardware (near-term focus).

Example commands:
- `uv run python -m pala.main --mode dev`
- `uv run python -m pala.main --mode jetson_perception`
- `uv run python -m pala.main --mode jetson_full`

Mode selection will also be supported via `config/robot.yaml` (single source of truth when set).

## Mode-First Usage
- `config/robot.yaml` is local-safe by default.
- Use `--mode` to switch runtime behavior:
  - `uv run python -m pala.main` (default local/dev)
  - `uv run python -m pala.main --mode jetson_full`

When `--mode dev` is active, runtime forces dummy detector and disables Cosmos.
When `--mode jetson_full` or `--mode jetson_perception` is active and detector is `dummy`, runtime promotes detector to `deepstream`.

## Requirements

### Mac (optional, for enhanced development experience + telemetry viewing)
- macOS + `ssh` access to Jetson (host alias: `jetson`)
- `rsync`
- `make`
- `uv` installed
- Python pinned via `uv` (**3.10.12**)

### Jetson
- `uv` installed (e.g. `~/.local/bin/uv`)
- Python 3.x available (JetPack default is fine)
- Network connectivity and SSH access

### Jetson Camera Setup (GStreamer)
For the Jetson camera backend and FPS tool, install GStreamer Python bindings:
```
sudo apt update
sudo apt install -y python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

If you want the venv to access system packages (recommended on Jetson for `gi`):
```
rm -rf .venv
uv venv --system-site-packages
uv sync
uv pip install -e .
```

Quick camera probe:
```
uv run python tools/test_camera_fps.py --seconds 5 --mode jetson_full
```

### Jetson DeepStream Setup (JetPack 6.x)
DeepStream is a system-level install on Jetson. For JetPack 6.4.x, use the
DeepStream 7.1 arm64 package from NVIDIA NGC:
```
sudo apt-get install ./deepstream-7.1_7.1.0-1_arm64.deb
```

Verify:
```
deepstream-app --version
```

## DeepStream PeopleNet Bring-Up (Validated)

This is the currently working process for `detector: deepstream` in `jetson_full`.

1) Prepare Jetson Python + uv environment:
```
sudo apt-get update
sudo apt-get install -y python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0
cd ~/pala
rm -rf .venv
uv venv --system-site-packages
uv sync
```

2) Install DeepStream Python bindings (`pyds`) for DS 7.1 / cp310:
```
WHEEL_URL=$(python3 -c "import json,urllib.request; d=json.load(urllib.request.urlopen('https://api.github.com/repos/NVIDIA-AI-IOT/deepstream_python_apps/releases/tags/v1.2.0')); print(next(a['browser_download_url'] for a in d['assets'] if a['name'].endswith('linux_aarch64.whl')))")
wget -O /tmp/pyds.whl "$WHEEL_URL"
mv /tmp/pyds.whl /tmp/pyds-1.2.0-cp310-cp310-linux_aarch64.whl
uv pip install /tmp/pyds-1.2.0-cp310-cp310-linux_aarch64.whl
uv run python -c "import gi, pyds; print('gi ok, pyds ok')"
```

3) Build DeepStream custom parser library once (if missing):
```
cd /opt/nvidia/deepstream/deepstream-7.1/sources/libs/nvdsinfer_customparser
sudo make CUDA_VER=12.6
```

4) Use the repo config:
- `config/robot.yaml`
  - `mode: jetson_full`
  - `detector: deepstream`
- `config/deepstream/peoplenet_int8.txt` (key settings)
  - `parse-bbox-func-name=NvDsInferParseCustomResnet`
  - `output-blob-names=output_bbox/BiasAdd:0;output_cov/Sigmoid:0`
  - `model-engine-file=../../../.cache/pala/engines/resnet34_peoplenet_int8.engine`

5) Keep engine cache outside repo (important with `make deploy`):
```
mkdir -p ~/.cache/pala/engines
cp -v ~/pala/models/peoplenet/resnet34_peoplenet_int8.onnx_b1_gpu0_int8.engine ~/.cache/pala/engines/resnet34_peoplenet_int8.engine
```

6) Run:
```
PALA_DS_INFER_TIMEOUT_S=10 PALA_LOG_LEVEL=INFO uv run python -m pala.main
```

Quick detector-only sanity check (no control loop dependencies):
```
uv run python tools/test_detector_stats.py --seconds 20 --mode jetson_full --detector deepstream
```

Manual servo calibration (safe, explicit enable required):
```
uv run python tools/hw_calibrate.py --list-joints
uv run python tools/hw_calibrate.py --enable --neutral --hold-s 1.0
uv run python tools/hw_calibrate.py --enable --joint yaw --deg 5 --hold-s 1.0
uv run python tools/hw_calibrate.py --enable --repl
```

### DeepStream Lessons Learned
- First engine build can take several minutes; short runtime limits can exit before serialization completes.
- `make deploy` uses `rsync --delete`; Jetson-side config edits and in-repo engine files are overwritten/deleted unless committed on Mac or stored outside `~/pala`.
- `zone=center` alone is not proof of detector output; perception falls back to a center dummy bbox when detections are empty.
- `Deserialize engine failed ...` is expected on first run before the engine exists.
- Keep NumPy pinned below 2 for current `pyds` compatibility on this stack.

## Cosmos (Brev) One-Command Bring-Up

On a fresh Brev VM:
```
cd ~/pala
export NGC_API_KEY='...'
./tools/brev_bootstrap_cosmos.sh --replace
```

This script:
- validates GPU/docker prereqs,
- logs into `nvcr.io`,
- starts Cosmos NIM container,
- waits for `http://127.0.0.1:8000/v1/health/ready`.

On Jetson (or Mac in `--mode dev`) verify planner integration:
```
cd ~/pala
export PALA_COSMOS_PROMPT="Prioritize calm, safe desk-companion behavior."
./tools/cosmos_planner_smoke.sh --base-url "http://<BREV_PUBLIC_IP>:8000" --mode dev --seconds 25
```

Characterize image round-trip latency (live capture, 1 Hz default):
```
cd ~/pala
uv run python tools/cosmos_image_probe.py --base-url "http://<BREV_PUBLIC_IP>:8000" --mode jetson_full --count 20
```

Characterize short video-sequence reasoning (multi-frame temporal context):
```
cd ~/pala
uv run python tools/cosmos_video_probe.py --base-url "http://<BREV_PUBLIC_IP>:8000" --mode jetson_full --count 12 --video-window-s 4 --video-max-frames 8
```

Ask Cosmos to describe what is happening in the camera image:
```
cd ~/pala
uv run python tools/cosmos_image_probe.py --base-url "http://<BREV_PUBLIC_IP>:8000" --mode jetson_full --task describe --question "What is happening in this image?" --count 10
```

Pass criteria:
- runtime logs include `cosmos stats requests=... successes=...`,
- new `logs/actions.jsonl` entries include `explanation` prefixed with `cosmos_remote:`.
