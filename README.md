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
