#PALA: Programmable Autonomous Lamp Assistant

##Overview
PALA is a physical AI project that runs on an NVIDIA Jetson and uses a fast deploy/run loop from a Mac dev machine. The repo is structured for rapid iteration: edit locally, sync to Jetson, and run on-device with a single command.

PALA uses an IKEA NYMANE lamp, modified to be actuated with 5 degrees of freedom. With custom designed 3d printed servo mounts, a PCA9685 servo hat, and a collection of COTS servos, PALA serves as a minimum viable development platform for at-home functional robotics features.

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
