# PALA documentation

PALA V1 is a supervised physical desk-companion demo. The default runtime holds
position; the opt-in live interaction uses camera snapshots and Gemini observations
to select locally controlled gestures. See the [project overview](../README.md).

| Guide | Purpose |
|---|---|
| [Recreating PALA](recreating_pala.md) | Parts, calibration, setup and remaining build details |
| [Licensing](licensing.md) | MIT software license and media/CAD scope |
| [Architecture](architecture.md) | Four loops, data contracts, model boundary, and safety limits |
| [Live interaction](live_interaction.md) | Current camera-driven V1 demo and rehearsal evidence |
| [Portfolio media](portfolio_media.md) | Demo release and ignored local production archive |
| [POV recording](pov_recording.md) | Optional live camera recording and stationary intro capture |
| [Manual interaction](manual_interaction.md) | Operator-triggered performances without Gemini |
| [Gesture workshop](gesture_workshop.md) | Tune and review individual physical performances |
| [Simulator](../tools/primitive_sim/README.md) | Local geometry, primitive tuning, and scripted playback |
| [Jetson workflow](jetson_agent_workflow.md) | SSH, tmux, deployment, and shared-shell conventions |
| [Telemetry](../tools/telemetry/README.md) | Optional camera/command viewer and historical replay |
| [Attention probe](attention_probe.md) | Stationary-view model diagnostic without model-triggered motion |
| [Single greeting test](live_greeting.md) | Earlier, explicitly armed one-greeting diagnostic |
| [Model transport](gemini_openai_sdk.md) | Generic endpoint diagnostics and credential conventions |
| [Memory boundary](memory_architecture.md) | Session bookkeeping and excluded persistent memory |
| [Workshop evidence](workshop_2026-09-07.md) | Dated operator ratings for specific gesture versions |
| [Known issues](bug_log.md) | Dated review backlog; entries are not newly reproduced findings |

Simulation and command traces do not measure physical joint position. Historical
ratings apply to the recorded recipe version, not automatically to later tuning.
Old competition, migration, and session-planning documents are outside the current
published guide set; source history preserves their context.
