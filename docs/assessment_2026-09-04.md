# PALA assessment — September 4, 2026

PALA can plausibly become a strong portfolio project using its present hardware and software foundation. The missing proof is a small, repeatable interaction that looks intentional on the physical lamp. My recommendation is to finish a camera-driven, expressive desk companion with one short social interaction, then close V1. Treat the remaining work as integration and movement design, with narrowly scoped code changes.

This assessment is based on the current checkout at `804acc5`, the repository's commit history, prior discussions in **Main Pala Builder** and **Ulmo**, selected original conversation records, historical runtime logs, current source inspection, and fresh local verification. It is not physical acceptance: I did not connect to the Jetson, actuate the lamp, test Gemini requests, or see current video of the mechanism. Historical logs are diagnostic examples, not measurements of the current hold-only runtime.

**Where the project stands**

| Area | Verified state | Implication |
|---|---|---|
| Runtime | Four loops; configured at 20/3/80/120 Hz; latest-value communication; supervised thread failures | Preserve the basic architecture |
| Perception | Camera capture, packet identity, freshness, timing, source health; no active semantic detection | There is currently no person recognition or scene understanding in the runtime |
| Behavior | `HoldBehaviorPolicy` returns one persistent hold action | Setting a model configuration flag will not restore intelligent behavior |
| Model support | Provider-neutral transport and JSON utilities remain | Reuse transport; implement a small behavior contract |
| Control | Nine typed primitives, joint clamps, velocity limiting, action IDs, execution status | Useful motion foundation, with execution semantics to tighten |
| Hardware | PCA9685 backend and five-joint calibration mapping exist | Physical calibration, startup posture, and repeatability need current evidence |
| Expression | A separate choreographed demo includes breathing, curious glances, and an excited movement sequence | Recover and tune these assets before expanding the behavior vocabulary |
| Tooling | Telemetry, probes, simulator, and dataset tools survive, including obsolete imports | Some tools are unusable after the reset |
| Documentation | Architecture document describes the reset; README still advertises removed detector paths | A reader cannot reliably infer current capabilities from README alone |

Fresh checks on this Mac:

- `uv run --offline pytest -q`: five collection errors caused by imports of deleted V4 modules.
- `uv run --offline pytest -q --continue-on-collection-errors`: **214 passed, 5 collection errors, 4.06 seconds**. This is not a green full suite.
- `PALA_MAX_RUNTIME_S=4 uv run --offline python -m pala.main`: exit 0; dummy perception around 20 Hz; persistent hold; clean shutdown.
- The checkout was clean before this assessment. No runtime, configuration, deployment, or hardware code was changed for the assessment.

The failing modules are `test_ft_capture_catalog`, `test_ft_capture_export`, `test_ft_capture_schema`, `test_probe_web`, and `test_tools_primitive_sim`. Their dependencies on `decision_schema_v4` and `mode_fsm_v4` survived the deletion of those modules. The older 311-pass campaign snapshot belongs to an earlier state and cannot establish readiness today.

**What the history says**

| Period / checkpoint | What happened | Assessment |
|---|---|---|
| Jan 31–Feb 9 | Deployment, servo port, GStreamer, capture improvements, detector, telemetry, calibration | Concrete integration progress with reusable results |
| Feb 11–18 | Async Cosmos, multimodal frame context, memory, remote-first orchestration | Scope moved from motion toward a general embodied agent |
| Feb 20–23 | Memory redesign, environment summarizer/planner split, V3, Gemini transport, hybrid local/model behavior | Several interacting explanations for failure changed at once |
| Feb 25–28 | Simulator and behavior expansion, then V4 at `0d0646b` | More explicit structure, but still a large semantic contract |
| Aug 5, `2159212` | Expressive movement demo checkpoint | A useful, separable asset for finishing V1 |
| Aug 5, `804acc5` | Capture-only / hold-only reset, 4,630 deletions across the reset change | Sensible simplification; integration and tooling cleanup remain incomplete |

Dates above are Git commit dates. Conversation chronology and remembered milestones should not substitute for those dates.

**Where the process went wrong**

1. **The definition of success expanded faster than physical proof.** Earlier conversations grew from attention and movement into identity, persistent memory, scene/event interpretation, search, speech, task lighting, and competition presentation. These are individually reasonable ideas. Together they made a convincing greeting depend on too many unfinished capabilities. Even recent advice described a “small” MVP with five modes, seven skills, reading recognition, and possible tracking. I would narrow that further.

2. **“Model collapse” became a name for several different failures.** In `logs/runs/20260222_010437`, there are 36 planner starts and 35 recorded completions: 14 parse failures and 21 successes. Eighteen successful top proposals are `home`. The 197 behavior trace records include 143 `same_signature` and 46 `utility_below_threshold` results. The action log contains 142 `home` entries, but only five distinct action IDs overall; repeated log rows are not repeated physical gestures. Most requests did contain an image (35 of 36), so this example is not simply missing vision. It demonstrates schema failures, repetitive model preference, and local arbitration effects in the same run. It does not prove one shared cause or a defect specific to Gemini.

3. **Authority was split in ways that could cancel the intended intelligence.** At V4, `_signals_from_perception` read `debug.person_present` and `debug.person_conf`. The perception node instead published the person through `primary_person` and `primary_person_conf`, with different debug fields. Meanwhile, `ActionGuard` rejected a decision whose mode differed from the FSM's current mode. Thus, the normal perception path did not supply the fields required for social transitions, and the model could not simply choose its way out. This is a concrete integration mismatch, not a prompting problem. References: `git show 0d0646b:pala/behavior/policy_v4.py`, `pala/perception/node.py`, and `pala/behavior/action_guard.py` at that commit.

4. **A motion vocabulary was treated as if it were already a character.** Current styles primarily scale amplitude, speed, duration, and settling. There is already some intentional shaping in glance/nod/breath. But “curious” still needs a recognizable physical performance: what leads, where the head points, how long it pauses, and how it responds to a person's action. More semantic labels do not establish that performance.

5. **Software validation outpaced experience validation.** Tests establish useful contracts. They cannot establish that a gesture looks friendly, the camera retains sight of a person during it, or the mechanism settles cleanly. The large surrounding tool investment is visible: approximately 22,480 tracked Python lines under `tools`, versus 3,304 under `pala`, excluding tests. This is not inherently bad, but another dashboard or generalized probe should now have to justify itself against finishing the demonstration.

6. **The assistants contributed to the scope growth.** Repeated requests for a system that felt smarter were met with major architecture changes and increasingly detailed frameworks. A better intervention would have been to isolate one failed observation-to-motion sequence and verify its cause. This is a process lesson for our collaboration, not just a critique of your decisions.

**The V1 I recommend**

Product sentence: **PALA is a curious, quiet desk lamp that notices you, acknowledges you with expressive movement, and settles when the interaction ends.**

Use one person, a fixed desk arrangement, ordinary indoor lighting, and a demonstrated viewing region. The shipped interaction is: wake into an observing pose; notice a person; orient coarsely toward them; greet once; attend while they remain; settle after they leave. Treat coarse orientation as the requirement. Do not promise smooth continuous person tracking without demonstrating it.

Reading-light repositioning, speech, finding objects, persistent autobiographical memory, fine-tuning, a new detector stack, full inverse kinematics, and hardware V2 remain outside this release. In particular, a reading task adds localization and functional illumination claims to a project whose immediate objective is character. It can be an extension after release.

Choose a consistent personality: curious, restrained, slightly playful. Make silence and stillness legitimate. Avoid requiring the model to invent a new action every tick or penalizing all repeated hold decisions; doing nothing can be correct when nothing changes.

| Movement | Intended reading | Initial design hypothesis |
|---|---|---|
| Wake / notice | Attention shifts to the environment or person | A small preparatory movement, head orientation, then a clear pause |
| Greeting | Recognition and warmth | One controlled acknowledgment with a clean recovery |
| Curious attention | Interest without agitation | A modest tilt or lean, target-directed attention, and enough stillness to read it |
| Settle | Interaction has ended | A slower return to a known observing/resting posture |

These are hypotheses to test on the body, not prescribed angles. The existing demo is the starting asset. Five degrees of freedom provide several expressive choices, but not every gesture should use every joint. Camera placement, gravity, backlash, and lamp proportions must decide which motions survive.

Apple's ELEGNT research is a directly relevant reference: it studies a lamp-like robot and compares expressive versus function-driven movements in six scenarios, reporting improved engagement and perceived robot qualities for expressive movements. My application to PALA is to design and evaluate movement deliberately before expecting model choice to create character. This is not evidence that any particular PALA gesture will succeed. [ELEGNT, Apple Machine Learning Research](https://machinelearning.apple.com/research/elegnt-expressive-functional-movement)

**How Gemini fits**

Keep your prior choice to leave local person detection out initially. Give Gemini the latest usable image, a concise description of the lamp, the few legal skills, and runtime-owned facts: current skill, its execution status, whether a greeting already occurred in this interaction, and recent accepted outcomes. Request one observation and one next skill choice. Keep motor coordinates out of the model response.

The model may choose `observe`, `greet`, `attend`, or `settle`, with a coarse target region where appropriate. Fixed choreography handles wake-up. “Continue” must preserve the existing action identity. Runtime bookkeeping owns skill progress, duplicate suppression, completion, faults, and a conservative re-arm rule for greeting after confirmed absence. The model supplies the semantic interpretation; the runtime enforces execution and objective validity.

Preserve `PerceptionState -> ActionPlan -> HardwareCommand`. Add a small internal decision object and skill runner inside behavior, plus a thread-safe execution snapshot from control. Do not hide semantic state in debug fields. Keep one async request in flight and consume the freshest context at the next dispatch. Reject results whose image or execution context is obsolete. The configured 3 Hz behavior tick is not a promise that the remote model will complete three requests per second.

Google currently documents access to Gemini through the OpenAI-compatible API, so the existing transport is a reasonable starting point. Actual model choice and cadence should follow a benchmark with your camera and intended scenes; this review made no paid model calls and establishes no latency claim. [Google Gemini compatibility documentation](https://ai.google.dev/gemini-api/docs/openai)

The camera may move with the lamp. If so, an image-space location is tied to the pose at capture. An old “left” observation after the lamp turns must not be reused blindly as a new leftward instruction. Start with coarse orientation and reacquisition from an observing pose. Confirm the camera mount before adding coordinate transforms.

**The small set of engineering issues that matter first**

- **Execution continuity:** the executor's docstring says actions run to completion unless canceled, but `_maybe_activate` replaces a different intent without consulting `cancel_current`. Decide the interruption contract and test it before a model chooses multi-part gestures. The behavior loop currently receives no executor status.
- **Commanded versus measured motion:** `_current` begins at zeros and is a software estimate. There is no joint-position feedback in the servo interface. `DONE` means the commanded trajectory reached its condition; it does not establish physical arrival. Startup `hold` sends the estimated zero pose, so physical startup posture matters.
- **Calibration consistency:** all five software ranges are ±1.57 radians. For `pitch2`, the configured conversion is `servo_deg = 2 * joint_deg + 50`, then clamped to 0–180 degrees. Its unsaturated mathematical range is only approximately −25 to +65 joint degrees. This does not establish safe physical limits; it shows why a simulator and real motion can disagree even when both “clamp correctly.” Reconcile the mappings against previously tested positions and current hardware.
- **Movement quality:** the executor limits velocity per tick, but has no general acceleration/jerk limit. Tune choreography first, then add bounded easing if filmed motion identifies abrupt starts or stops as a problem. Do not start a full motion-planning rewrite.
- **Failure handling:** the deadman is a software check in the hardware loop, not an independent electrical watchdog. It catches stale control commands while that loop is running. Establish what shutdown and torque removal actually do to this lamp; do not automatically equate disabled PWM with a physically safe resting state.
- **Release integrity:** repair or retire the obsolete tools deliberately and make the documented full test command honest. Refresh README to match the reset and eventual MVP. Avoid resurrecting V4 solely to satisfy old tests.

These issues have direct connections to repeatability or demo trust. The remaining bug log can stay a backlog unless an item affects the selected path. The expressive demo is a separate direct-servo runner; do not assume it inherits the main runtime's four-loop deadman behavior. Also, it constructs the configured servo backend even with `--dry-run`; a Jetson dry-run is not guaranteed to avoid hardware initialization.

**A bounded completion sequence**

| Step | Deliverable | Exit condition |
|---|---|---|
| 1. Verify the body | Short third-person clips plus matching camera view; known starting posture | Three useful gestures repeat without obvious collision, binding, or unacceptable oscillation; camera coverage understood |
| 2. Establish release baseline | Small cleanup of current-path tests/docs and critical execution/calibration inconsistencies | Documented test suite and dummy runtime pass; hardware assumptions recorded |
| 3. Prove model choices | Exact production prompt/schema on representative images and short sequences, with motion disabled | Can distinguish arrival/presence/departure; latency and rejection causes measured |
| 4. Connect one interaction | Observe → greet once → attend → settle | Repeated requests preserve gesture execution; stale responses do not restart motion; absence re-arms greeting appropriately |
| 5. Package and stop | Tagged snapshot, reproducible instructions, short case study, 60–90 second video | Demonstrated behavior matches claims; limitations and trigger method are clear |

Use roughly ten focused work sessions as a planning budget, not a delivery guarantee. Assess after the first two sessions whether mechanical reliability is the dominant blocker. If it is, reduce the motion envelope or number of gestures before expanding software. If Gemini cannot meet the timing needed for coarse attention after a small measured tuning pass, either explicitly reduce the demo claim or revisit a narrow local tracker as a separate decision. Do not spend the release budget cycling through providers and architectures.

Proposed acceptance targets, to agree before testing: at least 9 of 10 full interaction trials without operator correction; one greeting per continuous interaction; no false greeting in a five-minute empty scene; a ten-minute run without a crash; and documented rejection of malformed/stale model results. Begin with a two-second target for visible acknowledgment after a person becomes clearly visible and stationary in the demonstrated region, then measure its feasibility. Report the actual distribution and misses. These are V1 targets, not results or a general safety certification.

For expression, show unlabeled gesture clips to a few people and ask what they think the lamp is doing. If observers cannot distinguish greeting from curiosity, adjust posture and timing before adding more model context. This is formative feedback, not a statistically powered user study.

For the portfolio, explain the modified IKEA body, five-axis actuation, perception/behavior/control separation, the concrete failure mechanisms found during iteration, and the measured final behavior. Include an uncut interaction take alongside any edited overview. Label scripted startup, live model decisions, and operator-triggered material accurately. The value is a finished system whose boundaries you can explain.

**What would help next**

A current 30–60 second third-person video is the most useful missing evidence: full lamp and base visible, a familiar gentle motion sequence, normal audio, and a clear view of how it settles. Add a matching camera-view sample and photos showing the five axes and camera mount. Existing clips are sufficient to start; this assessment does not authorize or require new unattended hardware motion. Also identify any joint that slips, binds, heats, or loses its expected pose.

My next implementation recommendation is to extract and tune three reusable physical performances, then connect Gemini to those performances with the smallest execution-aware behavior loop. The first milestone is a convincing greeting that works repeatedly.
