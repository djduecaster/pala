# PALA x Cosmos Reason 2 — Strategy Context for Expert Review

> Historical February 2026 competition snapshot. The V3/V4 systems described
> below were subsequently removed. Its deadlines and measurements are not
> current runtime claims. See `docs/architecture.md` and `docs/todo.md` for the
> current baseline and portfolio milestone.

## 1) Project Backstory and Goal

PALA is a physical AI desk companion lamp running on Jetson hardware.  
The intended architecture is:

- Fast local loops for perception + control + hardware safety
- Slower high-level behavior decisions from a remote VLM (Cosmos Reason 2 originally; now testing Gemini for cost/iteration speed)

Primary goal for the submission: **a repeatable, reliable, impressive live demo** that still clearly demonstrates model-driven semantic behavior (not just scripted motion).

Current timeline constraint: **~1 week remaining**.

---

## 2) Runtime Architecture (Current)

Core loops (kept stable):

- Perception loop: high rate
- Behavior loop: lower rate
- Control loop: high rate
- Hardware loop: high rate + deadman safety

Behavior stack (V3-style):

- Remote env summarizer + remote intent proposer (both non-blocking, single in-flight each)
- Deterministic governor + arbiter
- Idle/presence engine for anti-dead-lamp behavior
- Health manager + breakers
- Committed-only memory writes (no non-committed decision contamination)

Key principle in code now: **remote semantics are proposals; deterministic layer decides safe/timed commit**.

---

## 3) Why Provider Switched (for now)

Cosmos/NIM integration worked but had iteration friction and cost/latency concerns for rapid tuning.
Gemini OpenAI-compat path was added to keep architecture intact while accelerating experiments.

Provider-agnostic transport + probe tooling now exist.

---

## 4) Current Measured Status (Recent Runs)

## 4.1 Provider probe (Gemini Flash-Lite) looked clean

From `logs/provider_probe/20260223_133022/report.json`:

- text_ping: 3/3 ok + parse_ok
- json probe: 3/3 ok + parse_ok
- planner probe: 3/3 ok + parse_ok
- planner latency ~1.07–1.27s

Interpretation: transport/auth/basic structured output path is viable.

## 4.2 API loop probe looked integration-healthy

From `logs/cosmos_api_probe/20260223_140820/summary.json`:

- Planner parse_ok: 76/80 (95%)
- Env parse_ok: 78/80 (97.5%)
- Latency p50 ~1.07–1.17s

Interpretation: enough reliability to move from API probing to live behavior tuning.

## 4.3 Live run trend

### Run A: `logs/runs/20260223_152100`

- Planner: 77/79 ok (~97.5%)
- Env: 17/17 ok
- Behavior still mixed with idle + orient patterns
- Zone signal weak (`zone_hint` mostly unknown)

### Run B: `logs/runs/20260223_153644`

- Planner: 44/44 ok
- Env: 8/9 ok
- Strong improvement in non-idle commits:
  - `orient_to_zone`: 36
  - `glance`: 7
  - `nod`: 1
  - `hold`: 1
- Zones became meaningful (left/center/right)

### Run C: `logs/runs/20260223_154554` (latest)

- Planner: 68/70 ok (97.1%), p50 latency ~1.31s
- Env: 53/61 ok (86.9%), p50 latency ~1.11s
- Action commits: 71 unique IDs, almost all `orient_to_zone` (70), little diversity
- `zone_hint` in env improved strongly (mostly left/center/right, rare unknown)
- Remaining issue: env parse errors from malformed/truncated JSON fragments

Interpretation of latest:

- **Integration is good enough**.
- **Behavior quality bottleneck remains** (over-convergence on one primitive).
- Env still needs robustness hardening, but no longer total blocker.

---

## 5) What Was Recently Changed

- Stricter proposal schema and parser behavior (fail closed on invalid mixed proposals)
- Planner prompt tightened:
  - requires at least one non-idle proposal
  - forbids invalid glance direction (`center`)
- Env cadence target increased to 1.0 Hz in config
- Env prompt compressed/strictened to reduce malformed output
- Zone fallback inference added from env text when explicit zone missing
- Info-level runtime logs now include planner response summary + env summary line items

---

## 6) Core Decision Under Debate

How much deterministic state logic should be introduced vs preserving model authority?

Concern: too much deterministic logic may reduce the “Cosmos intelligence” story.

Counterpoint: production robotics typically uses hierarchy:

- learned semantic policy at top,
- deterministic arbitration/safety/timing in middle,
- deterministic control at bottom.

---

## 7) Steelman of Competing Approaches

## Approach A — Model-Heavy Authority

Model chooses mode + action nearly every cycle; deterministic layer mostly clamps safety.

Pros:
- Strong AI narrative
- Maximum model autonomy

Cons:
- Highest demo risk (latency, formatting drift, collapse patterns)
- Hard to guarantee smooth repeatable show behavior

## Approach B — Balanced Hybrid (Hierarchical)

Model proposes intents/actions (and optionally mode hints); deterministic layer handles state transitions, dwell/cooldown/hysteresis/safety.

Pros:
- Strong reliability
- Still clearly model-driven if contracts/metrics are explicit
- Most realistic for one-week delivery

Cons:
- Requires clear authority contract to avoid becoming “mostly scripted”

## Approach C — Deterministic-Heavy

FSM drives most behavior; model is optional style/explanation enhancer.

Pros:
- Safest live demo
- Highly reproducible

Cons:
- Weak AI differentiation
- Risks looking hand-scripted to judges

---

## 8) Suggested One-Week Plan (Pragmatic)

Recommended target: **Approach B (Balanced Hybrid)**.

1. Freeze transport/parser churn unless severe failures recur.
2. Add explicit behavior mode machine with minimal states:
   - `IDLE_PRESENCE`
   - `ENGAGE_TRACK`
   - `ACKNOWLEDGE`
   - `RECOVER_RESET`
3. Keep model authority inside mode:
   - ranked proposals, style, confidence, urgency
4. Add deterministic diversity guards:
   - anti-repeat penalties
   - per-primitive max-duration/cooldown
5. Build competition evidence:
   - Cosmos on/off ablation
   - proposal->commit trace
   - reliability/latency charts

---

## 9) Specific Questions for Expert Review

1. For a one-week competition timeline, is Approach B the best risk/reward?
2. How to maximize perceived model authority while retaining deterministic safeguards?
3. Which 2–3 metrics best prove “model is meaningfully driving behavior”?
4. How much parser hardening is enough before stopping and focusing entirely on behavior/control polish?
5. Any recommended demo scenario design to best highlight embodied reasoning with current constraints?

---

## 10) Relevant Artifacts

- Provider probe: `logs/provider_probe/20260223_133022/report.json`
- API loop probe: `logs/cosmos_api_probe/20260223_140820/summary.json`
- Live runs:
  - `logs/runs/20260223_152100`
  - `logs/runs/20260223_153644`
  - `logs/runs/20260223_154554`
