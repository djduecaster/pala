"""Operator-driven, logged pose choreography using the production executor.

Run from the repository root: uv run python -m tools.gesture_workshop
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import shlex
import signal
import time
from typing import Callable
from uuid import uuid4

from pala.config import load_config
from pala.control import TrajectoryExecutor
from pala.control.executor import ExecutionStatus
from pala.hardware import DummyServo
from pala.main import _build_servo
from pala.types import ActionPlan, HoldCommand, MoveToCommand, PrimitiveKind

LOG = logging.getLogger("gesture_workshop")


@dataclass
class Step:
    name: str
    targets_deg: dict[str, float] = field(default_factory=dict)
    rate_deg_s: float = 12.0
    hold_s: float = 0.0


@dataclass
class Recipe:
    name: str
    steps: list[Step]


def templates(joints: list[str]) -> dict[str, Recipe]:
    zero = dict.fromkeys(joints, 0.0)
    return {
        "greeting": Recipe("greeting", [
            Step("Prepare neutral", zero.copy(), hold_s=2),
            Step("Anticipate", {"pitch3": -2}),
            Step("Acknowledge", {"pitch3": 6}, hold_s=0.3),
            Step("Recover", zero.copy(), rate_deg_s=8, hold_s=2),
        ]),
        "attention": Recipe("attention", [
            Step("Prepare neutral", zero.copy(), hold_s=2),
            Step("Turn", {"yaw": 10}),
            Step("Tilt and attend", {"roll": 4}, hold_s=2),
            Step("Recover", zero.copy(), rate_deg_s=8, hold_s=2),
        ]),
        "settling": Recipe("settling", [
            Step("Prepare neutral", zero.copy(), hold_s=2),
            Step("Prepare attentive pose", {"yaw": 10, "roll": 4}, hold_s=2),
            Step("Settle", zero.copy(), rate_deg_s=8, hold_s=3),
        ]),
    }


def number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}: expected a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label}: expected a finite {'positive ' if positive else ''}number")
    return result


def check_pose(cfg, pose: list[float]) -> None:
    if len(pose) != len(cfg.joint_names):
        raise ValueError("Pose must contain one value per configured joint")
    for name, angle, limits in zip(cfg.joint_names, pose, cfg.joint_limits_rad):
        value = number(angle, name)
        if not limits[0] - 1e-6 <= value <= limits[1] + 1e-6:
            raise ValueError(f"{name}: {math.degrees(value):.3f} degrees exceeds YAML limits")
        jc = cfg.servo_calibration.get("per_joint", {}).get(name)
        if jc is None:
            raise ValueError(f"Missing servo calibration for {name}; cannot check mapping")
        mapped = math.degrees(value) * number(jc["angle_scale"], name) + number(jc["angle_offset"], name)
        if not -1e-4 <= mapped <= 180 + 1e-4:
            raise ValueError(f"{name}: servo mapping would clip ({mapped:.3f} degrees)")


def resolve(cfg, recipe: Recipe, start: list[float]) -> list[list[float]]:
    """Preflight every endpoint; linear interpolation stays within these bounds."""
    if not recipe.name or not 1 <= len(recipe.steps) <= 64:
        raise ValueError("A named recipe must contain 1–64 steps")
    first = recipe.steps[0].targets_deg
    if set(first) != set(cfg.joint_names) or any(number(v, "start") != 0 for v in first.values()):
        raise ValueError("First step must explicitly prepare all joints at zero")
    current = list(start)
    check_pose(cfg, current)
    poses = []
    for step in recipe.steps:
        if not step.name:
            raise ValueError("Every step needs a name")
        number(step.rate_deg_s, "rate", positive=True)
        if number(step.hold_s, "hold") < 0:
            raise ValueError("Hold duration must be nonnegative")
        for name, value in step.targets_deg.items():
            if name not in cfg.joint_names:
                raise ValueError(f"Unknown joint: {name}")
            current[cfg.joint_names.index(name)] = math.radians(number(value, name))
        check_pose(cfg, current)
        # Accommodate sub-micro-radian rounding of degree endpoints in YAML.
        current = [max(lo, min(hi, v)) for v, (lo, hi) in zip(current, cfg.joint_limits_rad)]
        poses.append(current.copy())
    return poses


def decode_recipe(raw: dict) -> Recipe:
    return Recipe(name=raw["name"], steps=[Step(**item) for item in raw["steps"]])


def edit_recipe(recipe: Recipe, index: int, assignments: list[str]) -> Recipe:
    updated = deepcopy(recipe)
    if not 1 <= index <= len(updated.steps):
        raise ValueError("Step number is out of range")
    step = updated.steps[index - 1]
    for item in assignments:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError("Use joint=degrees, rate=degrees/s, hold=seconds, or name=text")
        if key == "name":
            step.name = value
        elif key == "rate":
            step.rate_deg_s = number(value, "rate", positive=True)
        elif key == "hold":
            step.hold_s = number(value, "hold")
        elif value == "keep":
            step.targets_deg.pop(key, None)
        else:
            step.targets_deg[key] = number(value, key)
    return updated


def show(cfg, recipe: Recipe, current: list[float], previous: Recipe | None = None) -> None:
    poses = resolve(cfg, recipe, current)
    LOG.info("\n%s | absolute joint degrees | %s", recipe.name.upper(), ", ".join(cfg.joint_names))
    for i, (step, pose) in enumerate(zip(recipe.steps, poses), 1):
        targets = " ".join(f"{name}={math.degrees(v):+.2f}" for name, v in zip(cfg.joint_names, pose))
        LOG.info("%d. %-25s %s | rate <= %.2f deg/s | hold %.2f s", i, step.name, targets, step.rate_deg_s, step.hold_s)
    LOG.info("Rate is a per-joint slew limit, not acceleration easing or measured arrival time.")
    LOG.info("Unspecified joints retain their previous targets. After completion: hold final pose until next command or quit.")
    if previous is not None and asdict(previous) != asdict(recipe):
        LOG.info("Changes since previous trial:")
        for i, (old, new) in enumerate(zip(previous.steps, recipe.steps), 1):
            if old != new:
                LOG.info("  Step %d: %s -> %s", i, asdict(old), asdict(new))
        if len(previous.steps) != len(recipe.steps) or previous.name != recipe.name:
            LOG.info("  Recipe changed from %s (%d steps)", previous.name, len(previous.steps))


class Session:
    def __init__(self, cfg, directory: Path, *, hardware: bool = False,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.directory = directory
        self.hardware = hardware
        self.clock, self.sleep = clock, sleep
        self.servo = None
        self.stopped = False
        self.current = [0.0] * len(cfg.joint_names)
        # Explicit degrees/s mean exactly that; disable implicit style multipliers.
        unit_style = {key: 1.0 for key in ("amp_scale", "rate_scale", "duration_scale", "settle_scale")}
        self.executor = TrajectoryExecutor(cfg.joint_limits_rad, style_profiles={"calm": unit_style},
                                           clock=clock, position_tolerance_rad=1e-6)
        directory.mkdir(parents=True, exist_ok=False)
        self.events = (directory / "trials.jsonl").open("a", encoding="utf-8")
        self.last_recipe: Recipe | None = None
        self.last_trial: str | None = None
        self.last_success = False
        self.event("session", hardware=hardware, joint_names=cfg.joint_names)

    def event(self, kind: str, **data) -> None:
        self.events.write(json.dumps({"type": kind, "time": datetime.now(timezone.utc).isoformat(), **data}, allow_nan=False) + "\n")
        self.events.flush()

    def connect(self) -> None:
        if self.stopped:
            raise RuntimeError("Outputs were disabled; restart and confirm the starting posture")
        if self.servo is None:
            cfg = replace(self.cfg, mode="jetson_full")
            self.servo = _build_servo(cfg) if self.hardware else DummyServo(log_every=1000000)

    def run(self, recipe: Recipe) -> str:
        poses = resolve(self.cfg, recipe, self.current)  # Before hardware construction/writes.
        if self.stopped:
            raise RuntimeError("Outputs were disabled; restart and confirm the starting posture")
        trial_id = uuid4().hex[:12]
        self.last_trial, self.last_recipe, self.last_success = trial_id, deepcopy(recipe), False
        start = self.clock()
        try:
            self.event("trial_start", trial=trial_id, recipe=asdict(recipe), start_rad=self.current.copy())
            with (self.directory / f"{trial_id}.jsonl").open("w", encoding="utf-8") as trace:
                self.connect()
                for i, (step, target) in enumerate(zip(recipe.steps, poses), 1):
                    LOG.info("[%s] STEP %d/%d: %s", trial_id, i, len(poses), step.name)
                    distance = max(abs(t - c) for t, c in zip(target, self.current))
                    budget = distance / math.radians(step.rate_deg_s) + 2.0
                    action = ActionPlan(PrimitiveKind.MOVE_TO,
                                        MoveToCommand(target, rate_rad_s=math.radians(step.rate_deg_s), timeout_s=budget),
                                        1.0, explanation=step.name)
                    self._phase(action, budget + 0.1, trace, trial_id, i, "move", start)
                    if step.hold_s:
                        hold = ActionPlan(PrimitiveKind.HOLD, HoldCommand(), 1.0, explanation=step.name)
                        self._phase(hold, step.hold_s, trace, trial_id, i, "hold", start)
            self.last_success = True
            self.event("trial_end", trial=trial_id, status="complete", elapsed_s=self.clock() - start)
            LOG.info("[%s] COMPLETE. Holding final commanded pose; enter a rating or another command.", trial_id)
            return trial_id
        except BaseException as exc:
            # Disable before logging failures; never automatically recover after an abort.
            self.stop()
            self.event("trial_end", trial=trial_id, status="aborted", reason=type(exc).__name__ + ": " + str(exc))
            raise

    def _phase(self, action, budget, trace, trial_id, index, phase, trial_start) -> None:
        start = last = self.clock()
        next_report = start
        period = 1.0 / self.cfg.loop_rates.control_hz
        while True:
            now = self.clock()
            if now - start >= budget:
                if phase == "hold":
                    return
                raise RuntimeError(f"Step {index} exceeded its movement budget")
            # Avoid a large trajectory jump if the process stalls.
            dt = min(now - last, period * 2)
            last = now
            command = self.executor.step(action, dt)
            self.servo.set_angles(command.joint_angles_rad)
            self.current = list(command.joint_angles_rad)
            status = self.executor.control_state.status
            trace.write(json.dumps({"trial": trial_id, "step": index, "phase": phase,
                                    "elapsed_s": now - trial_start, "status": status.value,
                                    "angles_rad": self.current, "commanded_enable": command.enable}) + "\n")
            trace.flush()
            if now >= next_report:
                LOG.info("  %s elapsed %.1f s | %s", phase, now - start, status.value)
                next_report = now + 1
            if status in (ExecutionStatus.TIMED_OUT, ExecutionStatus.REJECTED, ExecutionStatus.CANCELED):
                raise RuntimeError(f"Step {index}: {status.value}: {self.executor.control_state.reason}")
            if phase == "move" and status == ExecutionStatus.DONE:
                return
            self.sleep(period)

    def stop(self) -> None:
        self.stopped = True
        if self.servo is not None:
            self.servo.enable(False)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            try:
                if self.servo is not None:
                    self.servo.shutdown()
            finally:
                self.events.close()


def save_favorite(path: Path, name: str, recipe: Recipe, session: Session) -> None:
    data = json.loads(path.read_text()) if path.exists() else {"version": 1, "favorites": {}}
    if data.get("version") != 1:
        raise ValueError("Unsupported favorites version")
    tested = session.last_success and session.last_recipe == recipe
    data["favorites"][name] = {"recipe": asdict(recipe), "trial": session.last_trial if tested else None,
                               "session": str(session.directory) if tested else None,
                               "hardware_trial": bool(tested and session.hardware)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


HELP = """GESTURE WORKSHOP HELP

Inspect and choose a gesture:
  show                         Show numbered steps, resolved targets, and changes
                               since the previous trial.
  use greeting                 Select the greeting template.
  use attention                Select the turn-and-tilt template.
  use settling                 Select attention preparation and slow recovery.

Edit a numbered step (use show to find its number):
  edit 3 pitch3=9               Set pitch3's absolute joint target to +9 degrees.
  edit 3 yaw=-10 roll=5         Change multiple targets in the same step.
  edit 3 rate=12 hold=0.3       Set speed limit to 12 degrees/s and pause to 0.3 s.
  edit 3 name="Acknowledge"     Change the step label.
  edit 3 yaw=keep               Remove this step's yaw override; keep the preceding
                               target instead. Other step settings are unchanged.

Units and sequence rules:
  Targets are absolute PALA joint degrees, NOT increments or raw servo degrees.
  Joint names come from robot.yaml (normally yaw, pitch1, pitch2, roll, pitch3).
  rate= is a per-joint slew limit in degrees/s, not an exact movement duration.
  hold= is a pause in seconds AFTER commanded movement completion.
  Unspecified joints retain their previous targets. The first step must prepare
  all joints at zero. Invalid limits/mapping values are rejected before a run.

Run and review:
  run / r                      Show the sequence and ask before running it.
                               At the run prompt: Enter runs; other text cancels.
  rate 4 Clear nod              Save a 1–5 rating and optional notes for the MOST
                               RECENT trial. This does not change movement speed.
  neutral                      Separately confirm a return to zero at 8 degrees/s,
                               then hold 2 s. This is recorded as its own trial.

Save and compare:
  save greeting-a              Save the current candidate under this name.
  favorites                    List saved names.
  load greeting-a              Load a saved candidate; use run to replay it.
  use and load replace current edits: save anything you want to keep first.
  Saving an existing name replaces it. Unrun/edited candidates can be saved,
  but are not labeled as successfully tested. Ratings refer to the last trial;
  save refers to the current candidate (even after a separate neutral trial).

Example comparison (enter commands individually and answer each run prompt):
  use greeting
  run
  rate 3 Too subtle
  save greeting-a
  edit 3 pitch3=9
  run
  rate 4 Clear acknowledgment
  save greeting-b
  load greeting-a
  run

Holding and stopping:
  Completed trials keep the final pose commanded while you review or edit.
  stop / q / quit              Disable outputs and EXIT; no automatic return.
  Ctrl-C                       During motion or at a prompt: disable and exit.
  Do not type stop during motion; the text prompt is only read between trials.
  Outputs may release/sag when disabled. After an abort, restart from a known pose.
  This standalone tool has no main-runtime deadman; do not leave it unattended.

Files and help:
  Trial logs and a YAML snapshot: the session directory printed at startup.
  Saved candidates: config/gesture_favorites.json by default (--favorites overrides).
  Copy Jetson-only favorites back before deployment; deployment can delete them.
  help / ?                     Show this guide; does not move the lamp.
  For launch options: uv run python -m tools.gesture_workshop --help
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/robot.yaml")
    parser.add_argument("--gesture", choices=["greeting", "attention", "settling"], default="greeting")
    parser.add_argument("--hardware", action="store_true", help="Use Jetson servos; requires --enable and posture confirmation")
    parser.add_argument("--enable", action="store_true", help="Authorize hardware output with --hardware")
    parser.add_argument("--session-dir", type=Path, help="New directory; existing directories are never overwritten")
    parser.add_argument("--favorites", type=Path, default=Path("config/gesture_favorites.json"))
    args = parser.parse_args(argv)
    if args.hardware != args.enable:
        parser.error("Hardware requires both --hardware and --enable; omit both for dummy mode")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    library = templates(cfg.joint_names)
    recipe = deepcopy(library[args.gesture])
    resolve(cfg, recipe, [0.0] * len(cfg.joint_names))
    directory = args.session_dir or Path("logs/workshop") / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6])
    session = Session(cfg, directory, hardware=args.hardware)
    confirmed = not args.hardware
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        (directory / "robot.yaml").write_text(Path(args.config).read_text())
        LOG.info("%s mode. Session: %s", "HARDWARE" if args.hardware else "DUMMY", directory)
        LOG.info("Poses are commanded estimates. Completed trials hold PWM while you review; quit/stop disables it.")
        LOG.info("No camera or main-runtime deadman is started. Stop other servo controllers before hardware trials.")
        LOG.info(HELP)
        show(cfg, recipe, session.current)
        while True:
            try:
                parts = shlex.split(input("workshop> "))
                if not parts:
                    continue
                cmd, *rest = parts
                if cmd in ("q", "quit", "stop"):
                    break
                if cmd in ("help", "?"):
                    LOG.info(HELP)
                elif cmd == "show":
                    show(cfg, recipe, session.current, session.last_recipe)
                elif cmd == "edit":
                    candidate = edit_recipe(recipe, int(rest[0]), rest[1:])
                    resolve(cfg, candidate, session.current)
                    recipe = candidate
                    show(cfg, recipe, session.current, session.last_recipe)
                elif cmd == "use":
                    recipe = deepcopy(library[rest[0]])
                    show(cfg, recipe, session.current, session.last_recipe)
                elif cmd in ("run", "r", "neutral"):
                    candidate = recipe if cmd != "neutral" else Recipe("neutral", [Step("Return neutral", dict.fromkeys(cfg.joint_names, 0.0), 8, 2)])
                    show(cfg, candidate, session.current, session.last_recipe)
                    if not confirmed:
                        LOG.info("Before first motion: establish the known physical zero pose and clear the mechanism.")
                        if input("Type ZERO to confirm that starting posture: ").strip() != "ZERO":
                            continue
                        confirmed = True
                    if input("Press Enter to RUN this sequence; type anything to cancel: "):
                        continue
                    session.run(candidate)
                elif cmd == "rate":
                    if session.last_trial is None:
                        raise ValueError("Run a trial before rating it")
                    rating = int(rest[0])
                    if not 1 <= rating <= 5:
                        raise ValueError("Rating must be 1–5")
                    session.event("rating", trial=session.last_trial, rating=rating, notes=" ".join(rest[1:]))
                    LOG.info("Rating saved for %s", session.last_trial)
                elif cmd == "save":
                    save_favorite(args.favorites, rest[0], recipe, session)
                    LOG.info("Saved %s to %s (trial provenance retained only for unchanged completed candidates)", rest[0], args.favorites)
                elif cmd in ("load", "favorites"):
                    data = json.loads(args.favorites.read_text())
                    if data.get("version") != 1:
                        raise ValueError("Unsupported favorites version")
                    if cmd == "favorites":
                        LOG.info("Favorites: %s", ", ".join(data["favorites"]))
                    else:
                        candidate = decode_recipe(data["favorites"][rest[0]]["recipe"])
                        resolve(cfg, candidate, session.current)
                        recipe = candidate
                        show(cfg, recipe, session.current, session.last_recipe)
                else:
                    raise ValueError("Unknown command; type help")
            except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
                if session.stopped:
                    raise
                LOG.error("Not applied: %s", exc)
    except (KeyboardInterrupt, EOFError):
        LOG.info("Stopping; no automatic return movement.")
    except Exception:
        LOG.exception("Workshop failed; disabling outputs.")
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
