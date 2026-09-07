from dataclasses import asdict
import json
import math

import pytest

from pala.config import load_config
from pala.control import TrajectoryExecutor
from pala.types import ActionPlan, MoveToCommand, PrimitiveKind
from tools import gesture_workshop as w


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class Servo:
    def __init__(self):
        self.writes = []
        self.enabled = True

    def set_angles(self, angles):
        self.writes.append(list(angles))

    def enable(self, value):
        self.enabled = value

    def shutdown(self):
        self.enabled = False


@pytest.fixture
def cfg():
    return load_config("config/robot.yaml")


@pytest.fixture
def session(cfg, tmp_path):
    clock = Clock()
    result = w.Session(cfg, tmp_path / "session", clock=clock, sleep=clock.sleep)
    yield result
    result.close()


def test_templates_execute_with_exact_targets_and_one_executor(session, cfg):
    executor = session.executor
    for recipe in w.templates(cfg.joint_names).values():
        trial = session.run(recipe)
        assert session.last_success
        assert session.executor is executor
        assert max(abs(x) for x in session.current) <= 1e-6
        ticks = [json.loads(x) for x in (session.directory / f"{trial}.jsonl").read_text().splitlines()]
        assert {x["step"] for x in ticks} == set(range(1, len(recipe.steps) + 1))
        assert not any(x["status"] in ("timed_out", "rejected") for x in ticks)
        for index, step in enumerate(recipe.steps, 1):
            move_ticks = [x for x in ticks if x["step"] == index and x["phase"] == "move"]
            target = w.resolve(cfg, recipe, [0] * 5)[index - 1]
            assert move_ticks[-1]["angles_rad"] == pytest.approx(target, abs=1e-6)
            holds = [x for x in ticks if x["step"] == index and x["phase"] == "hold"]
            if step.hold_s:
                assert holds[-1]["elapsed_s"] - holds[0]["elapsed_s"] >= step.hold_s - 0.03


def test_invalid_later_step_prevents_any_backend_construction(session, cfg, monkeypatch):
    monkeypatch.setattr(w, "DummyServo", lambda **kw: pytest.fail("constructed before preflight"))
    recipe = w.templates(cfg.joint_names)["greeting"]
    recipe.steps[-1].targets_deg["pitch2"] = 80
    with pytest.raises(ValueError, match="limits"):
        session.run(recipe)
    assert session.servo is None


def test_mapping_clipping_detected_even_if_yaml_limits_permit(cfg):
    cfg.joint_limits_rad[2] = [-1.57, 1.57]
    recipe = w.templates(cfg.joint_names)["greeting"]
    recipe.steps[2].targets_deg["pitch2"] = 80
    with pytest.raises(ValueError, match="mapping would clip"):
        w.resolve(cfg, recipe, [0] * 5)


def test_confirmed_pitch2_endpoints_accept_yaml_rounding(cfg):
    recipe = w.templates(cfg.joint_names)["greeting"]
    for angle in [-25, 65]:
        recipe.steps[2].targets_deg["pitch2"] = angle
        w.resolve(cfg, recipe, [0] * 5)


@pytest.mark.parametrize("assignment", ["rate=0", "rate=nan", "hold=-1", "yaw=inf", "unknown=2"])
def test_bad_edit_does_not_mutate_original(cfg, assignment):
    recipe = w.templates(cfg.joint_names)["greeting"]
    before = asdict(recipe)
    with pytest.raises(ValueError):
        edited = w.edit_recipe(recipe, 2, [assignment])
        w.resolve(cfg, edited, [0] * 5)
    assert asdict(recipe) == before


def test_unspecified_joints_are_preserved(cfg):
    recipe = w.templates(cfg.joint_names)["attention"]
    poses = w.resolve(cfg, recipe, [0] * 5)
    assert poses[2][0] == poses[1][0] == pytest.approx(math.radians(10))
    assert poses[2][3] == pytest.approx(math.radians(4))


def test_abort_disables_and_does_not_return_to_zero(session, cfg):
    servo = Servo()
    session.servo = servo
    def fail_after_motion(seconds):
        session.clock.sleep(seconds)
        if len(servo.writes) > 200:
            raise KeyboardInterrupt
    session.sleep = fail_after_motion
    with pytest.raises(KeyboardInterrupt):
        session.run(w.templates(cfg.joint_names)["attention"])
    assert not servo.enabled
    assert session.stopped
    assert not session.last_success
    assert any(abs(x) > 0 for x in session.current)
    assert servo.writes[-1] == session.current
    with pytest.raises(RuntimeError, match="restart"):
        session.connect()


def test_timeout_disables_and_records_failure(session, cfg):
    servo = Servo()
    session.servo = servo
    session.sleep = lambda _: session.clock.sleep(20)
    with pytest.raises(RuntimeError, match="budget"):
        session.run(w.templates(cfg.joint_names)["attention"])
    assert not servo.enabled
    events = [json.loads(x) for x in (session.directory / "trials.jsonl").read_text().splitlines()]
    assert events[-1]["status"] == "aborted"


def test_dummy_never_constructs_hardware_even_with_full_config(session, cfg, monkeypatch):
    cfg.mode = "jetson_full"
    monkeypatch.setattr(w, "_build_servo", lambda cfg: pytest.fail("hardware constructed"))
    session.run(w.templates(cfg.joint_names)["settling"])
    assert isinstance(session.servo, w.DummyServo)


def test_favorites_roundtrip_and_edited_candidate_not_marked_tested(session, cfg, tmp_path):
    recipe = w.templates(cfg.joint_names)["greeting"]
    session.run(recipe)
    path = tmp_path / "favorites.json"
    w.save_favorite(path, "winner", recipe, session)
    item = json.loads(path.read_text())["favorites"]["winner"]
    assert w.decode_recipe(item["recipe"]) == recipe
    assert item["trial"] == session.last_trial
    assert not item["hardware_trial"]
    recipe.steps[2].targets_deg["pitch3"] = 9
    w.save_favorite(path, "changed", recipe, session)
    assert json.loads(path.read_text())["favorites"]["changed"]["trial"] is None


def test_hardware_not_constructed_when_start_confirmation_declined(tmp_path, monkeypatch):
    responses = iter(["run", "NO", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    monkeypatch.setattr(w, "_build_servo", lambda cfg: pytest.fail("hardware constructed"))
    assert w.main(["--hardware", "--enable", "--session-dir", str(tmp_path / "session")]) == 0


@pytest.mark.parametrize("flag", ["--hardware", "--enable"])
def test_hardware_requires_both_flags(flag):
    with pytest.raises(SystemExit) as exc:
        w.main([flag])
    assert exc.value.code == 2


def test_workshop_small_move_reaches_command_target_without_changing_default():
    clock = Clock()
    executor = TrajectoryExecutor([[-1, 1]], clock=clock, position_tolerance_rad=1e-6)
    target = math.radians(2)
    action = ActionPlan(PrimitiveKind.MOVE_TO, MoveToCommand([target], rate_rad_s=0.1), 1)
    for _ in range(100):
        command = executor.step(action, 0.01)
        clock.sleep(0.01)
        if executor.control_state.status == w.ExecutionStatus.DONE:
            break
    assert command.joint_angles_rad[0] == pytest.approx(target, abs=1e-6)
    default = TrajectoryExecutor([[-1, 1]], clock=clock)
    assert default._position_tolerance_rad == 0.02


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_executor_rejects_invalid_tolerance(value):
    with pytest.raises(ValueError):
        TrajectoryExecutor([[-1, 1]], position_tolerance_rad=value)


def test_interactive_edit_run_rate_save_load_in_dummy(tmp_path, monkeypatch):
    responses = iter([
        "edit 1 hold=0", "edit 3 pitch3=9 hold=0", "edit 4 hold=0", "run", "",
        "rate 4 Clear gesture", "save candidate", "edit 3 pitch3=6", "load candidate", "show", "q",
    ])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    original_session = w.Session
    def fast_session(*args, **kwargs):
        clock = Clock()
        return original_session(*args, **kwargs, clock=clock, sleep=clock.sleep)
    monkeypatch.setattr(w, "Session", fast_session)
    monkeypatch.setattr(w, "_build_servo", lambda cfg: pytest.fail("hardware constructed"))
    directory = tmp_path / "session"
    favorite_path = tmp_path / "favorites.json"
    assert w.main(["--session-dir", str(directory), "--favorites", str(favorite_path)]) == 0
    events = [json.loads(x) for x in (directory / "trials.jsonl").read_text().splitlines()]
    assert events[-1]["type"] == "rating"
    assert events[-1]["notes"] == "Clear gesture"
    candidate = json.loads(favorite_path.read_text())["favorites"]["candidate"]
    assert candidate["recipe"]["steps"][2]["targets_deg"]["pitch3"] == 9
    assert candidate["trial"] == events[-1]["trial"]


def test_trial_log_failure_disables_previously_connected_hardware(session, cfg, monkeypatch):
    servo = Servo()
    session.servo = servo
    original_event = session.event
    def fail_start(kind, **data):
        if kind == "trial_start":
            raise OSError("disk full")
        original_event(kind, **data)
    monkeypatch.setattr(session, "event", fail_start)
    with pytest.raises(OSError, match="disk full"):
        session.run(w.templates(cfg.joint_names)["greeting"])
    assert not servo.enabled
    assert servo.writes == []
