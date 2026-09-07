"""Exercise actual four-loop subprocesses using dummy camera and servo backends."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize('exit_mode', ['shutdown', 'interrupt', 'eof'])
def test_manual_runtime(tmp_path, exit_mode):
    raw = json.loads(Path('config/performances.json').read_text())
    # Small positions and short holds retain distinct poses while keeping tests fast.
    for pose in raw['poses_deg'].values():
        pose[:] = [v / 100 for v in pose]
    for performance in raw['performances'].values():
        for step in performance['steps']:
            if 'target_deg' in step:
                step['target_deg'] = [v / 100 for v in step['target_deg']]
            step['hold_s'] = .025
    library = tmp_path / 'performances.json'
    library.write_text(json.dumps(raw))
    env = dict(os.environ, PALA_RUN_LOG_ROOT=str(tmp_path), PALA_RUN_ID='run', PALA_RUN_SCOPED_LOGS='1')
    env.pop('PALA_MAX_RUNTIME_S', None)
    output_path = tmp_path / 'console.txt'
    with output_path.open('w') as output:
        process = subprocess.Popen([sys.executable, '-m', 'pala.main', '--manual', '--mode', 'dev', '--performances', str(library)], stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, env=env, text=True)
        def wait_for(text):
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if text in output_path.read_text():
                    return
                assert process.poll() is None, output_path.read_text()
                time.sleep(.025)
            pytest.fail(output_path.read_text())
        def send(command):
            process.stdin.write(command + '\n')
            process.stdin.flush()
        try:
            wait_for('performance=startup step=2 Enter rest phase=complete')
            time.sleep(.4)  # behavior observes control completion
            send('greet\ngreet')
            wait_for('request=greet accepted=False')
            wait_for('performance=greet step=5 Remain attentive phase=complete')
            time.sleep(.4)
            if exit_mode == 'shutdown':
                send('settle')
                wait_for('performance=settle step=3 Settle both pitch joints phase=complete')
                send('shutdown')
            elif exit_mode == 'interrupt':
                process.send_signal(signal.SIGINT)
            else:
                process.stdin.close()
            assert process.wait(timeout=8) == 0, output_path.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if not process.stdin.closed:
                process.stdin.close()
    rows = [json.loads(line) for line in (tmp_path / 'run' / 'interaction.jsonl').read_text().splitlines()]
    executions = [row for row in rows if row['type'] == 'execution']
    if exit_mode == 'interrupt':
        assert not any(row['report']['name'] == 'shutdown' for row in executions)
        assert any(abs(v) > 0 for v in executions[-1]['command']['joint_angles_rad'])
    else:
        assert executions[-1]['report']['name'] == 'shutdown'
        assert executions[-1]['report']['status'] == 'complete'
        assert executions[-1]['command']['joint_angles_rad'] == pytest.approx([0] * 5)
    assert (tmp_path / 'run' / 'robot.yaml').exists()


def test_manual_hardware_requires_enable_before_initialization(monkeypatch):
    import pala.main as main
    monkeypatch.setattr(main, '_build_servo', lambda cfg: pytest.fail('hardware initialized'))
    with pytest.raises(ValueError, match='--enable'):
        main.main(['--manual', '--mode', 'jetson_full'])
    monkeypatch.setattr('builtins.input', lambda _: 'cancel')
    assert main.main(['--manual', '--mode', 'jetson_full', '--enable']) == 0


def test_manual_deadman_latches_stop_after_control_stall(monkeypatch):
    import pala.main as main
    from pala.config import load_config
    cfg = load_config('config/robot.yaml')
    cfg.logging.enabled = False
    cfg.deadman_timeout_ms = 100
    monkeypatch.setattr(main, 'load_config', lambda _: cfg)
    monkeypatch.setenv('PALA_MAX_RUNTIME_S', '2')
    class Console:
        def __init__(self, stream):
            pass
        def poll(self):
            return []
    class Servo:
        calls = []
        def set_angles(self, angles):
            self.calls.append(('angles', tuple(angles)))
        def enable(self, enabled):
            self.calls.append(('enable', enabled))
        def shutdown(self):
            self.calls.append(('shutdown', None))
    servo = Servo()
    monkeypatch.setattr(main, 'ManualConsole', Console)
    monkeypatch.setattr(main, '_build_servo', lambda _: servo)
    original = main.PerformanceSequencer.tick
    count = 0
    def stalled(self, plan, dt):
        nonlocal count
        count += 1
        if count == 5:
            time.sleep(.3)
        return original(self, plan, dt)
    monkeypatch.setattr(main.PerformanceSequencer, 'tick', stalled)
    assert main.main(['--manual', '--mode', 'dev']) == 1
    assert any(kind == 'angles' for kind, _ in servo.calls)
    last_angles = max(i for i, (kind, _) in enumerate(servo.calls) if kind == 'angles')
    assert ('enable', False) in servo.calls[last_angles + 1:]
    assert ('enable', True) not in servo.calls[last_angles + 1:]
