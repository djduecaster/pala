from tools.live_greeting import LiveGreeting
from test_attention_probe import Clock, snapshot, REST

STATE = dict(REST, plan_id='startup-1', greeted=False)


def setup(tmp_path):
    clock=Clock()
    probe=LiveGreeting(tmp_path/'probe', 'mock', None, mock=True, clock=clock)
    return clock,probe


def answer(probe, clock, **updates):
    result=dict(trial_id='t', frame_mono_ns=int((clock.now-3)*1e9), generation=probe.generation,
                context_plan_id=STATE['plan_id'], status='valid', decision={'intent':'acknowledge'})
    result.update(updates)
    return result


def test_one_greeting_per_arm(tmp_path):
    clock,p=setup(tmp_path)
    p.command('arm',STATE)
    p.fresh_view=True
    r=answer(p,clock)
    p.on_result(r,STATE)
    assert p.take_request()=='greet'
    assert not p.armed
    p.on_result(r,STATE)
    assert p.take_request() is None


def test_pause_rearm_rejects_old_result(tmp_path):
    clock,p=setup(tmp_path)
    p.command('arm',STATE); old=answer(p,clock)
    p.command('pause',STATE);p.command('arm',STATE)
    p.fresh_view=True;p.on_result(old,STATE)
    assert p.take_request() is None


def test_stale_unknown_moving_and_camera_failure_do_not_move(tmp_path):
    clock,p=setup(tmp_path);p.command('arm',STATE);p.fresh_view=True
    for updates in [dict(frame_mono_ns=int((clock.now-7)*1e9)),dict(status='invalid'),dict(context_plan_id='old'),dict(decision={'intent':'wait'})]:
        p.on_result(answer(p,clock,**updates),STATE)
        assert p.take_request() is None
    p.on_result(answer(p,clock),dict(STATE,busy=True))
    assert p.take_request() is None
    p.fresh_view=False;p.on_result(answer(p,clock),STATE)
    assert p.take_request() is None


def test_arm_delay_single_flight_and_no_capture_after_greet(tmp_path):
    clock,p=setup(tmp_path);p.command('arm',STATE)
    clock.now+=4;p.tick(snapshot(clock),STATE)
    assert p.worker is None
    clock.now+=1.1;p.tick(snapshot(clock),STATE)
    p.worker.join(2)
    assert p.inflight
    p.tick(snapshot(clock),STATE)
    assert p.inflight is None  # mock wait result, no actuation
    assert p.take_request() is None
    p.tick(snapshot(clock),dict(STATE,busy=True))
    assert not p.armed and p.trial is None


def test_reset_and_timeout(tmp_path):
    clock,p=setup(tmp_path);p.command('arm',STATE)
    clock.now+=66;p.tick(snapshot(clock),STATE)
    assert not p.armed and p.worker is None
    p.command('reset',dict(STATE,state='attention',greeted=True))
    assert p.take_request()=='settle'
    assert not p.armed


def test_runtime_dispatches_one_greeting_and_reset(tmp_path):
    import json, os, subprocess, sys, time
    from pathlib import Path
    raw=json.loads(Path('config/performances.json').read_text())
    for pose in raw['poses_deg'].values():
        pose[:]=[v/100 for v in pose]
    for performance in raw['performances'].values():
        for step in performance['steps']:
            if 'target_deg' in step:
                step['target_deg']=[v/100 for v in step['target_deg']]
            step['hold_s']=.025
    library=tmp_path/'performances.json';library.write_text(json.dumps(raw))
    code='''
import sys
from tools.live_greeting import LiveGreeting
from pala.main import main
def respond(self, jpeg, metadata):
    self.results.put(dict(metadata, status='valid', decision={'intent':'acknowledge'}, latency_ms=0))
LiveGreeting._request=respond
raise SystemExit(main(['--manual','--live-greeting','--mode','dev','--probe-mock',*sys.argv[1:]]))
'''
    env=dict(os.environ,PALA_RUN_LOG_ROOT=str(tmp_path/'runtime'))
    env.pop('PALA_MAX_RUNTIME_S',None)
    output=tmp_path/'console.txt'
    with output.open('w') as stream:
        p=subprocess.Popen([sys.executable,'-c',code,'--performances',str(library),'--probe-output',str(tmp_path/'probe')],stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT,text=True,env=env)
        def wait(text):
            deadline=time.monotonic()+12
            while time.monotonic()<deadline:
                if text in output.read_text():return
                assert p.poll() is None,output.read_text()
                time.sleep(.025)
            raise AssertionError(output.read_text())
        def send(text):
            p.stdin.write(text+'\n');p.stdin.flush()
        try:
            wait('LIVE TEST READY at rest, disarmed')
            send('arm')
            wait('live request=greet accepted=True')
            wait('performance=greet step=5 Remain attentive phase=complete')
            time.sleep(.4)
            send('arm');wait('Cannot arm:')
            send('reset');wait('live request=settle accepted=True')
            wait('performance=settle step=3 Settle both pitch joints phase=complete')
            send('shutdown')
            assert p.wait(timeout=8)==0,output.read_text()
        finally:
            if p.poll() is None:p.kill();p.wait()
            p.stdin.close()
    assert output.read_text().count('live request=greet accepted=True')==1
