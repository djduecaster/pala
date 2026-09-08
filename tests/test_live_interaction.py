import json
from dataclasses import replace
from pathlib import Path

import pytest

from pala.behavior.social_observation import validate_response, make_request
from pala.behavior.model_clients import ModelResponse
from pala.behavior.manual import ManualBehaviorPolicy
from pala.control.performances import PerformanceLibrary, PerformanceSequencer
from pala.config import load_config
from tools.live_interaction import LiveInteraction
from test_attention_probe import Clock, snapshot, REST
from test_performances import finish


def setup(tmp_path):
    c=Clock();p=LiveInteraction(tmp_path/'probe','mock',None,mock=True,clock=c)
    state=dict(REST, plan_id='startup', greeted=False)
    p.command('arm',state);p.fresh_view=True
    return c,p,state


def answer(c,p,s, person='present', attention='toward_camera', gesture='none', **extra):
    return dict(trial_id='t',frame_mono_ns=int((c.now-2)*1e9),generation=p.generation,
                context_plan_id=s['plan_id'],stage=p.stage,status='valid',latency_ms=2000,
                decision=dict(person=person,attention=attention,gesture=gesture,intent='acknowledge' if attention=='toward_camera' else 'wait'),**extra)


def complete(c,p,s,request):
    target={'notice':'noticed','greet':'attention','excite':'excited','settle':'rest'}[request]
    s=dict(s,state=target,plan_id=request,performance=request,greeted=target in ('attention','excited'))
    c.now+=3;p.tick(snapshot(c),s)
    return s


def test_one_full_social_exchange_and_two_away_observations(tmp_path):
    c,p,s=setup(tmp_path)
    for request in ['notice','greet','excite']:
        p.on_result(answer(c,p,s,gesture='thumbs_up' if request=='excite' else 'none'),s)
        assert p.take_request()==request
        # Duplicate replies cannot launch another gesture while this one is pending.
        p.on_result(answer(c,p,s,gesture='thumbs_up'),s)
        assert p.take_request() is None
        s=complete(c,p,s,request)
    p.on_result(answer(c,p,s,gesture='thumbs_up'),s)
    assert p.take_request() is None
    p.on_result(answer(c,p,s,attention='elsewhere'),s)
    assert p.take_request() is None
    c.now+=4.1;p.on_result(answer(c,p,s,attention='elsewhere'),s)
    assert p.take_request()=='settle'
    s=complete(c,p,s,'settle')
    assert not p.armed
    p.close()


@pytest.mark.parametrize('condition',['stale','context','generation','camera','busy','pause','invalid'])
def test_invalid_or_obsolete_observations_never_move(tmp_path,condition):
    c,p,s=setup(tmp_path);r=answer(c,p,s,gesture='thumbs_up')
    if condition=='stale':r['frame_mono_ns']=int((c.now-7)*1e9)
    if condition=='context':r['context_plan_id']='other'
    if condition=='generation':r['generation']-=1
    if condition=='camera':p.fresh_view=False
    if condition=='busy':s=dict(s,busy=True)
    if condition=='pause':p.command('pause',s)
    if condition=='invalid':r['status']='invalid'
    p.on_result(r,s);assert p.take_request() is None
    assert p.early_thumb_trial is None
    p.close()


def test_mock_worker_and_timeout(tmp_path):
    c,p,s=setup(tmp_path);c.now+=5.1;p.tick(snapshot(c),s)
    p.worker.join(2);p.tick(snapshot(c),s)
    assert p.inflight is None and p.take_request() is None
    rows=[json.loads(x) for x in (p.output/'trials.jsonl').read_text().splitlines()]
    assert any(x.get('status')=='valid' for x in rows)
    c.now+=181;p.tick(snapshot(c),s);assert not p.armed
    p.close()


def test_rejected_motion_disarms(tmp_path):
    c,p,s=setup(tmp_path);p.on_result(answer(c,p,s),s)
    p.motion_result(p.take_request(),False)
    assert not p.armed and p.expected_motion is None


def test_contract_and_image_request():
    obj=dict(person='present',attention='toward_camera',gesture='thumbs_up',intent='acknowledge',evidence='A raised thumb and face turned toward the camera.')
    def response(o):return ModelResponse(True,200,0,{'choices':[{'message':{'content':json.dumps(o)}}]},None)
    assert validate_response(response(obj))==obj
    with pytest.raises(ValueError):validate_response(response(dict(obj,person='absent',attention='not_applicable',intent='wait')))
    with pytest.raises(ValueError):validate_response(response(dict(obj,motor_target=40)))
    assert make_request(b'jpg','mock').messages[1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;')


def test_desk_library_matches_accepted_rates_and_runs_all_paths():
    cfg=load_config('config/robot.yaml');lib=PerformanceLibrary('config/desk_performances.json',cfg)
    c=Clock();seq=PerformanceSequencer(cfg,clock=c);policy=ManualBehaviorPolicy(lib)
    report,_=finish(seq,policy.step(None),c,cfg);policy.step(report)
    for request in ['notice','greet','excite','settle','notice','settle','greet','settle','shutdown']:
        accepted,message=policy.request(request);assert accepted,message
        report,_=finish(seq,policy.step(None),c,cfg);policy.step(report)
    assert policy.status()['finished']
    assert max(s.rate_deg_s for s in lib.plan('excite').steps)==pytest.approx(30.8)
    assert lib.raw['provenance']['excite']['trial']=='996eff8ac06a'


def test_runtime_dispatches_complete_social_sequence(tmp_path):
    import os, subprocess, sys, time
    raw=json.loads(Path('config/desk_performances.json').read_text())
    for pose in raw['poses_deg'].values():pose[:]=[v/100 for v in pose]
    for perf in raw['performances'].values():
        for step in perf['steps']:
            if 'target_deg' in step:step['target_deg']=[v/100 for v in step['target_deg']]
            step['hold_s']=.025
    library=tmp_path/'library.json';library.write_text(json.dumps(raw))
    code='''
import sys
from tools.live_interaction import LiveInteraction
from pala.main import main
def respond(self,jpeg,metadata):
    away=metadata['stage']=='excited'
    self.results.put(dict(metadata,status='valid',latency_ms=0,decision=dict(
        person='present',attention='elsewhere' if away else 'toward_camera',
        gesture='thumbs_up' if metadata['stage']=='noticed' else 'none',intent='wait' if away else 'acknowledge')))
LiveInteraction._request=respond
raise SystemExit(main(['--manual','--live-interaction','--mode','dev','--probe-mock',*sys.argv[1:]]))
'''
    env=dict(os.environ,PALA_RUN_LOG_ROOT=str(tmp_path/'runtime'));env.pop('PALA_MAX_RUNTIME_S',None)
    console=tmp_path/'console.txt'
    with console.open('w') as output:
        p=subprocess.Popen([sys.executable,'-c',code,'--performances',str(library),'--probe-output',str(tmp_path/'probe')],stdin=subprocess.PIPE,stdout=output,stderr=subprocess.STDOUT,text=True,env=env)
        def wait_for(text):
            deadline=time.monotonic()+35
            while time.monotonic()<deadline:
                if text in console.read_text():return
                if p.poll() is not None:break
                time.sleep(.1)
            pytest.fail(console.read_text()[-4000:])
        try:
            wait_for('SOCIAL READY');p.stdin.write('arm\n');p.stdin.flush()
            wait_for('SOCIAL COMPLETE');p.stdin.write('shutdown\n');p.stdin.flush()
            assert p.wait(timeout=10)==0
        finally:
            if p.poll() is None:p.kill();p.wait()
    rows=[json.loads(l) for path in (tmp_path/'probe').glob('*/trials.jsonl') for l in path.read_text().splitlines()]
    requests=[r for r in rows if r.get('type')=='motion_request']
    assert [r['command'] for r in requests]==['notice','greet','excite','settle']
    assert all(r['accepted'] for r in requests)
    assert sum(r.get('type')=='sequence_continuation' for r in rows)==1
    assert not any(r.get('status')=='captured' and r.get('stage')=='attention' for r in rows)


def test_notice_allows_working_and_uncertainty_breaks_disengagement(tmp_path):
    c,p,s=setup(tmp_path)
    p.on_result(answer(c,p,s,attention='elsewhere'),s)
    assert p.take_request()=='notice'
    s=complete(c,p,s,'notice')
    p.on_result(answer(c,p,s,attention='elsewhere'),s)
    c.now+=5;p.on_result(answer(c,p,s,attention='elsewhere'),s)
    assert p.take_request() is None
    p.on_result(answer(c,p,s,person='uncertain',attention='uncertain',gesture='uncertain'),s)
    assert p.away_count==0
    c.now+=20;p.on_result(answer(c,p,s,attention='elsewhere'),s)
    assert p.take_request() is None
    c.now+=20;p.on_result(answer(c,p,s,attention='elsewhere'),s)
    assert p.take_request()=='settle'


def test_no_camera_request_during_motion(tmp_path):
    c,p,s=setup(tmp_path);p.on_result(answer(c,p,s),s);assert p.take_request()=='notice'
    c.now+=10;p.tick(snapshot(c),dict(s,busy=True,plan_id='notice'))
    assert p.worker is None and p.trial is None
    s=complete(c,p,s,'notice')
    p.tick(snapshot(c),s)
    assert p.worker is None  # Wait for fresh post-motion frame after .75s.


@pytest.mark.parametrize('start_stage',['rest','noticed'])
def test_early_thumb_runs_greet_then_excite_without_second_image(tmp_path,start_stage):
    c,p,s=setup(tmp_path)
    if start_stage=='noticed':
        p.on_result(answer(c,p,s),s);assert p.take_request()=='notice'
        s=complete(c,p,s,'notice')
    r=answer(c,p,s,gesture='thumbs_up')
    p.on_result(r,s);assert p.take_request()=='greet'
    p.on_result(r,s);assert p.take_request() is None
    s=complete(c,p,s,'greet')
    assert p.take_request()=='excite'
    assert p.worker is None
    s=complete(c,p,s,'excite')
    p.on_result(answer(c,p,s,gesture='thumbs_up'),s)
    assert p.take_request() is None


@pytest.mark.parametrize('cancel',['pause','reset','failure','closing','timeout','rejected','wrong_plan','expired'])
def test_early_thumb_continuation_cancellation(tmp_path,cancel):
    c,p,s=setup(tmp_path)
    p.on_result(answer(c,p,s,gesture='thumbs_up'),s);assert p.take_request()=='greet'
    if cancel in ('pause','reset'):p.command(cancel,dict(s,busy=True))
    elif cancel=='rejected':p.motion_result('greet',False)
    elif cancel in ('failure','closing'):
        p.tick(snapshot(c),dict(s,**{cancel:True}))
    elif cancel=='timeout':c.now=p.until+1
    elif cancel=='expired':c.now+=31
    if cancel=='wrong_plan':
        p.tick(snapshot(c),dict(s,state='attention',plan_id='unexpected',performance='attend'))
    else:s=complete(c,p,s,'greet')
    assert p.take_request() is None
    assert p.early_thumb_trial is None
