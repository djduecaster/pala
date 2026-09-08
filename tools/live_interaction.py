"""One explicitly armed desk interaction using accepted physical performances."""
from __future__ import annotations

import json
import logging
from uuid import uuid4

from pala.behavior.social_observation import PROMPT, make_request, validate_response
from tools.attention_probe import AttentionProbe, Trial

LOG = logging.getLogger(__name__)
HELP = ('arm: five-second countdown, then one interaction (up to 180s) | '
        'pause: stop observation, hold current posture | reset: settle and disarm | '
        'status | shutdown/q: zero and disable | stop/Ctrl-C: disable immediately. '
        'Sit down, look toward the lamp, give a thumbs-up, then return to work. '
        'An early thumbs-up triggers greeting followed by excitement without repeating the cue.')


class LiveInteraction(AttentionProbe):
    request_builder = staticmethod(make_request)
    response_validator = staticmethod(validate_response)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.armed = False
        self.generation = 0
        self.until = 0.0
        self.next_capture = 0.0
        self.pending_request = None
        self.early_thumb_trial: str | None = None
        self.early_thumb_until = 0.0
        self.expected_motion = None
        self.context_plan_id = None
        self.stage = 'rest'
        self.fresh_view = False
        self.away_since = None
        self.away_count = 0
        self.ready_announced = True
        self.live_ready = False
        (self.output / 'prompt.txt').write_text(PROMPT)
        path = self.output / 'session.json'
        session = json.loads(path.read_text())
        session.update(mode='live_interaction', motion_from_model=True,
            max_interactions_per_arm=1, max_frame_age_at_decision_s=6,
            arm_window_s=180, inter_request_pause_s=1, settle_observations=2,
            settle_min_span_s=4, stationary_delay_s=.75,
            early_thumbs_up='greet_then_excite', early_thumbs_up_sequence_timeout_s=30)
        path.write_text(json.dumps(session, indent=2))

    def mock_decision(self):
        return dict(super().mock_decision(), gesture='uncertain')

    @staticmethod
    def at_rest(state: dict) -> bool:
        # Base capture path calls this predicate; social observations allow all
        # supported stationary poses, never motion/shutdown/failure.
        return (state['state'] in {'rest','noticed','attention','excited'}
                and not state['busy'] and not state['closing'] and not state['failure'])

    def disarm(self):
        self.armed = False
        self.generation += 1
        self.trial = None
        self.pending_request = None
        self.expected_motion = None
        self.early_thumb_trial = None
        self.away_since = None
        self.away_count = 0

    def command(self, token: str, state: dict):
        if token in {'pause','cancel','reset'}:
            self.disarm()
            if token == 'reset' and self.at_rest(state) and state['state'] != 'rest':
                self.pending_request = 'settle'
            LOG.info('Paused; pending model decisions invalidated. %s', state['state'])
        elif token == 'arm':
            if self.armed or self.inflight or not self.at_rest(state) or state['state'] != 'rest':
                LOG.info('Arm requires idle rest and no pending model request')
                return
            self.generation += 1
            self.armed = True
            self.early_thumb_trial = None
            self.stage = 'rest'
            self.context_plan_id = state['plan_id']
            self.next_capture = self.clock() + 5
            self.until = self.next_capture + 180
            self.away_since, self.away_count = None, 0
            self.record({'type':'arm','generation':self.generation,'monotonic_s':self.clock()})
            LOG.info('ARMED: first image in five seconds; one complete desk interaction')
        else:
            LOG.info(HELP)

    def capture_context(self, state):
        return {'generation':self.generation, 'context_plan_id':state['plan_id'], 'stage':self.stage}

    def on_result(self, result, state):
        now = self.clock()
        age = now - result['frame_mono_ns'] / 1e9
        eligible = (self.armed and now <= self.until and self.expected_motion is None
            and self.at_rest(state) and state['state'] == self.stage and self.fresh_view
            and result.get('generation') == self.generation
            and result.get('stage') == self.stage
            and result.get('context_plan_id') == state['plan_id'] == self.context_plan_id
            and 0 <= age <= 6)
        request = None
        if eligible and result.get('status') == 'valid':
            d = result['decision']
            toward = d['person'] == 'present' and d['attention'] == 'toward_camera'
            away = d['person'] == 'absent' or (d['person'] == 'present' and d['attention'] == 'elsewhere')
            if self.stage in {'rest', 'noticed'} and toward and d['gesture'] == 'thumbs_up':
                request = 'greet'
                self.early_thumb_trial = result['trial_id']
                self.early_thumb_until = now + 30
                self.record({'type':'sequence_accepted', 'trial_id':result['trial_id'],
                             'sequence':['greet','excite'], 'monotonic_s':now})
            elif self.stage == 'rest':
                if d['person'] == 'present': request = 'notice'
            elif self.stage == 'noticed' and toward:
                request = 'greet'
            elif self.stage == 'attention' and toward and d['gesture'] == 'thumbs_up':
                request = 'excite'
            if away and self.stage != 'rest':
                self.away_count += 1
                if self.away_since is None: self.away_since = now
                # Someone still working gets time to invite interaction after notice.
                grace = 20 if self.stage == 'noticed' else 4
                if self.away_count >= 2 and now - self.away_since >= grace:
                    request = 'settle'
            else:
                self.away_since, self.away_count = None, 0
        else:
            self.away_since, self.away_count = None, 0
        self.record({'type':'decision_gate','trial_id':result['trial_id'], 'stage':self.stage,
                     'eligible':eligible,'request':request,'image_age_s':age})
        self.next_capture = now + 1
        if request:
            self.generation += 1
            self.trial = None
            self.pending_request = request
            self.expected_motion = request
            self.away_since, self.away_count = None, 0
            LOG.info('SOCIAL decision: %s -> %s (image age %.2fs)', self.stage, request, age)

    def motion_result(self, request, accepted):
        if not accepted:
            self.disarm()
            LOG.warning('Motion request rejected; disarmed')

    def take_request(self):
        request, self.pending_request = self.pending_request, None
        return request

    def tick(self, snapshot, state):
        now = self.clock()
        self.fresh_view = snapshot is not None and 0 <= now - snapshot.mono_ns / 1e9 <= .5
        if self.at_rest(state) and state['state'] == 'rest' and not self.live_ready:
            LOG.info('SOCIAL READY at rest, disarmed. %s', HELP)
            self.live_ready = True
        if self.armed:
            if now > self.until or state['closing'] or state['failure']:
                self.disarm()
                LOG.info('Observation window ended; holding posture, disarmed')
            elif self.expected_motion:
                target = {'notice':'noticed','greet':'attention','excite':'excited','settle':'rest'}[self.expected_motion]
                if self.at_rest(state) and state['plan_id'] != self.context_plan_id:
                    if state['state'] != target or not state.get('performance', '').startswith(self.expected_motion):
                        self.disarm()
                    elif target == 'rest':
                        self.disarm()
                        LOG.info('SOCIAL COMPLETE: settled to rest; arm explicitly for another interaction')
                    else:
                        self.stage = target
                        self.context_plan_id = state['plan_id']
                        self.expected_motion = None
                        self.next_capture = now + .75
                        if target == 'attention' and self.early_thumb_trial:
                            source_trial = self.early_thumb_trial
                            self.early_thumb_trial = None
                            if now <= self.early_thumb_until:
                                # Finish the already accepted two-part response;
                                # do not reinterpret an old image as a new cue.
                                self.generation += 1
                                self.pending_request = 'excite'
                                self.expected_motion = 'excite'
                                self.record({'type':'sequence_continuation',
                                             'trial_id':source_trial,'request':'excite',
                                             'monotonic_s':now})
                                LOG.info('Greeting complete; continuing accepted thumbs-up response with excitement')
                            else:
                                self.record({'type':'sequence_expired','trial_id':source_trial})
            elif state['busy'] or state['plan_id'] != self.context_plan_id:
                self.disarm()
                LOG.info('Unexpected posture change; disarmed')
        if self.armed and not self.expected_motion and self.trial is None and self.inflight is None:
            self.trial = Trial(uuid4().hex[:12], 'social', self.next_capture)
        super().tick(snapshot, state)
        if self.armed and not self.expected_motion and self.trial is None and self.inflight is None:
            self.next_capture = max(self.next_capture, now + 1)


def main():
    import sys
    from pala.main import main as runtime_main
    return runtime_main(['--manual','--live-interaction','--performances','config/desk_performances.json',*sys.argv[1:]])


if __name__ == '__main__':
    raise SystemExit(main())
