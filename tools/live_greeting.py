"""Supervised Gemini-triggered greeting; at most one motion per explicit arm."""
from __future__ import annotations

import json
import logging
from uuid import uuid4

from tools.attention_probe import AttentionProbe, Trial

LOG = logging.getLogger(__name__)
HELP = ('arm: five seconds to position, then observe for up to 60 seconds and greet once | '
        'pause: stop observations, discard pending decisions (does not abort active motion) | '
        'reset: settle to rest, disarmed | status | help | shutdown/q: zero and disable | '
        'stop/Ctrl-C: disable immediately. No observations while moving or after greeting.')


class LiveGreeting(AttentionProbe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.armed = False
        self.generation = 0
        self.until = 0.0
        self.next_capture = 0.0
        self.pending_request = None
        self.fresh_view = False
        self.context_plan_id = None
        self.live_ready = False
        path = self.output / 'session.json'
        session = json.loads(path.read_text())
        session.update(mode='live_greeting', motion_from_model=True, max_greetings_per_arm=1,
                       max_frame_age_at_decision_s=6, arm_window_s=60, inter_request_pause_s=1)
        path.write_text(json.dumps(session, indent=2))

    def disarm(self) -> None:
        self.armed = False
        self.generation += 1
        self.trial = None
        self.pending_request = None

    def command(self, token: str, state: dict) -> None:
        if token in {'pause', 'cancel', 'reset'}:
            self.disarm()
            if token == 'reset':
                self.pending_request = 'settle'
            LOG.info('Live observation paused; pending model decisions invalidated')
        elif token == 'arm':
            if not self.at_rest(state) or state.get('greeted') or self.inflight or self.armed:
                LOG.info('Cannot arm: require idle rest, no previous greeting, and no pending request')
                return
            self.generation += 1
            self.armed = True
            self.context_plan_id = state['plan_id']
            self.next_capture = self.clock() + 5
            self.until = self.next_capture + 60
            self.record({'type': 'arm', 'generation': self.generation, 'monotonic_s': self.clock()})
            LOG.info('ARMED: first snapshot in five seconds; one greeting maximum')
        else:
            LOG.info(HELP)

    def capture_context(self, state: dict) -> dict:
        return {'generation': self.generation, 'context_plan_id': state['plan_id']}

    def on_result(self, result: dict, state: dict) -> None:
        age = self.clock() - result['frame_mono_ns'] / 1e9
        eligible = (self.armed and self.clock() <= self.until and self.at_rest(state)
                    and not state.get('greeted') and self.fresh_view
                    and result.get('generation') == self.generation
                    and result.get('context_plan_id') == state['plan_id']
                    and 0 <= age <= 6)
        trigger = eligible and result.get('status') == 'valid' and result['decision']['intent'] == 'acknowledge'
        self.record({'type': 'decision_gate', 'trial_id': result['trial_id'],
                     'eligible': eligible, 'trigger_greeting': trigger, 'image_age_s': age})
        self.next_capture = self.clock() + 1
        if trigger:
            self.disarm()
            self.pending_request = 'greet'
            LOG.info('Gemini acknowledged apparent attention: requesting ONE greeting (image age %.2fs)', age)
        elif not eligible:
            LOG.info('Decision ignored: paused, stale, camera unavailable, or posture context changed')

    def tick(self, snapshot, state: dict) -> None:
        self.fresh_view = snapshot is not None and 0 <= self.clock() - snapshot.mono_ns / 1e9 <= .5
        if self.at_rest(state) and not self.live_ready:
            LOG.info('LIVE TEST READY at rest, disarmed. %s', HELP)
            self.live_ready = True
        # Suppress the numbered probe's ready message, retaining its capture/worker path.
        self.ready_announced = True
        if self.armed and (not self.at_rest(state) or state['plan_id'] != self.context_plan_id or self.clock() > self.until):
            self.disarm()
            LOG.info('Observation window ended or posture changed; disarmed')
        if self.armed and self.trial is None and self.inflight is None:
            self.trial = Trial(uuid4().hex[:12], 'live', self.next_capture)
        super().tick(snapshot, state)
        if self.armed and self.trial is None and self.inflight is None:
            self.next_capture = max(self.next_capture, self.clock() + 1)

    def take_request(self):
        request, self.pending_request = self.pending_request, None
        return request


def main() -> int:
    import sys
    from pala.main import main as runtime_main
    return runtime_main(['--manual', '--live-greeting', *sys.argv[1:]])


if __name__ == '__main__':
    raise SystemExit(main())
