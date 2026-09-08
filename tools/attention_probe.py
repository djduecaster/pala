"""Run the main runtime at rest with four timed, read-only Gemini trials."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
import logging
import math
import os
from pathlib import Path
import queue
import threading
import time
from uuid import uuid4

from PIL import Image

from pala.behavior.attention import PROMPT, make_request, validate_response
from pala.behavior.model_clients import build_model_client, ModelResponse
from pala.perception.frame_cache import FrameSnapshot

logger = logging.getLogger(__name__)
SCENARIOS = {'1': 'empty frame', '2': 'walk into view without looking at camera',
             '3': 'look directly at camera', '4': 'look at wall beside camera'}
HELP = ('1: empty frame | 2: enter without looking | 3: look at camera | '
        '4: look beside camera | cancel: cancel countdown | status | help | '
        'q/shutdown: return to zero and exit | stop/Ctrl-C: disable without recovery. '
        'Press a number then Enter. Each trial waits FIVE seconds. No model-driven movement.')


@dataclass
class Trial:
    trial_id: str
    scenario: str
    deadline: float
    last_count: int = 6


class AttentionProbe:
    """Supervisor-owned state; one auxiliary network worker, no motion authority."""
    request_builder = staticmethod(make_request)
    response_validator = staticmethod(validate_response)
    def __init__(self, output: Path, model: str, client, *, mock: bool = False, clock=time.monotonic):
        self.output, self.model, self.client = output, model, client
        self.mock, self.clock = mock, clock
        self.trial: Trial | None = None
        self.worker: threading.Thread | None = None
        self.results: queue.Queue = queue.Queue(maxsize=1)
        self.inflight: dict | None = None
        self.ready_announced = False
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / 'prompt.txt').write_text(PROMPT)
        (self.output / 'session.json').write_text(json.dumps({'model': model, 'mock': mock, 'scenarios': SCENARIOS,
            'countdown_s': 5, 'request_timeout_s': 20, 'sdk_retries': 0, 'posture': 'rest', 'motion_from_model': False}, indent=2))

    @classmethod
    def from_args(cls, args):
        mock = args.probe_mock
        key = None
        if not mock:
            key = Path(args.gemini_key_file).expanduser().read_text().strip() if args.gemini_key_file else (
                os.getenv('GEMINI_API_KEY') or os.getenv('PALA_GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'))
            if not key:
                raise ValueError('Set GEMINI_API_KEY or --gemini-key-file before starting the probe')
            if not args.gemini_model:
                raise ValueError('Set PALA_GEMINI_MODEL or --gemini-model to your available Gemini model ID')
        client = None if mock else build_model_client(provider='gemini',
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/', api_key=key)
        probe = cls(Path(args.probe_output) / (time.strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6]),
                   args.gemini_model or 'mock', client, mock=mock)
        (probe.output / 'performances.json').write_text(Path(args.performances).read_text())
        return probe

    def record(self, row: dict) -> None:
        with (self.output / 'trials.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')

    @staticmethod
    def at_rest(state: dict) -> bool:
        return state['state'] == 'rest' and not state['busy'] and not state['closing'] and not state['failure']

    def command(self, token: str, state: dict) -> None:
        if token == 'cancel':
            if self.trial:
                self.record({'trial_id': self.trial.trial_id, 'scenario': self.trial.scenario, 'status': 'canceled'})
                self.trial = None
                logger.info('Countdown canceled; no image sent')
            else:
                logger.info('No countdown to cancel; an already-sent request cannot be recalled')
        elif token not in SCENARIOS:
            logger.info('Probe accepts only 1–4, cancel, help, status, shutdown, or stop')
        elif not self.at_rest(state):
            logger.info('Wait until startup completes at rest')
        elif self.trial or self.inflight:
            logger.info('Trial already active; request rejected, not queued')
        else:
            self.trial = Trial(uuid4().hex[:12], token, self.clock() + 5)
            logger.info('Trial %s: %s. Snapshot in FIVE seconds.', self.trial.trial_id, SCENARIOS[token])

    def tick(self, snapshot: FrameSnapshot | None, state: dict) -> None:
        now = self.clock()
        if self.at_rest(state) and not self.ready_announced:
            logger.info('PROBE READY at rest. %s Logs: %s', HELP, self.output)
            self.ready_announced = True
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            result = None
        if result is not None:
            self.record(result)
            self.inflight = None
            self.on_result(result, state)
            logger.info('RESULT %s status=%s latency_ms=%.0f decision=%s error=%s', result['trial_id'],
                        result['status'], result['latency_ms'], result.get('decision'), result.get('error'))
        if self.trial is None:
            return
        trial = self.trial
        if not self.at_rest(state):
            self.command('cancel', state)
            return
        remaining = max(0, math.ceil(trial.deadline - now))
        if remaining != trial.last_count and remaining > 0:
            logger.info('Snapshot in %d...', remaining)
            trial.last_count = remaining
        if now < trial.deadline:
            return
        # Require a frame captured AFTER the countdown, not a cached earlier image.
        if snapshot is None or snapshot.mono_ns / 1e9 < trial.deadline or not 0 <= now - snapshot.mono_ns / 1e9 <= .5:
            if now > trial.deadline + 2:
                self.record({'trial_id': trial.trial_id, 'scenario': trial.scenario, 'status': 'no_fresh_frame'})
                logger.warning('No fresh post-countdown frame; trial canceled without API call')
                self.trial = None
            return
        image = Image.fromarray(snapshot.frame.copy())
        image.thumbnail((960, 960))
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=90)
        jpeg = buf.getvalue()
        (self.output / f'{trial.trial_id}.jpg').write_bytes(jpeg)
        metadata = {'trial_id': trial.trial_id, 'scenario': trial.scenario, 'scenario_description': SCENARIOS.get(trial.scenario, 'continuous live observation'),
                    'frame_mono_ns': snapshot.mono_ns, 'capture_delay_ms': (snapshot.mono_ns / 1e9 - trial.deadline) * 1000,
                    'frame_age_ms': (now - snapshot.mono_ns / 1e9) * 1000, 'captured_wall_s': time.time(),
                    'image': f'{trial.trial_id}.jpg', 'model': self.model, 'mock': self.mock}
        metadata.update(self.capture_context(state))
        self.record(dict(metadata, status='captured'))
        self.trial, self.inflight = None, metadata
        logger.info('SNAPSHOT saved; sending one image for interpretation. Lamp is stationary.')
        self.worker = threading.Thread(target=self._request, args=(jpeg, metadata), daemon=True, name='attention-request')
        self.worker.start()

    def capture_context(self, state: dict) -> dict:
        return {}

    def on_result(self, result: dict, state: dict) -> None:
        pass

    def mock_decision(self) -> dict:
        return {'person': 'uncertain', 'attention': 'uncertain', 'intent': 'wait',
                'evidence': 'Mock response; no visual inference performed.'}

    def _request(self, jpeg: bytes, metadata: dict) -> None:
        started = self.clock()
        result = dict(metadata)
        try:
            if self.mock:
                body = self.mock_decision()
                response = ModelResponse(True, 200, 0, {'choices': [{'message': {'content': json.dumps(body)}}]}, None)
            else:
                response = self.client.chat(self.request_builder(jpeg, self.model))
            result['http_status'] = response.status_code
            # Retain only assistant output and usage, never request headers or provider errors.
            if response.ok and response.response_json:
                result['response'] = {k: response.response_json.get(k) for k in ('choices', 'usage', 'model')}
            try:
                result['decision'] = self.response_validator(response)
                result['status'] = 'valid'
            except ValueError as exc:
                result.update(status='invalid', error=str(exc))
        except Exception as exc:
            result.update(status='request_failed', error=type(exc).__name__)
        result['latency_ms'] = (self.clock() - started) * 1000
        self.results.put(result)

    def close(self) -> None:
        if self.trial:
            self.record({'trial_id': self.trial.trial_id, 'scenario': self.trial.scenario, 'status': 'canceled_on_exit'})
            self.trial = None
        if self.inflight:
            try:
                self.record(self.results.get_nowait())
            except queue.Empty:
                self.record(dict(self.inflight, status='abandoned_on_exit'))
            self.inflight = None
        # Worker never touches files or actuators; shutdown never waits on network.


def main() -> int:
    import sys
    from pala.main import main as runtime_main
    return runtime_main(['--manual', '--attention-probe', *sys.argv[1:]])


if __name__ == '__main__':
    raise SystemExit(main())
