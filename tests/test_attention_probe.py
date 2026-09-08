import json
from pathlib import Path
import threading

import numpy as np
import pytest

from pala.behavior.attention import make_request, validate_response
from pala.behavior.model_clients import ModelResponse
from pala.perception.frame_cache import FrameSnapshot
from tools.attention_probe import AttentionProbe

REST = dict(state='rest', busy=False, closing=False, failure=None)


class Clock:
    now = 10.0
    def __call__(self):
        return self.now


def response(**updates):
    body = dict(person='present', attention='toward_camera', intent='acknowledge', evidence='Face points toward camera.')
    body.update(updates)
    return ModelResponse(True, 200, 12, {'choices': [{'message': {'content': json.dumps(body)}}]}, None)


class Client:
    def __init__(self):
        self.requests = []
    def chat(self, request):
        self.requests.append(request)
        return response()


def snapshot(clock):
    return FrameSnapshot(np.zeros((8, 12, 3), dtype=np.uint8), int(clock.now * 1e9), None)


def test_five_second_countdown_fresh_capture_and_unbiased_request(tmp_path):
    clock, client = Clock(), Client()
    probe = AttentionProbe(tmp_path / 'probe', 'test-model', client, clock=clock)
    probe.command('3', REST)
    probe.command('4', REST)  # no queue or replacement
    clock.now = 14.99
    probe.tick(snapshot(clock), REST)
    assert not client.requests
    old_frame = snapshot(clock)
    clock.now = 15.1
    probe.tick(old_frame, REST)
    assert not client.requests  # frame taken before countdown is not eligible
    probe.tick(snapshot(clock), REST)
    probe.worker.join(timeout=2)
    probe.tick(snapshot(clock), REST)
    assert len(client.requests) == 1
    text = json.dumps(client.requests[0].messages)
    assert 'scenario' not in text.lower()
    assert client.requests[0].max_retries == 0
    assert client.requests[0].timeout_s == 20
    rows = [json.loads(s) for s in (probe.output / 'trials.jsonl').read_text().splitlines()]
    assert rows[-1]['status'] == 'valid'
    assert rows[-1]['scenario'] == '3'
    assert rows[-1]['decision']['intent'] == 'acknowledge'
    assert len(list(probe.output.glob('*.jpg'))) == 1
    assert probe.inflight is None


@pytest.mark.parametrize('updates', [dict(intent='move'), dict(extra='x'), dict(person='absent'), dict(evidence=''), dict(attention='elsewhere'), dict(person='uncertain')])
def test_reject_bad_decision(updates):
    with pytest.raises(ValueError):
        validate_response(response(**updates))


def test_absence_and_uncertainty_are_wait():
    assert validate_response(response(person='absent', attention='not_applicable', intent='wait'))['intent'] == 'wait'
    assert validate_response(response(person='uncertain', attention='uncertain', intent='wait'))['intent'] == 'wait'
    with pytest.raises(ValueError, match='provider_request_failed'):
        validate_response(ModelResponse(False, 401, 0, None, 'SECRET'))


def test_rest_gate_cancel_and_stale_camera(tmp_path):
    clock, client = Clock(), Client()
    probe = AttentionProbe(tmp_path / 'probe', 'test', client, clock=clock)
    probe.command('1', dict(REST, busy=True))
    assert probe.trial is None
    probe.command('1', REST)
    probe.command('cancel', REST)
    clock.now += 7
    probe.tick(snapshot(clock), REST)
    assert not client.requests
    probe.command('1', REST)
    clock.now += 7.1
    probe.tick(None, REST)
    assert probe.trial is None
    assert not client.requests
    assert 'no_fresh_frame' in (probe.output / 'trials.jsonl').read_text()


def test_shutdown_does_not_wait_for_network_or_leak_errors(tmp_path):
    clock = Clock()
    release = threading.Event()
    class SlowClient:
        def chat(self, request):
            release.wait(2)
            raise ValueError('SECRET_API_KEY')
    probe = AttentionProbe(tmp_path / 'probe', 'test', SlowClient(), clock=clock)
    probe.command('1', REST)
    clock.now += 5
    probe.tick(snapshot(clock), REST)
    probe.close()  # no join; records abandoned request
    assert probe.worker.is_alive()
    release.set()
    probe.worker.join(timeout=2)
    result = probe.results.get_nowait()
    assert result['error'] == 'ValueError'
    assert 'SECRET' not in json.dumps(result)
    assert 'abandoned_on_exit' in (probe.output / 'trials.jsonl').read_text()


def test_key_missing_fails_before_hardware(monkeypatch, tmp_path):
    import pala.main as main
    for name in ('GEMINI_API_KEY', 'PALA_GEMINI_API_KEY', 'GOOGLE_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(main, '_build_servo', lambda _: pytest.fail('Hardware initialized'))
    with pytest.raises(ValueError, match='GEMINI_API_KEY'):
        main.main(['--manual', '--attention-probe', '--mode', 'jetson_full', '--enable'])
    with pytest.raises(ValueError, match='limited to dev'):
        main.main(['--manual', '--attention-probe', '--probe-mock', '--mode', 'jetson_full', '--enable'])


def test_probe_request_disables_sdk_retries(monkeypatch):
    from types import SimpleNamespace
    from pala.behavior.model_clients.openai_compat_client import OpenAICompatClient
    calls = []
    class SDK:
        def with_options(self, **kwargs):
            calls.append(kwargs)
            return self
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(status_code=200, parse=lambda: {'choices': []})
    sdk = SDK()
    sdk.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=sdk))
    monkeypatch.setattr('pala.behavior.model_clients.openai_compat_client._get_openai_client', lambda **kwargs: sdk)
    assert OpenAICompatClient('https://example.invalid', 'secret').chat(make_request(b'jpeg', 'test')).ok
    assert calls[0] == {'max_retries': 0}
    assert calls[1]['timeout'] == 20
    assert 'max_retries' not in calls[1]
