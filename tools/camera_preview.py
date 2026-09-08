"""Loopback browser preview of the existing Jetson JPEG tap; no camera/servo ownership."""
from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shlex
import subprocess
import threading
import time

LOG = logging.getLogger(__name__)

# Runs on Jetson. Its monotonic clock is used ONLY to compute age on Jetson.
REMOTE = r'''
import base64, json, pathlib, time
root = pathlib.Path.home() / 'pala/logs/telemetry/preview'
while True:
    try:
        before = (root / 'latest.json').read_bytes()
        jpeg = (root / 'latest.jpg').read_bytes()
        after = (root / 'latest.json').read_bytes()
        if before != after:
            continue
        meta = json.loads(after)
        age = max(0, (time.monotonic_ns() - meta['mono_ns']) / 1e9)
        print(json.dumps(dict(image=base64.b64encode(jpeg).decode(), age_s=age,
                              frame_id=meta['frame_id'])), flush=True)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps(dict(error=type(exc).__name__)), flush=True)
    time.sleep(.25)
'''

PAGE = '''<!doctype html><meta charset="utf-8"><title>PALA Camera Framing</title>
<style>body{margin:0;background:#111820;color:#edf3f9;font:18px system-ui;padding:24px}h1{font-size:24px}#status{padding:12px;background:#263443;border-radius:8px}img{display:block;max-width:100%;max-height:75vh;margin:18px auto;object-fit:contain}small{color:#aab9c9}</style>
<h1>PALA · Camera framing</h1><div id="status">Connecting to Jetson…</div>
<img id="camera" alt="Waiting for camera frames"><small>Read-only preview. No servo controls or Gemini requests. Frame age includes time on Jetson and since receipt on Mac; SSH transit time is not measured.</small>
<script>
async function refresh(){
 const status=document.getElementById('status');
 try {const r=await fetch('/frame',{cache:'no-store'});const s=await r.json();
 if(s.image){document.getElementById('camera').src='data:image/jpeg;base64,'+s.image;}
 const stale=s.error || s.age_s>1;
 status.style.background=stale?'#74382a':'#234b3b';
 status.textContent=s.error?'STALE / disconnected: '+s.error:
 (stale?'STALE':'LIVE')+' · frame '+s.frame_id+' · age ≥ '+s.age_s.toFixed(2)+' s';
 }catch(e){status.textContent='Disconnected from local preview';status.style.background='#74382a';}
 setTimeout(refresh,250);
}refresh();
</script>'''


class Feed:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest = {'error': 'Waiting for first frame'}
        self.received = time.monotonic()

    def read(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            try:
                row = json.loads(line)
                with self.lock:
                    self.latest, self.received = row, time.monotonic()
            except ValueError:
                continue
        with self.lock:
            self.latest = {'error': 'SSH stream closed; restart preview'}

    def snapshot(self) -> dict:
        with self.lock:
            row = dict(self.latest)
            if 'age_s' in row:
                row['age_s'] += time.monotonic() - self.received
            return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jetson-host', default='jetson-wifi')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    feed = Feed()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ('/', '/frame'):
                self.send_error(404)
                return
            body = PAGE.encode() if self.path == '/' else json.dumps(feed.snapshot()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8' if self.path == '/' else 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    process = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', args.jetson_host,
        'python3 -u -c ' + shlex.quote(REMOTE)], stdout=subprocess.PIPE, text=True)
    threading.Thread(target=feed.read, args=(process,), daemon=True).start()
    LOG.info('Camera preview: http://127.0.0.1:%s', args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == '__main__':
    main()
