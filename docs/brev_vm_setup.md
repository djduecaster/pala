# Brev VM Setup Plan (Cosmos Integration)

This runbook is the exact setup path used to bring up Cosmos Reason 2 NIM on a Brev VM and verify connectivity from Jetson.

## 1) Create VM in Brev
- Type: `VM`
- GPU: `L40S` for bring-up (move to `H100` later if needed)
- Lifecycle: `stoppable`
- Storage: `256 GB`
- Jupyter on host: disable

## 2) VM bootstrap checks
Run on the Brev VM:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## 3) Authenticate Docker to NGC
Set `NGC_API_KEY` in the VM shell and login. Username is the literal string `$oauthtoken`.

```bash
export NGC_API_KEY='PASTE_REAL_KEY_HERE'
echo "len=${#NGC_API_KEY}"
printf '%s' "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

If you see `Cannot perform an interactive login from a non TTY device`, the key is usually empty in that shell.

## 4) Start Cosmos Reason 2
Current tested image:
- `nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0`

Fast path using project script (on VM, inside repo):

```bash
cd ~/pala
./tools/brev_bootstrap_cosmos.sh
```

Manual equivalent:

```bash
mkdir -p ~/.cache/nim
docker run -d --name cosmos \
  --restart unless-stopped \
  --gpus all \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0
```

## 5) Wait for readiness
```bash
docker ps
docker logs -f cosmos
```

Ready signal in logs:
- route list includes `/v1/chat/completions`
- uvicorn shows `Application startup complete`

Health checks:

```bash
curl -sS http://127.0.0.1:8000/v1/health/ready
curl -sS http://127.0.0.1:8000/v1/models
```

## 6) Open Brev networking
Open inbound `TCP 8000` in Brev networking/security rules, then test from Jetson:

```bash
curl -sv --connect-timeout 5 --max-time 10 http://<BREV_PUBLIC_IP>:8000/v1/health/ready
curl -sv --connect-timeout 5 --max-time 10 http://<BREV_PUBLIC_IP>:8000/v1/models
```

## 7) Basic model probe from Jetson
```bash
curl -sS --max-time 30 http://<BREV_PUBLIC_IP>:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"nvidia/cosmos-reason2-2b","messages":[{"role":"user","content":"Reply with exactly: READY"}],"max_tokens":16,"stream":false}'
```

## 8) Teardown / recreate
- Stop and remove container:

```bash
docker rm -f cosmos
```

- Delete VM in Brev when done.
- Recreate by rerunning this document or `./tools/brev_bootstrap_cosmos.sh`.

## 9) Notes
- Keep `NGC_API_KEY` and any future PALA cloud keys out of git.
- Keep caches in `~/.cache/nim` so container restarts are faster.
- If you change container image/tag, pass `--image` to the bootstrap script.
