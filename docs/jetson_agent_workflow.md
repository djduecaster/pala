# Jetson agent workflow

This document describes the direct Mac-to-Jetson workflow for agents and
operators. It is intentionally separate from the PALA runtime architecture:
the Jetson shell and tmux session are operator tooling, not another runtime
loop.

## SSH paths

The local SSH configuration currently provides two aliases:

- `jetson`: the USB device-mode path at the `192.168.55.x` link-local address.
- `jetson-wifi`: the Jetson's normal Wi-Fi address.

Use `jetson-wifi` for direct agent work when Wi-Fi is available. The existing
deployment scripts currently call the `jetson` alias directly, so changing
`JETSON_HOST` on the `make` command line does not yet change where
`deploy_jetson.sh` or `run_jetson.sh` connect.

Check the resolved target before troubleshooting a connection:

```bash
ssh -G jetson-wifi | grep -E '^(hostname|user|port) '
```

## Shared tmux session

The `pala` tmux session lives on the Jetson. It stays alive independently of
the SSH connection, which lets an operator attach to the same shell that an
agent is using.

Create it if needed:

```bash
ssh -T jetson-wifi 'tmux new-session -A -d -s pala'
```

Attach from a visible Mac terminal:

```bash
ssh -t jetson-wifi 'tmux attach -t pala'
```

Detach with `Ctrl+B`, then `D`. Detaching leaves commands running. Typing
`exit` ends the shell in that tmux pane.

## Agent interaction

Agents should use one command at a time and capture the pane after the command
has had time to finish. The shared pane is an interactive surface, so an agent
should not send keys while the operator is typing or while an interactive
program owns the prompt.

Send a short, literal command visibly into the shared shell:

```bash
ssh -T jetson-wifi \
  'tmux send-keys -t pala -l "hostname; pwd"; tmux send-keys -t pala Enter'
```

Read the visible command and recent output:

```bash
ssh -T jetson-wifi 'tmux capture-pane -p -t pala -S -80'
```

Check the pane without changing it:

```bash
ssh -T jetson-wifi \
  'tmux display-message -p -t pala "#{pane_current_command} #{pane_current_path}"'
```

For commands that need reliable exit status, structured output, or complex
quoting, agents should use a normal noninteractive SSH command and report the
result in the conversation. `send-keys` is for shared visibility; it is not a
replacement for a script runner.

## Long-running PALA runs

Keep the shared `pala` shell available for inspection and run long-lived PALA
processes in a named tmux window:

```bash
ssh -T jetson-wifi \
  'tmux new-window -d -t pala: -n pala-run "cd ~/pala && ./run_on_jetson.sh"'
```

List windows and capture the run window:

```bash
ssh -T jetson-wifi 'tmux list-windows -t pala'
ssh -T jetson-wifi 'tmux capture-pane -p -t pala:pala-run -S -120'
```

Before starting another run, check whether `pala-run` already exists and
whether a runtime is still active. Do not stack multiple hardware runtimes on
the same Jetson.

## `make go` and a future tmux target

Today, `make go` does this:

1. `deploy_jetson.sh` synchronizes the Mac checkout to `~/pala`.
2. `run_jetson.sh` opens a foreground SSH command and runs
   `run_on_jetson.sh` directly.

That foreground behavior is useful for a normal agent command, but it does not
create or attach to tmux. Automatically attaching would also keep `make go`
running for the entire lifetime of PALA, which makes the command awkward for
noninteractive agents.

The preferred future change is an explicit opt-in target such as
`make go-tmux`:

- deploy to the selected SSH alias;
- start `run_on_jetson.sh` in a dedicated `pala` tmux window;
- return the command promptly;
- let the operator stay attached to the session and let the agent use
  `capture-pane` for text logs.

An optional `make go-attach` could provide the foreground attached behavior for
manual demonstrations. Keeping these behaviors separate avoids changing the
meaning of the existing `make go` command. Implementing either target requires
an explicit change to the deployment scripts and Makefile; this document does
not change those scripts.

## Safety and logging notes

- Keep Mac and Jetson paths explicit in every command.
- Use `PALA_LOG_LEVEL=INFO` or `DEBUG` only when the extra output is useful.
- Prefer the run-scoped files under `~/pala/logs/runs/` for post-run evidence;
  use `make pull-logs` after a run when its SSH target is correct.
- tmux persistence does not replace the runtime deadman timeout or any local
  hardware safety limit.
