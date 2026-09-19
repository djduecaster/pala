# Behavior memory boundary

The default hold policy has no scene interpretation or model memory. The opt-in
live interaction maintains session-local state: current stage, active performance,
presence already noticed, greeting/excitement limits, pointing release/cooldown,
observation generation, and automatic rearming.

Presence persists across automatic rearms so a seated person does not repeatedly
trigger notice. Two valid absent observations at rest spanning four seconds allow
a later arrival to be noticed again. This is bookkeeping, not person recognition.
The state is not persistent identity or autobiographical memory. A separate
summarizer, long-term learning, and transcript memory are outside V1.
