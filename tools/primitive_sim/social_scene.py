"""Generate the simulation-only desk interaction using the runtime executor."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from pala.config import load_config
from pala.types import ActionPlan
from tools.primitive_sim.run import _load_viewer_geometry_from_config
from tools.primitive_sim.simulate import SimSegment, simulate_segments, write_trace_json

ROOT = Path(__file__).resolve().parents[2]


def build_trace() -> dict[str, Any]:
    cfg = load_config(str(ROOT / 'config/robot.yaml'))
    scene = json.loads((ROOT / 'tools/primitive_sim/desk_scene.json').read_text())
    performances = json.loads((ROOT / 'config/desk_performances.json').read_text())
    poses = performances['poses_deg']
    segments: list[SimSegment] = []
    cues: dict[str, dict[str, str]] = {}
    previous = [0.0] * len(cfg.joint_names)
    for beat in scene['beats']:
        for index, step in enumerate(performances['performances'][beat['performance']]['steps']):
            name = f"{beat['id']} / {index + 1}: {step['name']}"
            cues[name] = {k: beat[k] for k in ('id', 'title', 'human', 'intent')}
            target = poses[step['pose']] if 'pose' in step else step.get('target_deg')
            if target is not None:
                if len(target) != len(previous):
                    raise ValueError(f'{name}: wrong joint count')
                radians = [math.radians(v) for v in target]
                if any(not lo <= v <= hi for v, (lo, hi) in zip(radians, cfg.joint_limits_rad)):
                    raise ValueError(f'{name}: target outside configured limits')
                rate = float(step['rate_deg_s'])
                if not math.isfinite(rate) or rate <= 0:
                    raise ValueError(f'{name}: rate must be positive and finite')
                duration = max(abs(a - b) for a, b in zip(previous, target)) / rate + 2
                segments.append(SimSegment(name, ActionPlan(
                    primitive='move_to', command={'target_rad': radians, 'rate_rad_s': math.radians(rate), 'timeout_s': duration},
                    confidence=1, style='calm', explanation=step['name']), duration + 1, True))
                previous = target
            if step.get('hold_s', 0):
                segments.append(SimSegment(name, ActionPlan(primitive='hold', command={}, confidence=1,
                    style='calm', explanation=step['name']), step['hold_s'], False))
    trace = simulate_segments(joint_names=cfg.joint_names, joint_limits_rad=cfg.joint_limits_rad,
        segments=segments, hz=80, position_tolerance_rad=1e-6,
        style_profiles={'calm': {'rate_scale': 1, 'amp_scale': 1, 'duration_scale': 1, 'settle_scale': 1}})
    trace['metadata']['lamp_geometry'] = _load_viewer_geometry_from_config(ROOT / 'config/robot.yaml')
    trace['metadata']['scene_title'] = scene['title']
    trace['metadata']['simulation_only'] = True
    for sample in trace['samples']:
        sample['scene'] = cues[sample['segment']]
    failures = {s['status'] for s in trace['samples']} & {'timed_out', 'rejected', 'canceled'}
    if failures:
        raise ValueError(f'Scene failed: {failures}')
    return trace


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    trace = build_trace()
    output = ROOT / 'logs/primitive_sim/desk_scene.json'
    write_trace_json(output, trace)
    logging.info('Wrote %s (%.1f seconds)', output, trace['samples'][-1]['t_s'])


if __name__ == '__main__':
    main()
