from __future__ import annotations

from typing import Dict

DEFAULT_STYLE_PROFILES: Dict[str, Dict[str, float]] = {
    "calm": {
        "amp_scale": 0.85,
        "rate_scale": 0.9,
        "duration_scale": 1.1,
        "settle_scale": 1.1,
    },
    "curious": {
        "amp_scale": 1.15,
        "rate_scale": 1.15,
        "duration_scale": 0.9,
        "settle_scale": 0.9,
    },
    "focused": {
        "amp_scale": 0.7,
        "rate_scale": 1.0,
        "duration_scale": 0.85,
        "settle_scale": 1.0,
    },
}


def default_style_profiles() -> Dict[str, Dict[str, float]]:
    return {name: dict(values) for name, values in DEFAULT_STYLE_PROFILES.items()}
