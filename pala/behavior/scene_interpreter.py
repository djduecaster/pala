from __future__ import annotations

import time
from typing import Callable, Optional

from ..types import PerceptionState
from .models import SceneObservation

_VALID_ZONES = {"left", "center", "right"}


class SceneInterpreter:
    """Converts raw perception packets into stable scene observations + events."""

    def __init__(self, *, clock: Optional[Callable[[], float]] = None):
        self._clock = clock or time.monotonic
        self._last_person_present = False
        self._curr_zone: Optional[str] = None
        self._zone_start_s: Optional[float] = None

    def observe(self, st: Optional[PerceptionState]) -> SceneObservation:
        now = self._clock()
        if st is None:
            self._last_person_present = False
            self._curr_zone = None
            self._zone_start_s = None
            return SceneObservation(
                ts_mono_s=now,
                person_present=False,
                person_conf=None,
                zone=None,
                zone_changed=False,
                zone_dwell_s=0.0,
                activity_hint=None,
                event="no_signal",
            )

        person_present = st.primary_person is not None
        person_conf = float(st.primary_person_conf) if st.primary_person_conf is not None else None
        zone = self._zone_for_state(st)
        zone_changed = False
        if zone != self._curr_zone:
            zone_changed = zone is not None and self._curr_zone is not None
            self._curr_zone = zone
            self._zone_start_s = now if zone is not None else None

        zone_dwell_s = 0.0
        if self._zone_start_s is not None and zone is not None:
            zone_dwell_s = max(0.0, now - self._zone_start_s)

        activity_hint = None
        if isinstance(st.debug, dict):
            raw_activity = st.debug.get("activity_hint")
            if isinstance(raw_activity, str):
                token = raw_activity.strip()
                if token:
                    activity_hint = token

        event = self._classify_event(person_present=person_present, zone_changed=zone_changed)
        self._last_person_present = person_present
        if not person_present:
            self._curr_zone = None
            self._zone_start_s = None

        return SceneObservation(
            ts_mono_s=now,
            person_present=person_present,
            person_conf=person_conf,
            zone=zone,
            zone_changed=zone_changed,
            zone_dwell_s=zone_dwell_s,
            activity_hint=activity_hint,
            event=event,
        )

    def _classify_event(self, *, person_present: bool, zone_changed: bool) -> str:
        if person_present and not self._last_person_present:
            return "person_entered"
        if (not person_present) and self._last_person_present:
            return "person_exited"
        if person_present and zone_changed:
            return "zone_changed"
        if person_present:
            return "person_present"
        return "no_person"

    @staticmethod
    def _zone_for_state(st: PerceptionState) -> Optional[str]:
        if st.primary_person is None:
            return None
        if isinstance(st.debug, dict):
            raw_zone = st.debug.get("zone_hint")
            if isinstance(raw_zone, str):
                token = raw_zone.strip().lower()
                if token in _VALID_ZONES:
                    return token
        cx = float(st.primary_person.cx)
        if cx < 0.33:
            return "left"
        if cx < 0.66:
            return "center"
        return "right"

