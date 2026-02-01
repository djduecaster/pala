"""Servo interfaces and dummy implementation.

TODO: Port legacy PCA9685 driver from ../pala_old/pala_project/src/hardware/servos.py
      into pala/hardware/servo_pca9685.py
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List
import logging
import time
import math

logger = logging.getLogger(__name__)

class ServoInterface:
    def set_angles(self, joint_angles_rad: Iterable[float]) -> None:
        raise NotImplementedError

    def enable(self, on: bool) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class ServoCalibration:
    bus_number: int
    address: int
    frequency: int
    channels: List[int]
    min_pulse_us: List[float]
    max_pulse_us: List[float]
    angle_scales: List[float]
    angle_offsets_deg: List[float]
    reverses: List[bool]


class DummyServo(ServoInterface):
    def __init__(self, log_every: int = 20):
        self._enabled = True
        self._counter = 0
        self._log_every = max(1, int(log_every))
        self._last = None

    def set_angles(self, joint_angles_rad: Iterable[float]) -> None:
        self._last = list(joint_angles_rad)
        self._counter += 1
        if self._counter % self._log_every == 0:
            logger.info("dummy servo angles(rad)=%s", self._last)

    def enable(self, on: bool) -> None:
        if self._enabled != on:
            self._enabled = on
            state = "enabled" if on else "disabled"
            logger.info("dummy servo %s", state)

    def shutdown(self) -> None:
        self.enable(False)
        time.sleep(0.01)


class PCA9685Servo(ServoInterface):
    def __init__(self, calibration: ServoCalibration) -> None:
        from .servo_pca9685 import PCA9685Servo as _PCA9685Servo

        self._backend = _PCA9685Servo(
            bus_number=calibration.bus_number,
            address=calibration.address,
            frequency_hz=calibration.frequency,
            channels=calibration.channels,
            min_pulse_us=calibration.min_pulse_us,
            max_pulse_us=calibration.max_pulse_us,
            angle_scales=calibration.angle_scales,
            angle_offsets_deg=calibration.angle_offsets_deg,
            reverses=calibration.reverses,
        )

    def set_angles(self, joint_angles_rad: Iterable[float]) -> None:
        angles_deg = [math.degrees(angle) for angle in joint_angles_rad]
        self._backend.set_angles_deg(angles_deg)

    def enable(self, on: bool) -> None:
        self._backend.enable(on)

    def shutdown(self) -> None:
        self._backend.shutdown()
