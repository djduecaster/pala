"""PCA9685 servo backend using smbus2 for Jetson."""
from __future__ import annotations

from typing import Iterable, List
import math
import time

import smbus2


class PCA9685Servo:
    def __init__(
        self,
        *,
        bus_number: int,
        address: int,
        frequency_hz: int,
        channels: List[int],
        min_pulse_us: List[float],
        max_pulse_us: List[float],
        angle_scales: List[float],
        angle_offsets_deg: List[float],
        reverses: List[bool],
    ) -> None:
        self._bus = smbus2.SMBus(bus_number)
        self._address = address
        self._channels = list(channels)
        self._min_pulse = list(min_pulse_us)
        self._max_pulse = list(max_pulse_us)
        self._angle_scales = list(angle_scales)
        self._angle_offsets = list(angle_offsets_deg)
        self._reverses = list(reverses)
        self._enabled = True

        self._resolution = 4096
        self._freq = int(frequency_hz)

        # Software reset to clear any previous state.
        self._bus.write_byte_data(self._address, 0x00, 0x20)
        time.sleep(0.01)
        self._set_pwm_freq(self._freq)

    def _set_pwm_freq(self, freq: int) -> None:
        prescale = int(25000000.0 / (self._resolution * freq)) - 1
        old_mode = self._bus.read_byte_data(self._address, 0x00)
        self._bus.write_byte_data(self._address, 0x00, (old_mode & 0x7F) | 0x10)
        self._bus.write_byte_data(self._address, 0xFE, prescale)
        self._bus.write_byte_data(self._address, 0x00, old_mode)
        time.sleep(0.005)
        self._bus.write_byte_data(self._address, 0x00, old_mode | 0x80)

    def _angle_to_pwm(self, angle_deg: float, idx: int) -> int:
        pulse = self._min_pulse[idx] + (angle_deg / 180.0) * (self._max_pulse[idx] - self._min_pulse[idx])
        return int((pulse * self._resolution * self._freq) / 1_000_000.0)

    def _write_pwm(self, channel: int, on_time: int) -> None:
        base = 0x06 + 4 * channel
        self._bus.write_byte_data(self._address, base, 0)
        self._bus.write_byte_data(self._address, base + 1, 0)
        self._bus.write_byte_data(self._address, base + 2, on_time & 0xFF)
        self._bus.write_byte_data(self._address, base + 3, on_time >> 8)

    def set_angles_deg(self, angles_deg: Iterable[float]) -> None:
        values = list(angles_deg)
        if len(values) != len(self._channels):
            raise ValueError(
                f"expected {len(self._channels)} servo angles, got {len(values)}"
            )
        try:
            values = [float(value) for value in values]
            finite = all(math.isfinite(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("servo angles must be finite numbers") from exc
        if not finite:
            raise ValueError("servo angles must be finite numbers")
        if not self._enabled:
            return
        for idx, angle_deg in enumerate(values):
            mapped = angle_deg * self._angle_scales[idx] + self._angle_offsets[idx]
            if self._reverses[idx]:
                mapped = 180.0 - mapped
            if mapped < 0.0:
                mapped = 0.0
            elif mapped > 180.0:
                mapped = 180.0
            on_time = self._angle_to_pwm(mapped, idx)
            self._write_pwm(self._channels[idx], on_time)

    def disable(self, channels: Iterable[int] | None = None) -> None:
        if channels is None:
            channels = self._channels
        for ch in channels:
            base = 0x06 + 4 * ch
            self._bus.write_byte_data(self._address, base, 0)
            self._bus.write_byte_data(self._address, base + 1, 0)
            self._bus.write_byte_data(self._address, base + 2, 0)
            self._bus.write_byte_data(self._address, base + 3, 0)

    def enable(self, on: bool) -> None:
        if self._enabled == on:
            return
        self._enabled = on
        if not on:
            self.disable()

    def shutdown(self) -> None:
        self.enable(False)
        self._bus.close()
