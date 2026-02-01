from __future__ import annotations

import importlib
import sys
import types


class _FakeSMBus:
    def __init__(self, _bus_number: int):
        self.writes = []
        self.reads = []

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self.writes.append((address, register, value))

    def read_byte_data(self, address: int, register: int) -> int:
        self.reads.append((address, register))
        return 0x00

    def close(self) -> None:
        return None


def _import_backend(monkeypatch):
    fake_mod = types.SimpleNamespace(SMBus=_FakeSMBus)
    monkeypatch.setitem(sys.modules, "smbus2", fake_mod)
    if "pala.hardware.servo_pca9685" in sys.modules:
        del sys.modules["pala.hardware.servo_pca9685"]
    return importlib.import_module("pala.hardware.servo_pca9685")


def _last_channel_write(writes, base):
    # Returns the last four writes for the channel base registers.
    channel_writes = [w for w in writes if w[1] in (base, base + 1, base + 2, base + 3)]
    return channel_writes[-4:]


def test_pca9685_writes_pwm_for_angles(monkeypatch):
    backend = _import_backend(monkeypatch)
    servo = backend.PCA9685Servo(
        bus_number=7,
        address=0x40,
        frequency_hz=50,
        channels=[0],
        min_pulse_us=[500],
        max_pulse_us=[2500],
        angle_scales=[1.0],
        angle_offsets_deg=[0.0],
        reverses=[False],
    )

    servo.set_angles_deg([0.0])
    base = 0x06
    last = _last_channel_write(servo._bus.writes, base)
    assert last[0] == (0x40, base, 0)
    assert last[1] == (0x40, base + 1, 0)
    # 500us at 50Hz -> 102.4 counts
    assert last[2] == (0x40, base + 2, 102)
    assert last[3] == (0x40, base + 3, 0)


def test_pca9685_reverse_and_clamp(monkeypatch):
    backend = _import_backend(monkeypatch)
    servo = backend.PCA9685Servo(
        bus_number=7,
        address=0x40,
        frequency_hz=50,
        channels=[0],
        min_pulse_us=[500],
        max_pulse_us=[2500],
        angle_scales=[1.0],
        angle_offsets_deg=[0.0],
        reverses=[True],
    )

    servo.set_angles_deg([200.0])
    base = 0x06
    last = _last_channel_write(servo._bus.writes, base)
    # Reversed 200 -> 180 - 200 = -20, clamped to 0 deg.
    assert last[2] == (0x40, base + 2, 102)
    assert last[3] == (0x40, base + 3, 0)


def test_pca9685_disable_zeroes_pwm(monkeypatch):
    backend = _import_backend(monkeypatch)
    servo = backend.PCA9685Servo(
        bus_number=7,
        address=0x40,
        frequency_hz=50,
        channels=[0, 1],
        min_pulse_us=[500, 500],
        max_pulse_us=[2500, 2500],
        angle_scales=[1.0, 1.0],
        angle_offsets_deg=[0.0, 0.0],
        reverses=[False, False],
    )

    servo.enable(False)
    base0 = 0x06
    base1 = 0x06 + 4
    last0 = _last_channel_write(servo._bus.writes, base0)
    last1 = _last_channel_write(servo._bus.writes, base1)
    assert last0 == [
        (0x40, base0, 0),
        (0x40, base0 + 1, 0),
        (0x40, base0 + 2, 0),
        (0x40, base0 + 3, 0),
    ]
    assert last1 == [
        (0x40, base1, 0),
        (0x40, base1 + 1, 0),
        (0x40, base1 + 2, 0),
        (0x40, base1 + 3, 0),
    ]
