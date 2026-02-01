import os

import pala.main as pala_main


def test_runtime_starts_in_dev_mode(monkeypatch):
    monkeypatch.setenv("PALA_MAX_RUNTIME_S", "0.5")
    result = pala_main.main()
    assert result == 0
