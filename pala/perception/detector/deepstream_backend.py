from __future__ import annotations

from typing import List, Optional
import threading
import time

import numpy as np

from .interface import Detection, DetectorInterface


class DeepStreamDetector(DetectorInterface):
    def __init__(
        self,
        *,
        config_path: Optional[str],
        person_class_id: int = 0,
        conf_threshold: Optional[float] = None,
    ) -> None:
        self._config_path = config_path
        self._person_class_id = int(person_class_id)
        self._conf_threshold = conf_threshold

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._has_unread = False
        self._latest_dets: List[Detection] = []
        self._last_error: Optional[str] = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        with self._lock:
            self._latest_frame = frame
            self._has_unread = True
            self._cond.notify_all()
            if self._last_error is not None:
                raise RuntimeError(self._last_error)
            return list(self._latest_dets)

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        if not self._config_path:
            with self._lock:
                self._last_error = "DeepStream config_path is required"
                self._stop.set()
                self._cond.notify_all()
            return

        while not self._stop.is_set():
            with self._lock:
                while not self._stop.is_set() and not self._has_unread:
                    self._cond.wait(timeout=0.05)
                if self._stop.is_set():
                    break
                frame = self._latest_frame
                self._has_unread = False

            if frame is None:
                continue

            try:
                dets = self._infer(frame)
            except Exception as exc:
                with self._lock:
                    self._last_error = repr(exc)
                    self._stop.set()
                    self._cond.notify_all()
                return

            with self._lock:
                self._latest_dets = dets

    def _infer(self, _frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError("DeepStreamDetector inference not wired yet")
