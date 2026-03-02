"""Background-threaded OpenCV frame reader.

Reads RTSP frames on a background thread so the main loop never
blocks waiting on network I/O.  Only the most recent frame is kept;
older frames are silently dropped.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np


class OpenCVFrameReader:
    """Read frames from a VideoCapture on a background thread.

    Only the most recent frame is retained — stale frames are discarded.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ret = False
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "OpenCVFrameReader":
        self._running = True
        self._thread = threading.Thread(
            target=self._reader, name="frame-reader", daemon=True
        )
        self._thread.start()
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the most recent (ret, frame) — non-blocking."""
        with self._lock:
            return self._ret, self._frame

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _reader(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
