"""Lightweight motion gate using frame differencing.

Compares consecutive frames to detect significant pixel changes.
When the scene is static, returns False so the caller can skip
expensive detection (YOLO).  This avoids burning GPU/CPU cycles
on frames where nothing has changed.
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    """Determine whether a frame has enough motion to warrant detection.

    Parameters
    ----------
    threshold : float
        Percentage (0-100) of pixels that must change to consider the
        frame as having motion.  Default 0.5 means 0.5% of pixels.
    blur_size : int
        Gaussian blur kernel size applied before differencing to
        reduce noise.  Must be odd.
    diff_threshold : int
        Per-pixel intensity difference (0-255) required to count as
        a changed pixel.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        blur_size: int = 21,
        diff_threshold: int = 25,
    ) -> None:
        self._threshold = threshold
        self._blur_size = blur_size
        self._diff_threshold = diff_threshold
        self._prev_gray: np.ndarray | None = None

    def has_motion(self, frame: np.ndarray) -> bool:
        """Return True if the frame differs enough from the previous one.

        Always returns True for the first frame (no baseline yet).

        Downsamples to a tiny resolution first so blur + diff is nearly
        free on CPU — motion detection doesn't need high resolution.
        """
        # Downsample to ~160px wide before any processing
        h, w = frame.shape[:2]
        small_w = 160
        small_h = max(1, int(h * (small_w / w)))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._blur_size, self._blur_size), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return True

        delta = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        changed_pixels = np.count_nonzero(delta > self._diff_threshold)
        total_pixels = gray.shape[0] * gray.shape[1]
        pct_changed = (changed_pixels / total_pixels) * 100

        return pct_changed >= self._threshold
