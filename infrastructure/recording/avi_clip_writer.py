"""AVI clip writer with automatic clip rotation.

Encapsulates OpenCV VideoWriter, clip rotation every N seconds,
and timestamped filename generation.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import numpy as np


class AviClipWriter:
    """Write annotated video frames to AVI clips, rotating every N seconds.

    Uses MJPEG codec so partial clips are playable even if the process
    is killed before the writer is properly released.

    Parameters
    ----------
    output_dir : str
        Directory where clip files are written.
    fps : float
        Frame rate of the output video.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    clip_duration_seconds : int
        How many seconds of footage per clip before rotation (default 30).
    """

    def __init__(
        self,
        output_dir: str,
        fps: float,
        width: int,
        height: int,
        clip_duration_seconds: int = 30,
    ) -> None:
        self._output_dir = output_dir
        self._fps = fps
        self._width = width
        self._height = height
        self._clip_duration = clip_duration_seconds
        self._fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer: cv2.VideoWriter | None = None
        self._clip_start_time: float = 0.0
        self._clip_number: int = 0
        self._current_path: str = ""

        os.makedirs(output_dir, exist_ok=True)
        self._start_new_clip()

    @property
    def current_path(self) -> str:
        """Path of the clip currently being written."""
        return self._current_path

    def write(self, frame: np.ndarray) -> None:
        """Write a frame; auto-rotate to a new clip when the duration is exceeded."""
        if time.time() - self._clip_start_time >= self._clip_duration:
            self._rotate_clip()
        if self._writer is not None:
            self._writer.write(frame)

    def release(self) -> None:
        """Flush and close the current clip."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # ── Internal ────────────────────────────────────────────────────

    def _start_new_clip(self) -> None:
        self._clip_number += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_path = os.path.join(
            self._output_dir,
            f"detection_{timestamp}_{self._clip_number:04d}.avi",
        )
        self._writer = cv2.VideoWriter(
            self._current_path,
            self._fourcc,
            self._fps,
            (self._width, self._height),
        )
        self._clip_start_time = time.time()
        print(f"Recording: {self._current_path}")

    def _rotate_clip(self) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"Saved clip: {self._current_path}")
        self._start_new_clip()
