"""Hardware-accelerated frame reader using FFmpeg with NVDEC.

Spawns an FFmpeg subprocess that decodes an RTSP stream using the GPU's
dedicated NVDEC hardware decoder, optionally scales on GPU via scale_cuda,
and pipes raw BGR24 frames to this process.

Falls back to software decoding if NVDEC is not available.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Maximum output resolution for detection — YOLO uses 640x640 internally,
# so anything larger is wasted CPU moving pixels through the pipe.
MAX_OUTPUT_WIDTH = 1280


def _nvdec_available() -> bool:
    """Check if FFmpeg reports cuda/cuvid hardware acceleration."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hwaccels"],
            capture_output=True, text=True, timeout=5,
        )
        return "cuda" in result.stdout or "cuvid" in result.stdout
    except Exception:
        return False


class FFmpegNvdecReader:
    """Read RTSP frames via FFmpeg with NVDEC hardware decoding.

    Falls back to FFmpeg software decoding if NVDEC is unavailable,
    and returns None if FFmpeg is not installed at all.

    Parameters
    ----------
    rtsp_url : str
        RTSP stream URL.
    width : int
        Native stream width (used to compute scale ratio).
    height : int
        Native stream height.
    max_output_width : int
        Frames wider than this are downscaled (on GPU when using NVDEC).
        Set to 0 to disable scaling.
    target_fps : int
        Drop frames to this rate.  0 = no frame dropping.
    """

    def __init__(
        self,
        rtsp_url: str,
        width: int = 1920,
        height: int = 1080,
        max_output_width: int = MAX_OUTPUT_WIDTH,
        target_fps: int = 5,
    ) -> None:
        self._url = rtsp_url
        self._use_nvdec = _nvdec_available()
        self._target_fps = target_fps

        # Compute output dimensions (scale down if wider than max)
        if max_output_width and width > max_output_width:
            scale = max_output_width / width
            self._width = max_output_width
            # height must be divisible by 2 for most codecs
            self._height = int(height * scale) // 2 * 2
        else:
            self._width = width
            self._height = height

        self._native_width = width
        self._native_height = height
        self._frame_bytes = self._width * self._height * 3  # BGR24

        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._frame: np.ndarray | None = None
        self._ret = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

        if self._use_nvdec:
            logger.info("FFmpeg NVDEC hardware decoding available")
        else:
            logger.info("NVDEC not available, using FFmpeg software decoding")

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def using_nvdec(self) -> bool:
        return self._use_nvdec

    def start(self) -> "FFmpegNvdecReader":
        """Start the FFmpeg subprocess and reader thread."""
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        needs_scale = (self._width != self._native_width)

        if self._use_nvdec:
            # Decode on GPU, optionally scale on GPU, then download to system memory
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                    "-c:v", "h264_cuvid"]

        cmd += [
            "-rtsp_transport", "tcp",
            "-i", self._url,
        ]

        # Build filter chain
        filters = []

        if self._use_nvdec:
            if needs_scale:
                filters.append(f"scale_cuda={self._width}:{self._height}")
            filters.append("hwdownload")
            filters.append("format=nv12")
        else:
            if needs_scale:
                filters.append(f"scale={self._width}:{self._height}")

        if self._target_fps > 0:
            filters.append(f"fps={self._target_fps}")

        if filters:
            cmd += ["-vf", ",".join(filters)]

        cmd += [
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-sn",
            "pipe:1",
        ]

        scale_info = f" scaled {self._native_width}x{self._native_height} -> {self._width}x{self._height}" if needs_scale else ""
        fps_info = f" @ {self._target_fps}fps" if self._target_fps > 0 else ""
        logger.info(
            "Starting FFmpeg (%s%s%s)",
            "NVDEC" if self._use_nvdec else "CPU",
            scale_info,
            fps_info,
        )

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_bytes * 2,
        )

        self._running = True
        self._thread = threading.Thread(
            target=self._reader, name="ffmpeg-nvdec-reader", daemon=True
        )
        self._thread.start()
        return self

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None]:
        """Block until a new frame arrives or timeout expires.

        Returns (ret, frame).  Returns (False, None) on timeout.
        """
        if not self._new_frame.wait(timeout=timeout):
            return False, None
        self._new_frame.clear()
        with self._lock:
            return self._ret, self._frame

    def stop(self) -> None:
        """Terminate the FFmpeg process and reader thread."""
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _reader(self) -> None:
        """Background loop: read raw frames from FFmpeg stdout."""
        while self._running and self._proc is not None:
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) != self._frame_bytes:
                with self._lock:
                    self._ret = False
                    self._frame = None
                if self._running:
                    stderr = self._proc.stderr.read()
                    if stderr:
                        logger.error("FFmpeg error: %s", stderr.decode(errors="replace")[:500])
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self._height, self._width, 3)
            )
            with self._lock:
                self._ret = True
                self._frame = frame
            self._new_frame.set()
