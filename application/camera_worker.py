"""Reusable camera processing worker.

Encapsulates the frame-read -> detect -> match -> draw -> record loop
that was previously inline in interfaces/watch_stream.py.  Each
CameraWorker runs in its own thread and can be started/stopped
independently.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

import cv2

from application.detection_pipeline import DetectionPipeline
from application.stream_processor import draw_detections
from domain.detection.events import PersonDetection, FaceMatchResult
from infrastructure.camera.ffmpeg_nvdec_reader import FFmpegNvdecReader
from infrastructure.camera.opencv_frame_reader import OpenCVFrameReader
from infrastructure.camera.onvif_camera_source import OnvifCameraSource
from infrastructure.camera.rtsp_camera_source import RtspCameraSource
from infrastructure.camera.synology_camera_source import SynologyCameraSource
from infrastructure.recording.avi_clip_writer import AviClipWriter


class CameraWorker:
    """Self-contained camera processing loop.

    Parameters
    ----------
    camera_config : dict
        Camera definition with at least ``id``, ``label``, ``source_type``.
    detection_config : dict
        Detection settings (confidence_threshold, encodings_path, etc.).
    recording_config : dict
        Recording settings (enabled, output_dir, clip_duration).
    synology_config : dict | None
        Synology connection config (ip_address, port, username, ...).
    onvif_config : dict | None
        ONVIF connection config (ip, port, username, password).
    source_file : str | None
        Path to a video file (overrides live camera).
    loop_file : bool
        Whether to loop a video file when it ends.
    onvif_profile : int
        ONVIF media profile index.
    show_display : bool
        Whether to call cv2.imshow (set False for headless).
    on_detection : Callable | None
        Optional callback invoked with (camera_label, detections, match_results)
        after each frame that contains detections.
    """

    def __init__(
        self,
        camera_config: dict,
        detection_config: dict,
        recording_config: dict,
        synology_config: dict | None = None,
        onvif_config: dict | None = None,
        rtsp_config: dict | None = None,
        source_file: str | None = None,
        loop_file: bool = False,
        onvif_profile: int = 0,
        show_display: bool = True,
        on_detection: Callable | None = None,
        shared_detector=None,
    ) -> None:
        self._camera_config = camera_config
        self._detection_config = detection_config
        self._recording_config = recording_config
        self._synology_config = synology_config
        self._onvif_config = onvif_config
        self._rtsp_config = rtsp_config
        self._source_file = source_file
        self._loop_file = loop_file
        self._onvif_profile = onvif_profile
        self._show_display = show_display
        self._on_detection = on_detection
        self._shared_detector = shared_detector

        self._thread: threading.Thread | None = None
        self._running = False
        self._label = camera_config.get("label", f"camera_{camera_config.get('id', 0)}")

    @property
    def label(self) -> str:
        return self._label

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the processing loop on a background thread."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"worker-{self._label}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the loop to stop and wait for the thread to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def run_blocking(self) -> None:
        """Run the processing loop on the current thread (blocks until done)."""
        self._running = True
        self._run()

    # ── Internal ─────────────────────────────────────────────────────

    def _get_rtsp_url(self) -> str | None:
        """Try to obtain an RTSP URL from the camera source."""
        source_type = self._camera_config.get("source_type", "synology")
        camera_id = self._camera_config.get("id", 1)

        try:
            if source_type == "synology" and self._synology_config:
                source = SynologyCameraSource(self._synology_config)
                return source.get_rtsp_url(camera_id)
            elif source_type == "onvif" and self._onvif_config:
                source = OnvifCameraSource(self._onvif_config)
                return source.get_rtsp_url(profile_index=self._onvif_profile)
            elif source_type == "rtsp" and self._rtsp_config:
                source = RtspCameraSource(self._rtsp_config)
                return source.get_rtsp_url()
        except Exception as exc:
            print(f"[{self._label}] Could not get RTSP URL: {exc}")
        return None

    def _open_source(self):
        """Open and return (cap, is_file) for the video source."""
        if self._source_file is not None:
            print(f"[{self._label}] Opening video file: {self._source_file}")
            cap = cv2.VideoCapture(self._source_file)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {self._source_file}")
            return cap, True

        source_type = self._camera_config.get("source_type", "synology")
        camera_id = self._camera_config.get("id", 1)

        if source_type == "synology":
            if not self._synology_config:
                raise RuntimeError("Synology config required for synology source_type")
            source = SynologyCameraSource(self._synology_config)
            cap = source.open(camera_id)
        elif source_type == "onvif":
            if not self._onvif_config:
                raise RuntimeError("ONVIF config required for onvif source_type")
            source = OnvifCameraSource(self._onvif_config)
            cap = source.open(profile_index=self._onvif_profile)
        elif source_type == "rtsp":
            if not self._rtsp_config:
                raise RuntimeError("RTSP config required for rtsp source_type")
            source = RtspCameraSource(self._rtsp_config)
            cap = source.open()
        else:
            raise RuntimeError(f"Unknown source_type: {source_type}")

        return cap, False

    def _run(self) -> None:
        """Core processing loop."""
        det = self._detection_config
        rec = self._recording_config

        # Build detection pipeline
        print(f"[{self._label}] Initialising detection pipeline...")
        pipeline = DetectionPipeline(
            confidence_threshold=det.get("confidence_threshold", 0.5),
            encodings_path=det.get("encodings_path", "./faces-output/encodings.pkl"),
            match_tolerance=det.get("match_tolerance", 0.9),
            match_min_confidence=det.get("match_min_confidence", 0.5),
            match_skip_frames=det.get("match_skip_frames", 5),
            shared_detector=self._shared_detector,
        )
        pipeline.start()

        # Try FFmpeg NVDEC for live streams (hardware-accelerated H.264 decode)
        nvdec_reader = None
        cap = None
        is_file = False

        if self._source_file is None:
            rtsp_url = self._get_rtsp_url()
            if rtsp_url:
                # Probe stream dimensions with OpenCV briefly
                import shutil
                if shutil.which("ffmpeg"):
                    print(f"[{self._label}] Probing stream for dimensions...")
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                    probe_cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    if probe_cap.isOpened():
                        probe_w = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                        probe_h = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                        probe_fps = int(probe_cap.get(cv2.CAP_PROP_FPS)) or 15
                        probe_cap.release()

                        nvdec_reader = FFmpegNvdecReader(
                            rtsp_url=rtsp_url,
                            width=probe_w,
                            height=probe_h,
                        )
                        nvdec_reader.start()
                        print(
                            f"[{self._label}] FFmpeg {'NVDEC' if nvdec_reader.using_nvdec else 'software'} "
                            f"reader: {probe_w}x{probe_h} -> {nvdec_reader.width}x{nvdec_reader.height} @ 5fps"
                        )
                    else:
                        probe_cap.release()

        # Fall back to OpenCV if NVDEC reader wasn't created
        if nvdec_reader is None:
            try:
                cap, is_file = self._open_source()
            except RuntimeError as exc:
                print(f"[{self._label}] Error: {exc}")
                pipeline.stop()
                return

        # Video properties
        if nvdec_reader:
            width = nvdec_reader.width
            height = nvdec_reader.height
            fps = probe_fps
        else:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        source_desc = self._source_file if is_file else "stream"
        print(f"[{self._label}] Opened {source_desc}: {width}x{height} @ {fps}fps")

        # Background frame reader for OpenCV live streams
        reader = None
        if cap is not None and not is_file:
            reader = OpenCVFrameReader(cap).start()

        # Clip writer
        clip_writer = None
        if rec.get("enabled", True):
            output_dir = rec.get("output_dir", "./recordings")
            os.makedirs(output_dir, exist_ok=True)
            clip_writer = AviClipWriter(
                output_dir=output_dir,
                fps=fps,
                width=width,
                height=height,
                clip_duration_seconds=rec.get("clip_duration", 30),
            )

        print(f"[{self._label}] Starting detection loop...")
        frame_count = 0

        try:
            while self._running:
                # Read frame
                if nvdec_reader:
                    ret, frame = nvdec_reader.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue
                elif reader:
                    ret, frame = reader.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue
                else:
                    ret, frame = cap.read()
                    if not ret:
                        if is_file and self._loop_file:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        print(f"[{self._label}] End of video file.")
                        break

                frame_count += 1

                # Detect people (synchronous)
                detections = pipeline.process_frame(frame, frame_count)

                # Poll latest face-match results (non-blocking)
                match_results = pipeline.get_latest_matches()

                # Draw detections on frame
                annotated_frame = draw_detections(frame.copy(), detections, match_results)

                # Write to video file
                if clip_writer:
                    clip_writer.write(annotated_frame)

                # Fire detection callback
                if detections and self._on_detection:
                    self._on_detection(self._label, detections, match_results)

                # Log detections
                if detections:
                    names = [
                        mr.person_name
                        for mr in match_results
                        if mr.matched
                    ]
                    summary = f"{len(detections)} person(s)"
                    if names:
                        summary += f" [{', '.join(names)}]"
                    print(f"[{self._label}] Frame {frame_count}: {summary}")

                # Display frame (optional)
                if self._show_display:
                    cv2.imshow(f"Detection - {self._label}", annotated_frame)
                    wait_ms = max(1, int(1000 / fps)) if is_file else 1
                    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print(f"\n[{self._label}] Stopping...")

        finally:
            if nvdec_reader:
                nvdec_reader.stop()
            if reader:
                reader.stop()
            pipeline.stop()
            if clip_writer:
                clip_writer.release()
            if cap is not None:
                cap.release()
            if self._show_display:
                cv2.destroyAllWindows()
            self._running = False
            print(f"[{self._label}] Done!")
