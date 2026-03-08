"""Multi-camera orchestrator.

Creates and manages a fleet of CameraWorker instances, one per camera
defined in the configuration file.  Shared infrastructure (Redis producer,
image storage, identity smoother per camera) is wired up here so that each
worker's ``on_detection`` callback saves images, builds a PersonLogEntry,
and publishes to Redis.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import numpy as np

from application.camera_worker import CameraWorker
from application.detection_smoother import IdentitySmoother
from domain.detection.events import (
    FaceMatchResult,
    PersonDetection,
    PersonLogEntry,
)
from infrastructure.config import (
    get_camera_configs,
    get_detection_config,
    get_onvif_config,
    get_recording_config,
    get_redis_config,
    get_smoothing_config,
    get_storage_config,
    get_synology_config,
)
from infrastructure.messaging.redis_stream_producer import RedisStreamProducer
from infrastructure.storage.person_image_storage import PersonImageStorage

logger = logging.getLogger(__name__)


class MultiCameraOrchestrator:
    """Starts, manages, and stops multiple CameraWorker threads.

    Parameters
    ----------
    config : dict
        Full configuration dictionary as returned by ``load_config()``.
    """

    def __init__(self, config: dict) -> None:
        self._config = config

        # Extract sub-configs
        self._camera_configs = get_camera_configs(config)
        self._detection_config = get_detection_config(config)
        self._recording_config = get_recording_config(config)
        self._smoothing_config = get_smoothing_config(config)
        self._redis_config = get_redis_config(config)
        self._storage_config = get_storage_config(config)
        self._synology_config = get_synology_config(config)
        self._onvif_config = get_onvif_config(config)

        # Shared infrastructure (thread-safe)
        self._redis_producer = RedisStreamProducer(
            host=self._redis_config.get("host", "localhost"),
            port=int(self._redis_config.get("port", 6379)),
            stream_name=self._redis_config.get("stream_name", "person_detections"),
        )
        self._image_storage = PersonImageStorage(
            base_dir=self._storage_config.get("person_images_dir", "./recordings/persons"),
        )

        # Shared YOLO detector (single model on GPU, used by all cameras)
        from infrastructure.detection.yolo_person_detector import PersonDetector
        self._shared_detector = PersonDetector(
            confidence_threshold=self._detection_config.get("confidence_threshold", 0.5),
        )
        logger.info("Shared PersonDetector using device: %s", self._shared_detector.device)

        # Build workers
        self._workers: list[CameraWorker] = []
        self._smoothers: dict[str, IdentitySmoother] = {}
        self._build_workers()

    # ── Worker construction ───────────────────────────────────────────

    def _build_workers(self) -> None:
        """Create a CameraWorker for each camera entry in the config."""
        for cam_cfg in self._camera_configs:
            label = cam_cfg.get("label", f"camera_{cam_cfg.get('id', 0)}")
            source_type = cam_cfg.get("source_type", "synology")

            # Per-camera smoother (stateful, not shared)
            smoother = IdentitySmoother(
                window_size=self._smoothing_config.get("window_size", 10),
                min_hit_ratio=self._smoothing_config.get("min_hit_ratio", 0.7),
                min_avg_confidence=self._smoothing_config.get("min_avg_confidence", 0.7),
            )
            self._smoothers[label] = smoother

            # Resolve per-camera connection config.  A camera may override
            # the top-level onvif/synology section with its own nested block.
            synology_cfg = self._synology_config
            onvif_cfg = self._onvif_config

            if source_type == "onvif" and "onvif" in cam_cfg:
                per_cam = cam_cfg["onvif"]
                onvif_cfg = {
                    "ip": per_cam.get("ip", onvif_cfg.get("ip", "")),
                    "port": per_cam.get("port", onvif_cfg.get("port", "80")),
                    "username": per_cam.get("username", onvif_cfg.get("username", "")),
                    "password": per_cam.get("password", onvif_cfg.get("password", "")),
                }

            if source_type == "synology" and "synology" in cam_cfg:
                per_cam = cam_cfg["synology"]
                synology_cfg = {
                    **synology_cfg,
                    **{k: v for k, v in per_cam.items() if v is not None},
                }

            onvif_profile = cam_cfg.get("onvif_profile", 0)
            source_file = cam_cfg.get("source", None)
            loop_file = cam_cfg.get("loop", False)

            # Capture loop variables for the closure
            _label = label
            _cam_cfg = cam_cfg
            _smoother = smoother

            def make_callback(
                lbl: str, cfg: dict, sm: IdentitySmoother
            ):
                """Return an on_detection callback bound to a specific camera."""
                def _on_detection(
                    camera_label: str,
                    detections: list[PersonDetection],
                    match_results: list[FaceMatchResult],
                ) -> None:
                    self._handle_detections(lbl, cfg, sm, detections, match_results)
                return _on_detection

            worker = CameraWorker(
                camera_config=cam_cfg,
                detection_config=self._detection_config,
                recording_config=self._recording_config,
                synology_config=synology_cfg,
                onvif_config=onvif_cfg,
                source_file=source_file,
                loop_file=loop_file,
                onvif_profile=onvif_profile,
                show_display=False,
                on_detection=make_callback(_label, _cam_cfg, _smoother),
                shared_detector=self._shared_detector,
            )
            self._workers.append(worker)

    # ── Detection callback ────────────────────────────────────────────

    def _handle_detections(
        self,
        camera_label: str,
        camera_config: dict,
        smoother: IdentitySmoother,
        detections: list[PersonDetection],
        match_results: list[FaceMatchResult],
    ) -> None:
        """Called from each CameraWorker thread when detections occur.

        1. Smooth identities via IdentitySmoother.
        2. Save face/body crops via PersonImageStorage.
        3. Build PersonLogEntry and publish to Redis.
        """
        smoothed_list = smoother.smooth(detections, match_results)
        now_iso = datetime.now().isoformat()
        camera_id = camera_config.get("id", 0)

        for si in smoothed_list:
            detection_id = str(uuid.uuid4())

            # Extract face crop and encoding from the match result if available
            face_crop = self._extract_face_crop(si.person_detection, match_results)
            face_encoding = self._extract_face_encoding(si.person_detection, match_results)
            body_crop = si.person_detection.person_crop

            # Save images (thread-safe — unique detection_id per call)
            face_path, body_path = self._image_storage.save(
                detection_id=detection_id,
                face_crop=face_crop,
                body_crop=body_crop,
                timestamp=now_iso,
            )

            entry = PersonLogEntry(
                detection_id=detection_id,
                timestamp=now_iso,
                camera_id=camera_id,
                camera_label=camera_label,
                person_name=si.person_name if si.person_name != "Unknown" else None,
                confidence=si.confidence,
                face_crop_path=face_path,
                body_crop_path=body_path,
                face_encoding=face_encoding,
                track_id=si.track_id,
            )

            self._redis_producer.publish(entry)

    @staticmethod
    def _find_closest_match(
        detection: PersonDetection,
        match_results: list[FaceMatchResult],
    ) -> FaceMatchResult | None:
        """Find the closest FaceMatchResult by bounding-box center distance."""
        if not match_results:
            return None
        best: FaceMatchResult | None = None
        best_dist = float("inf")
        dx1, dy1, dx2, dy2 = detection.box
        cx = (dx1 + dx2) * 0.5
        cy = (dy1 + dy2) * 0.5
        for mr in match_results:
            mx1, my1, mx2, my2 = mr.person_detection.box
            mx = (mx1 + mx2) * 0.5
            my = (my1 + my2) * 0.5
            d = (cx - mx) ** 2 + (cy - my) ** 2
            if d < best_dist:
                best_dist = d
                best = mr
        return best

    @staticmethod
    def _extract_face_crop(
        detection: PersonDetection,
        match_results: list[FaceMatchResult],
    ) -> np.ndarray | None:
        """Try to extract a face crop from matching FaceMatchResults."""
        mr = MultiCameraOrchestrator._find_closest_match(detection, match_results)
        if mr is None or mr.face_location is None:
            return None
        top, right, bottom, left = mr.face_location
        crop = detection.person_crop
        h, w = crop.shape[:2]
        t = max(0, top)
        b = min(h, bottom)
        l = max(0, left)
        r = min(w, right)
        if b > t and r > l:
            return crop[t:b, l:r]
        return None

    @staticmethod
    def _extract_face_encoding(
        detection: PersonDetection,
        match_results: list[FaceMatchResult],
    ) -> bytes | None:
        """Extract the face encoding from the closest FaceMatchResult."""
        mr = MultiCameraOrchestrator._find_closest_match(detection, match_results)
        if mr is None:
            return None
        return mr.face_encoding

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all camera workers.  Logs which cameras succeed or fail."""
        logger.info("Starting %d camera worker(s)...", len(self._workers))

        for worker in self._workers:
            try:
                worker.start()
                logger.info("Started camera: %s", worker.label)
            except Exception:
                logger.exception("Failed to start camera: %s", worker.label)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop all camera workers gracefully."""
        logger.info("Stopping %d camera worker(s)...", len(self._workers))

        for worker in self._workers:
            try:
                worker.stop(timeout=timeout)
                logger.info("Stopped camera: %s", worker.label)
            except Exception:
                logger.exception("Error stopping camera: %s", worker.label)

        self._redis_producer.close()
        logger.info("All cameras stopped.")
