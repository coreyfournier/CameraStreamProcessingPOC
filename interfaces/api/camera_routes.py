"""REST endpoints for camera management and live MJPEG streaming."""

from __future__ import annotations

import os
import threading
import time
from typing import Any
from urllib.parse import quote

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api")

# Module-level reference to camera configs (set by server.py)
_camera_configs: list[dict] = []
_rtsp_configs: dict[int, dict] = {}  # camera_id -> rtsp connection info
_synology_config: dict = {}
_onvif_config: dict = {}


def configure(
    camera_configs: list[dict],
    synology_config: dict,
    onvif_config: dict,
    rtsp_config: dict,
) -> None:
    """Set camera configurations from the app factory."""
    global _camera_configs, _synology_config, _onvif_config
    _camera_configs = camera_configs
    _synology_config = synology_config
    _onvif_config = onvif_config

    # Build per-camera RTSP connection info
    for cam in camera_configs:
        cam_id = cam.get("id", 0)
        source_type = cam.get("source_type", "synology")

        if source_type == "rtsp":
            per_cam = cam.get("rtsp", {})
            _rtsp_configs[cam_id] = {
                "ip": per_cam.get("ip", rtsp_config.get("ip", "")),
                "port": per_cam.get("port", rtsp_config.get("port", "554")),
                "username": per_cam.get("username", rtsp_config.get("username", "")),
                "password": per_cam.get("password", rtsp_config.get("password", "")),
                "path": per_cam.get("path", rtsp_config.get("path", "")),
            }


def _build_rtsp_url(cfg: dict) -> str:
    """Build an RTSP URL from a connection config dict."""
    ip = cfg.get("ip", "")
    port = int(cfg.get("port", 554))
    username = cfg.get("username", "")
    password = cfg.get("password", "")
    path = cfg.get("path", "")

    creds = ""
    if username:
        creds = f"{quote(username, safe='')}:{quote(password, safe='')}@"

    return f"rtsp://{creds}{ip}:{port}{path}"


def _get_rtsp_url_for_camera(camera_id: int) -> str | None:
    """Resolve the RTSP URL for a given camera ID."""
    if camera_id in _rtsp_configs:
        return _build_rtsp_url(_rtsp_configs[camera_id])

    # For synology cameras, try to get the URL via the source
    cam = next((c for c in _camera_configs if c.get("id") == camera_id), None)
    if cam is None:
        return None

    source_type = cam.get("source_type", "synology")

    if source_type == "synology" and _synology_config:
        try:
            from infrastructure.camera.synology_camera_source import SynologyCameraSource
            source = SynologyCameraSource(_synology_config)
            return source.get_rtsp_url(camera_id)
        except Exception:
            return None

    if source_type == "onvif" and _onvif_config:
        try:
            from infrastructure.camera.onvif_camera_source import OnvifCameraSource
            source = OnvifCameraSource(_onvif_config)
            return source.get_rtsp_url()
        except Exception:
            return None

    return None


@router.get("/cameras")
def list_cameras() -> list[dict[str, Any]]:
    """Return all configured cameras with their connection info."""
    result = []
    for cam in _camera_configs:
        result.append({
            "id": cam.get("id", 0),
            "label": cam.get("label", f"camera_{cam.get('id', 0)}"),
            "source_type": cam.get("source_type", "synology"),
            "stream_url": f"/api/stream/{cam.get('id', 0)}",
        })
    return result


def _mjpeg_generator(camera_id: int):
    """Yield MJPEG frames from an RTSP stream."""
    rtsp_url = _get_rtsp_url_for_camera(camera_id)
    if not rtsp_url:
        return

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    target_fps = 5
    frame_interval = 1.0 / target_fps

    try:
        while True:
            start = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                break

            # Downscale for preview (max 640px wide)
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

            elapsed = time.monotonic() - start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        cap.release()


@router.get("/stream/{camera_id}")
def stream_camera(camera_id: int):
    """MJPEG stream proxy for a configured camera."""
    cam = next((c for c in _camera_configs if c.get("id") == camera_id), None)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    return StreamingResponse(
        _mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
