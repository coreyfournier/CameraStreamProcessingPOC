"""Redis Streams producer for person detection events.

Publishes PersonLogEntry records to a Redis Stream via XADD.
Designed to be resilient — if Redis is unavailable the error is logged
and the camera worker continues uninterrupted.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from domain.detection.events import PersonLogEntry

logger = logging.getLogger(__name__)


class RedisStreamProducer:
    """Publishes person detections to a Redis Stream."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        stream_name: str = "person_detections",
    ) -> None:
        self._host = host
        self._port = port
        self._stream_name = stream_name
        self._redis: Any = None  # lazy connection

    # ── Connection ────────────────────────────────────────────────────

    def _ensure_connected(self) -> Any:
        """Lazily create the Redis connection on first use."""
        if self._redis is None:
            import redis
            self._redis = redis.Redis(host=self._host, port=self._port, decode_responses=True)
        return self._redis

    # ── Publish ───────────────────────────────────────────────────────

    def publish(self, entry: PersonLogEntry) -> None:
        """Serialize a PersonLogEntry and XADD it to the stream.

        Binary face_encoding is base64-encoded; all other fields are stored
        as strings.  If Redis is unreachable the error is logged and the
        call returns silently.
        """
        try:
            r = self._ensure_connected()
            fields: dict[str, str] = {
                "detection_id": entry.detection_id,
                "timestamp": entry.timestamp,
                "camera_id": str(entry.camera_id),
                "camera_label": entry.camera_label,
                "person_name": entry.person_name if entry.person_name is not None else "",
                "confidence": str(entry.confidence),
                "face_crop_path": entry.face_crop_path if entry.face_crop_path is not None else "",
                "body_crop_path": entry.body_crop_path if entry.body_crop_path is not None else "",
                "face_encoding": base64.b64encode(entry.face_encoding).decode("ascii") if entry.face_encoding is not None else "",
                "track_id": str(entry.track_id) if entry.track_id is not None else "",
            }
            r.xadd(self._stream_name, fields)
        except Exception:
            logger.exception("Failed to publish detection %s to Redis", entry.detection_id)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the Redis connection if open."""
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                logger.exception("Error closing Redis connection")
            self._redis = None
