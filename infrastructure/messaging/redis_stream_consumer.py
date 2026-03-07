"""Redis Streams consumer for person detection events.

Reads PersonLogEntry messages from a Redis Stream via blocking XREAD
and inserts them into a SQLite database through PersonLogDB.
"""

from __future__ import annotations

import base64
import logging
import threading
from typing import Any

from domain.detection.events import PersonLogEntry
from infrastructure.database.person_log_db import PersonLogDB

logger = logging.getLogger(__name__)


class RedisStreamConsumer:
    """Consumes person detections from a Redis Stream and writes to SQLite."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        stream_name: str = "person_detections",
        db: PersonLogDB | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._stream_name = stream_name
        self._db = db

        self._redis: Any = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_id = "0-0"  # start from the beginning of the stream

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the consume loop in a background thread."""
        import redis
        self._redis = redis.Redis(host=self._host, port=self._port, decode_responses=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._consume_loop, name="redis-consumer", daemon=True)
        self._thread.start()
        logger.info("Redis consumer started on %s:%s stream=%s", self._host, self._port, self._stream_name)

    def stop(self) -> None:
        """Signal the consume loop to stop and wait for the thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                logger.exception("Error closing Redis connection")
            self._redis = None
        logger.info("Redis consumer stopped")

    # ── Consume loop ──────────────────────────────────────────────────

    def _consume_loop(self) -> None:
        """Blocking XREAD loop that processes messages until stop is signalled."""
        while not self._stop_event.is_set():
            try:
                # Block for up to 1 second waiting for new messages
                result = self._redis.xread(
                    {self._stream_name: self._last_id},
                    count=10,
                    block=1000,
                )
                if not result:
                    continue

                for _stream_name, messages in result:
                    for msg_id, fields in messages:
                        try:
                            entry = self._deserialize(fields)
                            self._db.insert_detection(entry)
                            self._last_id = msg_id
                        except Exception:
                            logger.exception("Failed to process message %s", msg_id)
                            self._last_id = msg_id  # skip bad message

            except Exception:
                if not self._stop_event.is_set():
                    logger.exception("Error in Redis consumer loop")

    # ── Deserialization ───────────────────────────────────────────────

    @staticmethod
    def _deserialize(fields: dict[str, str]) -> PersonLogEntry:
        """Convert Redis hash fields back into a PersonLogEntry."""
        face_encoding_str = fields.get("face_encoding", "")
        face_encoding = base64.b64decode(face_encoding_str) if face_encoding_str else None

        person_name = fields.get("person_name", "")
        face_crop_path = fields.get("face_crop_path", "")
        body_crop_path = fields.get("body_crop_path", "")
        track_id_str = fields.get("track_id", "")

        return PersonLogEntry(
            detection_id=fields["detection_id"],
            timestamp=fields["timestamp"],
            camera_id=int(fields["camera_id"]),
            camera_label=fields["camera_label"],
            person_name=person_name if person_name else None,
            confidence=float(fields["confidence"]),
            face_crop_path=face_crop_path if face_crop_path else None,
            body_crop_path=body_crop_path if body_crop_path else None,
            face_encoding=face_encoding,
            track_id=int(track_id_str) if track_id_str else None,
        )
