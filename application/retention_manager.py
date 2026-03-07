"""Periodic cleanup of old person detection records and associated images."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta

from infrastructure.database.person_log_db import PersonLogDB
from infrastructure.storage.person_image_storage import PersonImageStorage

logger = logging.getLogger(__name__)


class RetentionManager:
    """Deletes person detection records and images older than a configurable threshold."""

    def __init__(
        self,
        db: PersonLogDB,
        image_storage: PersonImageStorage,
        retention_days: int = 30,
    ) -> None:
        self._db = db
        self._image_storage = image_storage
        self._retention_days = retention_days
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── One-shot cleanup ──────────────────────────────────────────────

    def cleanup(self) -> dict:
        """Run a single cleanup pass and return summary statistics."""
        cutoff_dt = datetime.now() - timedelta(days=self._retention_days)
        cutoff_iso = cutoff_dt.isoformat()

        logger.info(
            "Retention cleanup starting: removing records older than %s (%d days)",
            cutoff_iso,
            self._retention_days,
        )

        # Images must be deleted BEFORE db records (db records contain the file paths)
        self._image_storage.delete_before(cutoff_iso)
        deleted_count = self._db.delete_before(cutoff_iso)

        logger.info(
            "Retention cleanup complete: deleted %d record(s) with cutoff %s",
            deleted_count,
            cutoff_iso,
        )

        return {"deleted_count": deleted_count, "cutoff_date": cutoff_iso}

    # ── Scheduled background cleanup ──────────────────────────────────

    def start_scheduled(self, interval_hours: float = 24) -> None:
        """Start a background thread that runs cleanup periodically."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Scheduled retention cleanup is already running")
            return

        self._stop_event.clear()
        interval_seconds = interval_hours * 3600

        def _loop() -> None:
            logger.info(
                "Retention scheduler started: every %.1f hour(s), keeping %d days",
                interval_hours,
                self._retention_days,
            )
            while not self._stop_event.is_set():
                try:
                    self.cleanup()
                except Exception:
                    logger.exception("Retention cleanup failed")

                # Sleep in small increments so stop() is responsive
                self._stop_event.wait(timeout=interval_seconds)

        self._thread = threading.Thread(
            target=_loop, name="retention-cleanup", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background cleanup thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
            logger.info("Retention scheduler stopped")
