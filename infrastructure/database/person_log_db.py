"""SQLite persistence for person detection logs and unknown-person clusters."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from domain.detection.events import PersonLogEntry

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class PersonLogDB:
    """Lightweight SQLite wrapper for the person detection log."""

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._apply_schema()

    # ── Schema bootstrap ─────────────────────────────────────────────

    def _apply_schema(self) -> None:
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        self._conn.executescript(schema_sql)

    # ── Insert ───────────────────────────────────────────────────────

    def insert_detection(self, entry: PersonLogEntry) -> None:
        self._conn.execute(
            """
            INSERT INTO person_detections
                (detection_id, timestamp, camera_id, camera_label,
                 person_name, confidence, face_crop_path, body_crop_path,
                 face_encoding, track_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.detection_id,
                entry.timestamp,
                entry.camera_id,
                entry.camera_label,
                entry.person_name,
                entry.confidence,
                entry.face_crop_path,
                entry.body_crop_path,
                entry.face_encoding,
                entry.track_id,
            ),
        )
        self._conn.commit()

    # ── Queries ──────────────────────────────────────────────────────

    def get_detections(
        self,
        person_name: str | None = None,
        camera_id: int | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list = []
        if person_name is not None:
            clauses.append("person_name = ?")
            params.append(person_name)
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT * FROM person_detections
            {where}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_detection_by_id(self, detection_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM person_detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_person_names(self) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT DISTINCT person_name FROM person_detections
            WHERE person_name IS NOT NULL
            ORDER BY person_name
            """
        ).fetchall()
        return [r["person_name"] for r in rows]

    def get_recent_activity(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM person_detections ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Updates ──────────────────────────────────────────────────────

    def update_person_name(self, detection_id: str, name: str) -> None:
        self._conn.execute(
            "UPDATE person_detections SET person_name = ? WHERE detection_id = ?",
            (name, detection_id),
        )
        self._conn.commit()

    # ── Deletes ──────────────────────────────────────────────────────

    def delete_detection(self, detection_id: str) -> None:
        self._conn.execute(
            "DELETE FROM person_detections WHERE detection_id = ?",
            (detection_id,),
        )
        self._conn.commit()

    def delete_before(self, cutoff_iso: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM person_detections WHERE timestamp < ?",
            (cutoff_iso,),
        )
        self._conn.commit()
        return cur.rowcount

    # ── Clustering helpers ───────────────────────────────────────────

    def get_unidentified_with_encodings(self) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT * FROM person_detections
            WHERE person_name IS NULL
              AND face_encoding IS NOT NULL
            ORDER BY timestamp DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def assign_cluster(self, detection_id: str, cluster_id: int) -> None:
        self._conn.execute(
            "UPDATE person_detections SET cluster_id = ? WHERE detection_id = ?",
            (cluster_id, detection_id),
        )
        self._conn.commit()

    def create_cluster(self, representative_encoding: bytes) -> int:
        cur = self._conn.execute(
            "INSERT INTO unknown_clusters (representative_encoding) VALUES (?)",
            (representative_encoding,),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_clusters(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM unknown_clusters ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_cluster_suggestion(self, cluster_id: int, suggested_name: str) -> None:
        self._conn.execute(
            "UPDATE unknown_clusters SET suggested_name = ? WHERE cluster_id = ?",
            (suggested_name, cluster_id),
        )
        self._conn.commit()

    def confirm_cluster(self, cluster_id: int, person_name: str) -> None:
        self._conn.execute(
            "UPDATE unknown_clusters SET confirmed = 1, suggested_name = ? WHERE cluster_id = ?",
            (person_name, cluster_id),
        )
        self._conn.execute(
            "UPDATE person_detections SET person_name = ? WHERE cluster_id = ?",
            (person_name, cluster_id),
        )
        self._conn.commit()

    def get_cluster_detections(self, cluster_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM person_detections WHERE cluster_id = ? ORDER BY timestamp DESC",
            (cluster_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()
