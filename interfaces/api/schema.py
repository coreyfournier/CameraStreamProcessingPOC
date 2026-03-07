"""Strawberry GraphQL schema for person detection data."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import strawberry
from strawberry.types import Info

from infrastructure.database.person_log_db import PersonLogDB


# ── Helper ────────────────────────────────────────────────────────────


def _path_to_url(path: str | None) -> str | None:
    """Convert an absolute or relative file path to an image URL path."""
    if not path:
        return None
    # Normalise to forward slashes
    path = path.replace("\\", "/")
    # Strip any leading "./" prefix
    if path.startswith("./"):
        path = path[2:]
    return f"/images/{path}"


def _get_db(info: Info) -> PersonLogDB:
    return info.context["db"]


# ── Types ─────────────────────────────────────────────────────────────


@strawberry.type
class PersonType:
    name: str
    detection_count: int
    last_seen: Optional[datetime]
    first_seen: Optional[datetime]


@strawberry.type
class DetectionType:
    detection_id: str
    timestamp: datetime
    camera_id: int
    camera_label: str
    person_name: Optional[str]
    confidence: float
    face_crop_url: Optional[str]
    body_crop_url: Optional[str]
    cluster_id: Optional[int]
    track_id: Optional[int]


@strawberry.type
class UnknownClusterType:
    cluster_id: int
    detection_count: int
    suggested_name: Optional[str]
    confirmed: bool


@strawberry.type
class CameraType:
    camera_id: int
    label: str


# ── Row converters ────────────────────────────────────────────────────


def _row_to_detection(row: dict) -> DetectionType:
    ts = row["timestamp"]
    if isinstance(ts, str):
        # Handle fractional-seconds ISO format from SQLite
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")) if "Z" in ts else datetime.fromisoformat(ts)
                break
            except ValueError:
                continue
    return DetectionType(
        detection_id=row["detection_id"],
        timestamp=ts,
        camera_id=row["camera_id"],
        camera_label=row["camera_label"],
        person_name=row.get("person_name"),
        confidence=row["confidence"],
        face_crop_url=_path_to_url(row.get("face_crop_path")),
        body_crop_url=_path_to_url(row.get("body_crop_path")),
        cluster_id=row.get("cluster_id"),
        track_id=row.get("track_id"),
    )


def _row_to_cluster(row: dict) -> UnknownClusterType:
    return UnknownClusterType(
        cluster_id=row["cluster_id"],
        detection_count=row.get("detection_count", 0),
        suggested_name=row.get("suggested_name"),
        confirmed=bool(row.get("confirmed", 0)),
    )


# ── Query ─────────────────────────────────────────────────────────────


@strawberry.type
class Query:
    @strawberry.field
    def persons(
        self,
        info: Info,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PersonType]:
        db = _get_db(info)
        rows = db._conn.execute(
            """
            SELECT person_name,
                   COUNT(*) AS detection_count,
                   MAX(timestamp) AS last_seen,
                   MIN(timestamp) AS first_seen
            FROM person_detections
            WHERE person_name IS NOT NULL
            GROUP BY person_name
            ORDER BY last_seen DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        results: list[PersonType] = []
        for r in rows:
            row = dict(r)
            last_seen = row["last_seen"]
            first_seen = row["first_seen"]
            if isinstance(last_seen, str):
                last_seen = datetime.fromisoformat(last_seen.replace("Z", "+00:00")) if last_seen else None
            if isinstance(first_seen, str):
                first_seen = datetime.fromisoformat(first_seen.replace("Z", "+00:00")) if first_seen else None
            results.append(PersonType(
                name=row["person_name"],
                detection_count=row["detection_count"],
                last_seen=last_seen,
                first_seen=first_seen,
            ))
        return results

    @strawberry.field
    def detections(
        self,
        info: Info,
        person_name: Optional[str] = None,
        camera_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DetectionType]:
        db = _get_db(info)
        rows = db.get_detections(
            person_name=person_name,
            camera_id=camera_id,
            start=start_date,
            end=end_date,
            limit=limit,
            offset=offset,
        )
        return [_row_to_detection(r) for r in rows]

    @strawberry.field
    def detection(self, info: Info, detection_id: str) -> Optional[DetectionType]:
        db = _get_db(info)
        row = db.get_detection_by_id(detection_id)
        if row is None:
            return None
        return _row_to_detection(row)

    @strawberry.field
    def unknown_clusters(self, info: Info) -> list[UnknownClusterType]:
        db = _get_db(info)
        rows = db.get_clusters()
        return [_row_to_cluster(r) for r in rows]

    @strawberry.field
    def cluster_detections(self, info: Info, cluster_id: int) -> list[DetectionType]:
        db = _get_db(info)
        rows = db.get_cluster_detections(cluster_id)
        return [_row_to_detection(r) for r in rows]

    @strawberry.field
    def recent_activity(self, info: Info, limit: int = 20) -> list[DetectionType]:
        db = _get_db(info)
        rows = db.get_recent_activity(limit=limit)
        return [_row_to_detection(r) for r in rows]

    @strawberry.field
    def cameras(self, info: Info) -> list[CameraType]:
        db = _get_db(info)
        rows = db._conn.execute(
            """
            SELECT DISTINCT camera_id, camera_label
            FROM person_detections
            ORDER BY camera_label
            """
        ).fetchall()
        return [CameraType(camera_id=r["camera_id"], label=r["camera_label"]) for r in rows]


# ── Mutation ──────────────────────────────────────────────────────────


@strawberry.type
class Mutation:
    @strawberry.mutation
    def update_person_name(self, info: Info, detection_id: str, name: str) -> DetectionType:
        db = _get_db(info)
        db.update_person_name(detection_id, name)
        row = db.get_detection_by_id(detection_id)
        if row is None:
            raise ValueError(f"Detection {detection_id} not found")
        return _row_to_detection(row)

    @strawberry.mutation
    def confirm_cluster(self, info: Info, cluster_id: int, person_name: str) -> UnknownClusterType:
        db = _get_db(info)
        db.confirm_cluster(cluster_id, person_name)
        # Fetch the updated cluster
        row = db._conn.execute(
            "SELECT * FROM unknown_clusters WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Cluster {cluster_id} not found")
        return _row_to_cluster(dict(row))

    @strawberry.mutation
    def reject_cluster_suggestion(self, info: Info, cluster_id: int) -> UnknownClusterType:
        db = _get_db(info)
        db._conn.execute(
            "UPDATE unknown_clusters SET suggested_name = NULL WHERE cluster_id = ?",
            (cluster_id,),
        )
        db._conn.commit()
        row = db._conn.execute(
            "SELECT * FROM unknown_clusters WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Cluster {cluster_id} not found")
        return _row_to_cluster(dict(row))

    @strawberry.mutation
    def delete_detection(self, info: Info, detection_id: str) -> bool:
        db = _get_db(info)
        row = db.get_detection_by_id(detection_id)
        if row is None:
            return False
        db.delete_detection(detection_id)
        return True


# ── Schema ────────────────────────────────────────────────────────────

schema = strawberry.Schema(query=Query, mutation=Mutation)
