"""Domain event dataclasses for the detection pipeline.

Provides value objects for passing data between components.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# ── Event data classes ──────────────────────────────────────────────


@dataclass
class FrameContext:
    """Metadata about the frame being processed."""
    frame: np.ndarray
    frame_number: int
    timestamp: float


@dataclass
class PersonDetection:
    """A single person detected in a frame."""
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    person_crop: np.ndarray         # BGR crop of the person bounding box


@dataclass
class PersonDetectionEvent:
    """Emitted when one or more people are detected in a frame."""
    context: FrameContext
    detections: list[PersonDetection]


@dataclass
class FaceMatchResult:
    """Result of attempting to match a detected person to a known face."""
    person_detection: PersonDetection
    matched: bool
    person_name: str       # Name if matched, "Unknown" otherwise
    confidence: float      # Normalised confidence: 1.0 = perfect, 0.0 = at tolerance boundary
    face_location: tuple | None  # (top, right, bottom, left) within the crop, or None
    face_encoding: bytes | None = None  # serialised 512-d float32 embedding


@dataclass
class SmoothedIdentity:
    """A temporally smoothed identity for a tracked person."""
    person_detection: PersonDetection
    person_name: str        # smoothed identity or "Unknown"
    confidence: float       # smoothed confidence (avg of top-name entries)
    is_smoothed: bool       # True if identity came from temporal smoothing
    track_id: int


@dataclass
class FaceMatchEvent:
    """Emitted after face matching completes for a set of detections."""
    context: FrameContext
    results: list[FaceMatchResult]


@dataclass
class PersonLogEntry:
    """A single person detection record for persistent storage."""
    detection_id: str           # UUID
    timestamp: str              # ISO 8601
    camera_id: int
    camera_label: str
    person_name: str | None     # None if unidentified
    confidence: float
    face_crop_path: str | None
    body_crop_path: str | None
    face_encoding: bytes | None  # serialized 512-d float32 for clustering
    track_id: int | None
