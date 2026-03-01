"""Domain event dataclasses for the detection pipeline.

Provides value objects for passing data between components.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    confidence: float      # Distance-based confidence (lower distance = higher confidence)
    face_location: tuple | None  # (top, right, bottom, left) within the crop, or None


@dataclass
class FaceMatchEvent:
    """Emitted after face matching completes for a set of detections."""
    context: FrameContext
    results: list[FaceMatchResult]
