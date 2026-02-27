"""Event infrastructure for the detection pipeline.

Provides dataclasses for passing data between components and a simple
EventEmitter for decoupled observer/listener communication.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

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


# ── EventEmitter ────────────────────────────────────────────────────


class EventEmitter:
    """Simple thread-safe event emitter with on/off/emit."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()

    def on(self, event_type: str, callback: Callable) -> None:
        """Register a listener for *event_type*."""
        with self._lock:
            self._listeners.setdefault(event_type, []).append(callback)

    def off(self, event_type: str, callback: Callable) -> None:
        """Remove a previously registered listener."""
        with self._lock:
            listeners = self._listeners.get(event_type, [])
            try:
                listeners.remove(callback)
            except ValueError:
                pass

    def emit(self, event_type: str, event: Any = None) -> None:
        """Invoke all listeners registered for *event_type*."""
        with self._lock:
            listeners = list(self._listeners.get(event_type, []))
        for cb in listeners:
            cb(event)
