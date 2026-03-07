"""Orchestrator that wires PersonDetector → AsyncFaceMatcherWrapper.

Exposes a simple API for the main loop:
    pipeline.process_frame(frame, frame_number)  → synchronous detections
    pipeline.get_latest_matches()                → non-blocking poll
    pipeline.on_face_matched(callback)           → register external listener
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

from shared.event_emitter import EventEmitter
from domain.detection.events import (
    FaceMatchEvent,
    FaceMatchResult,
    FrameContext,
    PersonDetection,
    PersonDetectionEvent,
)
from infrastructure.detection.facenet_face_matcher import FaceMatcher
from infrastructure.detection.yolo_person_detector import PersonDetector


# ── Async wrapper ───────────────────────────────────────────────────


class AsyncFaceMatcherWrapper(EventEmitter):
    """Run a FaceMatcher on a background thread with frame skipping.

    Parameters
    ----------
    matcher : FaceMatcher
        The underlying (synchronous) face matcher.
    max_queue_size : int
        Bounded queue depth — stale events are dropped.
    skip_frames : int
        Only attempt matching every *skip_frames*-th person-detection event.
    """

    def __init__(
        self,
        matcher: FaceMatcher,
        max_queue_size: int = 2,
        skip_frames: int = 5,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.skip_frames = skip_frames
        self._queue: queue.Queue[PersonDetectionEvent | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._latest_results: list[FaceMatchResult] = []
        self._results_lock = threading.Lock()
        self._event_count = 0
        self._thread: threading.Thread | None = None
        self._running = False

        # Forward face_matched events from the inner matcher
        self.matcher.on("face_matched", self._on_inner_match)

    # ── Public API ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background worker thread."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="face-matcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to exit and wait for it."""
        self._running = False
        # Unblock the worker if it's waiting on the queue
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def enqueue(self, event: PersonDetectionEvent) -> None:
        """Submit a detection event for async matching (with frame skipping)."""
        self._event_count += 1
        if self._event_count % self.skip_frames != 0:
            return
        # Drop stale events when the queue is full
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass

    def get_latest_matches(self) -> list[FaceMatchResult]:
        """Return the most recent match results (non-blocking)."""
        with self._results_lock:
            return list(self._latest_results)

    # ── Internal ────────────────────────────────────────────────────

    def _worker(self) -> None:
        """Background loop: pull events from the queue and match."""
        while self._running:
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event is None:  # Poison pill
                break
            try:
                self.matcher.on_person_detected(event)
            except Exception:
                import traceback
                traceback.print_exc()

    def _on_inner_match(self, event: FaceMatchEvent) -> None:
        """Store latest results and re-emit for external listeners."""
        with self._results_lock:
            self._latest_results = list(event.results)
        self.emit("face_matched", event)


# ── DetectionPipeline ───────────────────────────────────────────────


class DetectionPipeline:
    """High-level orchestrator consumed by the main loop.

    Parameters
    ----------
    confidence_threshold : float
        Confidence cutoff for person detection.
    encodings_path : str
        Path to ``encodings.pkl`` for face matching.
    match_tolerance : float
        Max face distance to consider a match.
    match_min_confidence : float
        Ignore person detections below this confidence for matching.
    match_skip_frames : int
        Only attempt face matching every N frames with people.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        encodings_path: str = "./faces/encodings.pkl",
        match_tolerance: float = 0.6,
        match_min_confidence: float = 0.5,
        match_skip_frames: int = 5,
    ) -> None:
        self.detector = PersonDetector(
            confidence_threshold=confidence_threshold,
        )
        self.matcher = FaceMatcher(
            encodings_path=encodings_path,
            tolerance=match_tolerance,
            min_detection_confidence=match_min_confidence,
        )
        self.async_matcher = AsyncFaceMatcherWrapper(
            matcher=self.matcher,
            skip_frames=match_skip_frames,
        )

        # Wire detector → async matcher
        self.detector.on("person_detected", self.async_matcher.enqueue)

    def start(self) -> None:
        """Start the background face-matching thread."""
        self.async_matcher.start()

    def stop(self) -> None:
        """Stop the background thread and clean up."""
        self.async_matcher.stop()

    def process_frame(
        self, frame, frame_number: int
    ) -> list[PersonDetection]:
        """Run person detection on a frame (synchronous).

        Returns detections immediately for drawing. Face matching
        happens asynchronously in the background.
        """
        context = FrameContext(
            frame=frame,
            frame_number=frame_number,
            timestamp=time.time(),
        )
        return self.detector.process_frame(context)

    def get_latest_matches(self) -> list[FaceMatchResult]:
        """Poll the most recent face-match results (non-blocking)."""
        return self.async_matcher.get_latest_matches()

    def on_face_matched(self, callback: Callable) -> None:
        """Register an external listener for face-match events."""
        self.async_matcher.on("face_matched", callback)
