"""Face matching against a known-faces encoding database.

Listens for ``"person_detected"`` events, runs ``face_recognition`` to
identify people, and emits ``"face_matched"`` events with results.

Gracefully degrades when dlib / face_recognition are not installed —
the listener becomes a silent no-op.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from events import (
    EventEmitter,
    FaceMatchEvent,
    FaceMatchResult,
    PersonDetection,
    PersonDetectionEvent,
)

# ── Optional dependency guard ───────────────────────────────────────

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


# ── FaceMatcher ─────────────────────────────────────────────────────


class FaceMatcher(EventEmitter):
    """Match detected people against a pre-built face encoding database.

    Parameters
    ----------
    encodings_path : str | Path
        Path to ``encodings.pkl`` produced by ``ExportLightroomFaces.py``.
    tolerance : float
        Maximum face distance to consider a match (lower = stricter).
    min_detection_confidence : float
        Ignore person detections below this MobileNet confidence.
    """

    def __init__(
        self,
        encodings_path: str | Path = "./faces/encodings.pkl",
        tolerance: float = 0.6,
        min_detection_confidence: float = 0.5,
    ) -> None:
        super().__init__()
        self.tolerance = tolerance
        self.min_detection_confidence = min_detection_confidence
        self.known_names: list[str] = []
        self.known_encodings: list[np.ndarray] = []
        self.available = False

        if not FACE_RECOGNITION_AVAILABLE:
            print(
                "WARNING: face_recognition not installed — "
                "face matching disabled (person detection still works)"
            )
            return

        encodings_path = Path(encodings_path)
        if not encodings_path.exists():
            print(
                f"WARNING: encodings file not found at {encodings_path} — "
                "face matching disabled"
            )
            return

        with open(encodings_path, "rb") as f:
            encodings_dict: dict[str, list[np.ndarray]] = pickle.load(f)

        # Flatten to parallel lists for vectorised distance computation
        for name, encs in encodings_dict.items():
            for enc in encs:
                self.known_names.append(name)
                self.known_encodings.append(enc)

        if not self.known_encodings:
            print("WARNING: encodings file is empty — face matching disabled")
            return

        self.available = True
        print(
            f"FaceMatcher loaded {len(self.known_encodings)} encodings "
            f"for {len(encodings_dict)} people"
        )

    # ── Event handler ───────────────────────────────────────────────

    def on_person_detected(self, event: PersonDetectionEvent) -> None:
        """Handle a person-detected event by attempting face matching."""
        if not self.available:
            return

        results: list[FaceMatchResult] = []

        for det in event.detections:
            result = self._match_detection(det)
            results.append(result)

        if results:
            self.emit(
                "face_matched",
                FaceMatchEvent(context=event.context, results=results),
            )

    # ── Internal ────────────────────────────────────────────────────

    def _match_detection(self, det: PersonDetection) -> FaceMatchResult:
        """Try to match a single person crop against known faces."""
        if det.confidence < self.min_detection_confidence:
            return FaceMatchResult(
                person_detection=det,
                matched=False,
                person_name="Unknown",
                confidence=0.0,
                face_location=None,
            )

        # Convert BGR (OpenCV) → RGB (face_recognition)
        rgb_crop = cv2.cvtColor(det.person_crop, cv2.COLOR_BGR2RGB)

        # Detect faces within the person crop
        face_locations = face_recognition.face_locations(rgb_crop, model="hog")

        if not face_locations:
            return FaceMatchResult(
                person_detection=det,
                matched=False,
                person_name="Unknown",
                confidence=0.0,
                face_location=None,
            )

        # Use the largest face (by area) if multiple are found
        largest = max(
            face_locations,
            key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
        )

        encodings = face_recognition.face_encodings(
            rgb_crop, known_face_locations=[largest]
        )
        if not encodings:
            return FaceMatchResult(
                person_detection=det,
                matched=False,
                person_name="Unknown",
                confidence=0.0,
                face_location=largest,
            )

        # Compare against all known encodings
        distances = face_recognition.face_distance(
            self.known_encodings, encodings[0]
        )
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        if best_distance <= self.tolerance:
            return FaceMatchResult(
                person_detection=det,
                matched=True,
                person_name=self.known_names[best_idx],
                confidence=1.0 - best_distance,
                face_location=largest,
            )

        return FaceMatchResult(
            person_detection=det,
            matched=False,
            person_name="Unknown",
            confidence=1.0 - best_distance,
            face_location=largest,
        )
