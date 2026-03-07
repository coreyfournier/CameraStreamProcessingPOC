"""Face matching against a known-faces encoding database.

Listens for ``"person_detected"`` events, runs ``facenet-pytorch`` (MTCNN +
InceptionResnetV1) to identify people, and emits ``"face_matched"`` events.

Gracefully degrades when facenet-pytorch is not installed — the listener
becomes a silent no-op.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from shared.event_emitter import EventEmitter
from domain.detection.events import (
    FaceMatchEvent,
    FaceMatchResult,
    PersonDetection,
    PersonDetectionEvent,
)

# ── Optional dependency guard ───────────────────────────────────────

try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1

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
        Maximum L2 distance between 512-d embeddings to consider a match
        (lower = stricter). Typical range: 0.7–1.1; default 0.9.
    min_detection_confidence : float
        Ignore person detections below this confidence.
    """

    def __init__(
        self,
        encodings_path: str | Path = "./faces/encodings.pkl",
        tolerance: float = 0.9,
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
                "WARNING: facenet-pytorch not installed — "
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._mtcnn = MTCNN(keep_all=False, device=device)
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self._device = device
        self._known_matrix = np.array(self.known_encodings)  # (N, 512) for fast batch distance

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
                face_encoding=None,
            )

        # Convert BGR (OpenCV) → PIL RGB (facenet-pytorch)
        pil_crop = Image.fromarray(cv2.cvtColor(det.person_crop, cv2.COLOR_BGR2RGB))

        # Detect face bounding box(es) in the person crop
        boxes, probs = self._mtcnn.detect(pil_crop)

        if boxes is None or len(boxes) == 0:
            return FaceMatchResult(
                person_detection=det,
                matched=False,
                person_name="Unknown",
                confidence=0.0,
                face_location=None,
                face_encoding=None,
            )

        # Use the most prominent face (highest probability)
        best_face_idx = int(np.argmax(probs))
        x1, y1, x2, y2 = boxes[best_face_idx]
        face_location = (int(y1), int(x2), int(y2), int(x1))  # (top, right, bottom, left)

        # Get aligned face tensor for embedding
        face_tensor = self._mtcnn(pil_crop)
        if face_tensor is None:
            return FaceMatchResult(
                person_detection=det,
                matched=False,
                person_name="Unknown",
                confidence=0.0,
                face_location=face_location,
                face_encoding=None,
            )

        # Compute 512-d embedding
        with torch.no_grad():
            embedding = (
                self._model(face_tensor.unsqueeze(0).to(self._device))
                .cpu()
                .numpy()[0]
            )

        # Serialize embedding for storage
        encoding_bytes = embedding.astype(np.float32).tobytes()

        # L2 distances against all known encodings
        distances = np.linalg.norm(self._known_matrix - embedding, axis=1)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        # Normalise confidence: 0.5 at tolerance boundary, 1.0 at distance 0
        confidence = max(0.0, 1.0 - 0.5 * best_distance / self.tolerance)

        if best_distance <= self.tolerance:
            return FaceMatchResult(
                person_detection=det,
                matched=True,
                person_name=self.known_names[best_idx],
                confidence=confidence,
                face_location=face_location,
                face_encoding=encoding_bytes,
            )

        return FaceMatchResult(
            person_detection=det,
            matched=False,
            person_name="Unknown",
            confidence=confidence,
            face_location=face_location,
            face_encoding=encoding_bytes,
        )
