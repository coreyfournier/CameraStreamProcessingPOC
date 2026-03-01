"""Person-only object detection using YOLOv8.

Wraps the Ultralytics YOLOv8 nano model and filters detections to
class 0 (person).  Returns detections synchronously AND emits events
for async listeners.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from shared.event_emitter import EventEmitter
from domain.detection.events import (
    FrameContext,
    PersonDetection,
    PersonDetectionEvent,
)

PERSON_CLASS_ID = 0  # COCO class 0 = person


class PersonDetector(EventEmitter):
    """Loads YOLOv8 and detects people in video frames.

    Inherits from EventEmitter so callers can subscribe to
    ``"person_detected"`` events.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = YOLO(model_name, verbose=False)
        self.confidence_threshold = confidence_threshold

    def process_frame(self, context: FrameContext) -> list[PersonDetection]:
        """Run detection on *context.frame*, return person detections.

        YOLOv8 handles multi-scale detection natively (640x640 input
        with feature pyramid), so no tiling is needed even for
        high-resolution frames.

        Also emits a ``"person_detected"`` event when at least one person
        is found so that async listeners (e.g. FaceMatcher) can react.
        """
        frame = context.frame
        h, w = frame.shape[:2]

        results = self.model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: list[PersonDetection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                confidence = float(box.conf[0])

                # Clamp to frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2].copy()

                detections.append(
                    PersonDetection(
                        confidence=confidence,
                        box=(x1, y1, x2, y2),
                        person_crop=crop,
                    )
                )

        if detections:
            self.emit(
                "person_detected",
                PersonDetectionEvent(context=context, detections=detections),
            )

        return detections
