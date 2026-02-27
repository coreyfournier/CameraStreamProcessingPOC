"""Person-only object detection using MobileNet-SSD.

Wraps the Caffe model and filters detections to class 15 (person).
Returns detections synchronously AND emits events for async listeners.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from events import (
    EventEmitter,
    FrameContext,
    PersonDetection,
    PersonDetectionEvent,
)

PERSON_CLASS_ID = 15


class PersonDetector(EventEmitter):
    """Loads MobileNet-SSD and detects people in video frames.

    Inherits from EventEmitter so callers can subscribe to
    ``"person_detected"`` events.
    """

    def __init__(
        self,
        prototxt: str | None = None,
        weights: str | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        prototxt = prototxt or os.path.join(
            ".", "MobileNetSSN", "MobileNetSSD_deploy.prototxt"
        )
        weights = weights or os.path.join(
            ".", "MobileNetSSN", "MobileNetSSD_deploy.caffemodel"
        )
        self.net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        self.confidence_threshold = confidence_threshold

    def process_frame(self, context: FrameContext) -> list[PersonDetection]:
        """Run detection on *context.frame*, return person detections.

        Also emits a ``"person_detected"`` event when at least one person
        is found so that async listeners (e.g. FaceMatcher) can react.
        """
        frame = context.frame
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        raw = self.net.forward()

        detections: list[PersonDetection] = []
        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            class_id = int(raw[0, 0, i, 1])

            if class_id != PERSON_CLASS_ID:
                continue
            if confidence < self.confidence_threshold:
                continue

            box = raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

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
