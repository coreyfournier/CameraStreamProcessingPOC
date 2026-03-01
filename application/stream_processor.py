"""Stream annotation helpers.

Provides draw_detections() and the box-association utilities used by the
main processing loop to overlay bounding boxes and labels on frames.
"""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np

from domain.detection.events import FaceMatchResult, PersonDetection

# Colors (BGR)
COLOR_MATCHED = (0, 200, 0)     # Green — known face
COLOR_UNMATCHED = (0, 220, 255) # Yellow — person, unknown face


def _box_center(box):
    """Return (cx, cy) of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _box_distance_sq(box_a, box_b):
    """Squared Euclidean distance between two box centers."""
    ax, ay = _box_center(box_a)
    bx, by = _box_center(box_b)
    return (ax - bx) ** 2 + (ay - by) ** 2


def _associate_matches(
    detections: list[PersonDetection],
    match_results: list[FaceMatchResult],
) -> dict[int, FaceMatchResult]:
    """Associate async match results to current detections by box proximity.

    Returns a dict mapping detection index → FaceMatchResult (or None).
    """
    associations: dict[int, FaceMatchResult] = {}
    if not match_results:
        return associations

    # For each current detection, find the closest match result by box center
    used: set[int] = set()
    for i, det in enumerate(detections):
        best_j = None
        best_dist = float("inf")
        for j, mr in enumerate(match_results):
            if j in used:
                continue
            d = _box_distance_sq(det.box, mr.person_detection.box)
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j is not None:
            associations[i] = match_results[best_j]
            used.add(best_j)

    return associations


def draw_detections(
    frame: np.ndarray,
    detections: list[PersonDetection],
    match_results: list[FaceMatchResult],
) -> np.ndarray:
    """Draw bounding boxes and labels on frame.

    - Green box + person name for matched faces
    - Yellow box + "Person" for unmatched detections
    """
    associations = _associate_matches(detections, match_results)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.box
        mr = associations.get(i)

        if mr and mr.matched:
            color = COLOR_MATCHED
            label = f"{mr.person_name} ({mr.confidence:.0%})"
        else:
            color = COLOR_UNMATCHED
            label = f"Person ({det.confidence:.0%})"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
        )

    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame, timestamp, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    return frame
