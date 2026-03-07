"""Temporal identity smoothing for tracked people.

Tracks people across frames by bounding-box center distance, maintains a
sliding window of identity observations per track, and emits a smoothed
identity that resists frame-to-frame flip-flops.

Usage::

    smoother = IdentitySmoother()

    # Once per frame:
    smoothed = smoother.smooth(detections, match_results)
    for si in smoothed:
        print(si.track_id, si.person_name, si.confidence)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from domain.detection.events import (
    FaceMatchResult,
    PersonDetection,
    SmoothedIdentity,
)


# ── Internal track representation ────────────────────────────────────


@dataclass
class _Track:
    track_id: int
    last_box: tuple[int, int, int, int]
    frames_since_seen: int = 0
    window: deque[tuple[str, float]] = field(default_factory=deque)


# ── Helpers ──────────────────────────────────────────────────────────


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _center_distance_sq(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    return (ax - bx) ** 2 + (ay - by) ** 2


# ── IdentitySmoother ────────────────────────────────────────────────


class IdentitySmoother:
    """Stateful per-frame smoother that stabilises identity labels.

    Parameters
    ----------
    window_size : int
        Number of recent observations kept per track.
    min_hit_ratio : float
        Minimum fraction of window entries that must agree on a name
        (excluding "Unknown") for the identity to be emitted.
    min_avg_confidence : float
        Minimum average confidence among the top-name entries.
    max_stale_frames : int
        Remove a track after this many consecutive frames without a
        matching detection.
    max_match_distance : float
        Maximum *squared* center distance (pixels) to associate a
        detection with an existing track.  Default 10 000 corresponds
        to ~100 px of movement between frames.
    """

    def __init__(
        self,
        window_size: int = 10,
        min_hit_ratio: float = 0.7,
        min_avg_confidence: float = 0.7,
        max_stale_frames: int = 30,
        max_match_distance: float = 10_000.0,
    ) -> None:
        self.window_size = window_size
        self.min_hit_ratio = min_hit_ratio
        self.min_avg_confidence = min_avg_confidence
        self.max_stale_frames = max_stale_frames
        self.max_match_distance = max_match_distance

        self._tracks: list[_Track] = []
        self._next_id: int = 0

    # ── Public API ───────────────────────────────────────────────────

    def smooth(
        self,
        detections: list[PersonDetection],
        match_results: list[FaceMatchResult],
    ) -> list[SmoothedIdentity]:
        """Process one frame and return smoothed identities.

        Parameters
        ----------
        detections
            Current-frame person detections (synchronous output of the
            person detector).
        match_results
            Latest face-match results from the async matcher.  May be
            from a slightly earlier frame; association is by box
            proximity, same as ``stream_processor._associate_matches``.
        """
        # 1. Build a lookup from detection index -> FaceMatchResult
        det_to_match = self._associate_matches(detections, match_results)

        # 2. Associate detections to existing tracks (greedy nearest)
        det_to_track = self._associate_detections(detections)

        # 3. Age all tracks, then update matched ones
        for track in self._tracks:
            track.frames_since_seen += 1

        results: list[SmoothedIdentity] = []

        for i, det in enumerate(detections):
            track = det_to_track.get(i)
            if track is None:
                track = self._create_track(det.box)
                det_to_track[i] = track

            # Update track state
            track.last_box = det.box
            track.frames_since_seen = 0

            # Push observation into the sliding window
            mr = det_to_match.get(i)
            if mr is not None:
                name = mr.person_name if mr.matched else "Unknown"
                conf = mr.confidence if mr.matched else 0.0
            else:
                name = "Unknown"
                conf = 0.0
            track.window.append((name, conf))
            if len(track.window) > self.window_size:
                track.window.popleft()

            # Compute smoothed identity
            si = self._resolve_identity(det, track)
            results.append(si)

        # 4. Expire stale tracks
        self._tracks = [
            t for t in self._tracks
            if t.frames_since_seen <= self.max_stale_frames
        ]

        return results

    # ── Internal helpers ─────────────────────────────────────────────

    def _create_track(self, box: tuple[int, int, int, int]) -> _Track:
        track = _Track(
            track_id=self._next_id,
            last_box=box,
            window=deque(maxlen=self.window_size),
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _associate_detections(
        self,
        detections: list[PersonDetection],
    ) -> dict[int, _Track]:
        """Greedy nearest-neighbor assignment of detections to tracks.

        O(n * m) where n = len(detections), m = len(tracks).
        """
        if not self._tracks:
            return {}

        assigned_dets: dict[int, _Track] = {}
        used_tracks: set[int] = set()

        # Build all pairs sorted by distance for greedy assignment
        pairs: list[tuple[float, int, int]] = []
        for i, det in enumerate(detections):
            for j, track in enumerate(self._tracks):
                d = _center_distance_sq(det.box, track.last_box)
                if d <= self.max_match_distance:
                    pairs.append((d, i, j))

        pairs.sort()

        for d, i, j in pairs:
            if i in assigned_dets or j in used_tracks:
                continue
            assigned_dets[i] = self._tracks[j]
            used_tracks.add(j)

        return assigned_dets

    @staticmethod
    def _associate_matches(
        detections: list[PersonDetection],
        match_results: list[FaceMatchResult],
    ) -> dict[int, FaceMatchResult]:
        """Map detection index -> closest FaceMatchResult by box center."""
        if not match_results:
            return {}

        associations: dict[int, FaceMatchResult] = {}
        used: set[int] = set()

        for i, det in enumerate(detections):
            best_j: int | None = None
            best_dist = float("inf")
            for j, mr in enumerate(match_results):
                if j in used:
                    continue
                d = _center_distance_sq(det.box, mr.person_detection.box)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j is not None:
                associations[i] = match_results[best_j]
                used.add(best_j)

        return associations

    def _resolve_identity(
        self,
        detection: PersonDetection,
        track: _Track,
    ) -> SmoothedIdentity:
        """Derive a smoothed identity from the track's observation window."""
        # Count named (non-Unknown) observations
        name_counts: dict[str, int] = {}
        name_conf_sums: dict[str, float] = {}

        for name, conf in track.window:
            if name == "Unknown":
                continue
            name_counts[name] = name_counts.get(name, 0) + 1
            name_conf_sums[name] = name_conf_sums.get(name, 0.0) + conf

        window_len = len(track.window)

        if name_counts and window_len > 0:
            top_name = max(name_counts, key=name_counts.__getitem__)
            hit_count = name_counts[top_name]
            hit_ratio = hit_count / window_len
            avg_confidence = name_conf_sums[top_name] / hit_count

            if (
                hit_ratio >= self.min_hit_ratio
                and avg_confidence >= self.min_avg_confidence
            ):
                return SmoothedIdentity(
                    person_detection=detection,
                    person_name=top_name,
                    confidence=avg_confidence,
                    is_smoothed=True,
                    track_id=track.track_id,
                )

        # Not enough evidence — label as Unknown
        return SmoothedIdentity(
            person_detection=detection,
            person_name="Unknown",
            confidence=0.0,
            is_smoothed=False,
            track_id=track.track_id,
        )
