"""Batch clustering of unidentified person detections by face similarity.

Uses a simple union-find structure and pairwise L2 distance to group
unknown faces into clusters without requiring scikit-learn.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from infrastructure.database.person_log_db import PersonLogDB

logger = logging.getLogger(__name__)


# ── Union-Find ────────────────────────────────────────────────────────


class _UnionFind:
    """Minimal disjoint-set with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ── Clusterer ─────────────────────────────────────────────────────────


class UnknownPersonClusterer:
    """Groups unidentified person detections by face-encoding similarity.

    Parameters
    ----------
    distance_threshold:
        Maximum L2 distance between two 512-d face encodings to consider
        them the same person.
    min_cluster_size:
        Minimum number of detections required to form a new cluster.
    """

    def __init__(
        self,
        distance_threshold: float = 0.8,
        min_cluster_size: int = 3,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size

    # ── Public API ────────────────────────────────────────────────────

    def run(
        self,
        db: PersonLogDB,
        encodings_path: str | None = None,
    ) -> None:
        """Convenience entry point: cluster then optionally suggest identities."""
        self.cluster(db)
        if encodings_path is not None:
            self.suggest_identities(db, encodings_path)

    def cluster(self, db: PersonLogDB) -> None:
        """Cluster all unidentified detections that have a face encoding."""
        rows = db.get_unidentified_with_encodings()
        if not rows:
            logger.info("No unidentified detections with encodings to cluster.")
            return

        # Separate already-clustered from unassigned detections.
        unassigned = [r for r in rows if r.get("cluster_id") is None]
        logger.info(
            "Clustering: %d total unidentified, %d unassigned.",
            len(rows),
            len(unassigned),
        )

        # Phase 1 — try to assign unassigned detections to existing clusters.
        existing_clusters = db.get_clusters()
        still_unassigned = self._assign_to_existing_clusters(
            db, unassigned, existing_clusters
        )

        # Phase 2 — form new clusters from remaining unassigned detections.
        if still_unassigned:
            self._form_new_clusters(db, still_unassigned)

    def suggest_identities(
        self,
        db: PersonLogDB,
        encodings_path: str,
        match_threshold: float = 0.8,
    ) -> None:
        """Compare cluster centroids against known encodings and suggest names."""
        path = Path(encodings_path)
        if not path.exists():
            logger.warning("Encodings file not found: %s", encodings_path)
            return

        with open(path, "rb") as f:
            known: dict[str, list] = pickle.load(f)

        if not known:
            logger.info("Known-encodings file is empty; skipping suggestions.")
            return

        # Pre-stack all known encodings for vectorised comparison.
        known_names: list[str] = []
        known_encs: list[np.ndarray] = []
        for name, enc_list in known.items():
            for enc in enc_list:
                known_names.append(name)
                known_encs.append(np.asarray(enc, dtype=np.float32))
        known_matrix = np.stack(known_encs)  # (K, 512)

        clusters = db.get_clusters()
        for cluster in clusters:
            # Skip clusters that already have a suggestion or are confirmed.
            if cluster.get("suggested_name") or cluster.get("confirmed"):
                continue

            raw = cluster.get("representative_encoding")
            if raw is None:
                continue

            centroid = np.frombuffer(raw, dtype=np.float32)
            distances = np.linalg.norm(known_matrix - centroid, axis=1)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist < match_threshold:
                best_name = known_names[best_idx]
                logger.info(
                    "Cluster %d -> suggested '%s' (dist=%.3f)",
                    cluster["cluster_id"],
                    best_name,
                    best_dist,
                )
                db.update_cluster_suggestion(cluster["cluster_id"], best_name)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _decode_encoding(raw: bytes) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.float32)

    def _assign_to_existing_clusters(
        self,
        db: PersonLogDB,
        unassigned: list[dict],
        clusters: list[dict],
    ) -> list[dict]:
        """Try to slot unassigned detections into existing clusters.

        Returns the detections that could not be assigned.
        """
        if not clusters:
            return unassigned

        # Build matrix of cluster centroids.
        cluster_ids: list[int] = []
        centroids: list[np.ndarray] = []
        for c in clusters:
            raw = c.get("representative_encoding")
            if raw is None:
                continue
            cluster_ids.append(c["cluster_id"])
            centroids.append(self._decode_encoding(raw))

        if not centroids:
            return unassigned

        centroid_matrix = np.stack(centroids)  # (C, 512)
        still_unassigned: list[dict] = []

        for det in unassigned:
            enc = self._decode_encoding(det["face_encoding"])
            distances = np.linalg.norm(centroid_matrix - enc, axis=1)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist < self.distance_threshold:
                cid = cluster_ids[best_idx]
                db.assign_cluster(det["detection_id"], cid)
                logger.debug(
                    "Assigned detection %s to existing cluster %d (dist=%.3f)",
                    det["detection_id"],
                    cid,
                    best_dist,
                )
            else:
                still_unassigned.append(det)

        assigned_count = len(unassigned) - len(still_unassigned)
        if assigned_count:
            logger.info(
                "Assigned %d detections to existing clusters.", assigned_count
            )

        return still_unassigned

    def _form_new_clusters(self, db: PersonLogDB, detections: list[dict]) -> None:
        """Use union-find to discover new clusters among unassigned detections."""
        n = len(detections)
        if n < self.min_cluster_size:
            logger.info(
                "Only %d unassigned detections; need %d to form a cluster.",
                n,
                self.min_cluster_size,
            )
            return

        # Decode all encodings and build a matrix for pairwise distances.
        encodings = np.stack(
            [self._decode_encoding(d["face_encoding"]) for d in detections]
        )  # (N, 512)

        uf = _UnionFind(n)

        # O(n^2) pairwise merge — acceptable for periodic batch runs.
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(encodings[i] - encodings[j]))
                if dist < self.distance_threshold:
                    uf.union(i, j)

        # Gather groups by root.
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = uf.find(i)
            groups.setdefault(root, []).append(i)

        new_cluster_count = 0
        for member_indices in groups.values():
            if len(member_indices) < self.min_cluster_size:
                continue

            # Compute centroid as the mean encoding of all members.
            member_encodings = encodings[member_indices]
            centroid = member_encodings.mean(axis=0).astype(np.float32)
            centroid_bytes = centroid.tobytes()

            cluster_id = db.create_cluster(centroid_bytes)
            for idx in member_indices:
                db.assign_cluster(detections[idx]["detection_id"], cluster_id)

            new_cluster_count += 1
            logger.info(
                "Created cluster %d with %d detections.",
                cluster_id,
                len(member_indices),
            )

        if new_cluster_count:
            logger.info("Formed %d new cluster(s).", new_cluster_count)
        else:
            logger.info("No new clusters formed (none met min_cluster_size=%d).", self.min_cluster_size)
