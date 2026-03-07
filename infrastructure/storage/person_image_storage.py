"""Filesystem storage for person face and body crop images."""

from __future__ import annotations

import os
import shutil
from datetime import datetime

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class PersonImageStorage:
    """Saves and manages person detection crop images on disk.

    Directory layout::

        {base_dir}/YYYY/MM/DD/{detection_id}_face.jpg
        {base_dir}/YYYY/MM/DD/{detection_id}_body.jpg
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = base_dir

    # ── Save ─────────────────────────────────────────────────────────

    def save(
        self,
        detection_id: str,
        face_crop: np.ndarray | None,
        body_crop: np.ndarray | None,
        timestamp: str,
    ) -> tuple[str | None, str | None]:
        """Save crop images and return (face_path, body_path) relative to base_dir."""
        dt = datetime.fromisoformat(timestamp)
        day_dir = os.path.join(
            dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
        )
        abs_day_dir = os.path.join(self._base_dir, day_dir)
        os.makedirs(abs_day_dir, exist_ok=True)

        import cv2

        face_path: str | None = None
        body_path: str | None = None

        if face_crop is not None and face_crop.size > 0:
            rel = os.path.join(day_dir, f"{detection_id}_face.jpg")
            cv2.imwrite(os.path.join(self._base_dir, rel), face_crop)
            face_path = rel

        if body_crop is not None and body_crop.size > 0:
            rel = os.path.join(day_dir, f"{detection_id}_body.jpg")
            cv2.imwrite(os.path.join(self._base_dir, rel), body_crop)
            body_path = rel

        return face_path, body_path

    # ── Cleanup ──────────────────────────────────────────────────────

    def delete_before(self, cutoff_iso: str) -> None:
        """Delete image directories for dates strictly before the cutoff."""
        cutoff = datetime.fromisoformat(cutoff_iso).date()
        for year_name in self._listdir(self._base_dir):
            year_path = os.path.join(self._base_dir, year_name)
            if not os.path.isdir(year_path):
                continue
            for month_name in self._listdir(year_path):
                month_path = os.path.join(year_path, month_name)
                if not os.path.isdir(month_path):
                    continue
                for day_name in self._listdir(month_path):
                    day_path = os.path.join(month_path, day_name)
                    if not os.path.isdir(day_path):
                        continue
                    try:
                        folder_date = datetime(
                            int(year_name), int(month_name), int(day_name)
                        ).date()
                    except (ValueError, TypeError):
                        continue
                    if folder_date < cutoff:
                        shutil.rmtree(day_path, ignore_errors=True)

    # ── Path resolution ──────────────────────────────────────────────

    def get_image_path(self, path: str) -> str:
        """Resolve a relative crop path to an absolute filesystem path."""
        return os.path.join(self._base_dir, path)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _listdir(path: str) -> list[str]:
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []
