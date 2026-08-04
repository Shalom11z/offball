"""Adapters for third-party football datasets.

Each converts a dataset's own annotation schema into the plain structures the
rest of the package uses, so no dataset's format leaks into the core.
"""

from __future__ import annotations

from .soccernet import (
    SOCCERNET_LINE_CLASSES,
    homography_from_annotation,
    iter_annotated_frames,
    pitch_lines,
)

__all__ = [
    "SOCCERNET_LINE_CLASSES",
    "homography_from_annotation",
    "iter_annotated_frames",
    "pitch_lines",
]
