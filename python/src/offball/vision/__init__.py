"""Computer-vision stage: detection, tracking, team assignment, calibration.

Submodules import their heavy dependencies (OpenCV, ultralytics, torch) lazily,
so importing this package is cheap and works without the ``vision`` extra
installed.
"""

from __future__ import annotations

from .calibration import Calibration, CalibrationConfig, HomographySmoother, calibrate_frame
from .detection import Detector, DetectorConfig, ScriptedDetector
from .lines import ClassicalKeypointSource, PitchLineConfig
from .teams import TeamAssigner
from .tracking import Track, Tracker, TrackerConfig

__all__ = [
    "Calibration",
    "CalibrationConfig",
    "ClassicalKeypointSource",
    "Detector",
    "DetectorConfig",
    "HomographySmoother",
    "PitchLineConfig",
    "ScriptedDetector",
    "TeamAssigner",
    "Track",
    "Tracker",
    "TrackerConfig",
    "calibrate_frame",
]
