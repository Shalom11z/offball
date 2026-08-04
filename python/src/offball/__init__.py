"""offball — automated off-the-ball positioning analysis from match footage.

Quick start::

    from offball.pipeline import Pipeline, PipelineConfig
    from offball.vision.detection import YoloDetector

    pipeline = Pipeline(detector=YoloDetector(), keypoints=my_keypoint_model)
    result = pipeline.run(frames)
    print(result.report.to_dict())

The tactics layer (:mod:`offball.tactics`, :mod:`offball.kernels`) has no heavy
dependencies and can be imported and used on tracking data from any source —
you do not need the vision stack to use the metrics.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .kernels import BACKEND, using_rust
from .types import BBox, Detection, FrameState, PlayerObservation, Team

__all__ = [
    "BACKEND",
    "BBox",
    "Detection",
    "FrameState",
    "PlayerObservation",
    "Team",
    "__version__",
    "using_rust",
]
