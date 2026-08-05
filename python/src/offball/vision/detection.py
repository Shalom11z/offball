"""Object detection: players, ball, and officials.

The pipeline depends on the :class:`Detector` protocol, never on a concrete
model. Two implementations ship:

:class:`YoloDetector`
    Production path. Wraps an Ultralytics YOLO checkpoint. ``ultralytics`` and
    ``torch`` are imported lazily inside the constructor so that importing
    ``offball`` costs nothing when you are only running the tactics layer.

:class:`ScriptedDetector`
    A deterministic detector replaying a fixed list of detections. This is what
    makes the pipeline testable end to end without weights, footage, or a GPU —
    see ``tests/test_pipeline.py``.

Fine-tuning notes for the production path are in ``docs/02-vision-pipeline.md``;
the short version is that a COCO-pretrained YOLO finds players acceptably and
the ball badly, and the ball is the part worth your annotation budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..types import BBox, Detection

__all__ = ["Detector", "DetectorConfig", "ScriptedDetector", "YoloDetector"]


@runtime_checkable
class Detector(Protocol):
    """Anything that turns a frame into detections.

    ``frame`` is an HxWx3 BGR array (the OpenCV convention). It is typed loosely
    because this module must import without numpy or OpenCV present.
    """

    def detect(self, frame) -> list[Detection]:
        ...


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    #: Minimum confidence for player detections. Tracking does its own,
    #: stricter filtering for track *initialisation*.
    confidence: float = 0.25
    #: Minimum confidence for the **ball**, kept separate from players so the
    #: two can be tuned independently.
    #:
    #: Left equal to the player threshold, because lowering it was **measured
    #: to make things worse**, which was not the expected result:
    #:
    #: ===========  ========  =========  ========
    #: ball_conf    detected  usable     coverage
    #: ===========  ========  =========  ========
    #: 0.25              32%       58%       56%
    #: 0.05              55%       54%       51%
    #: 0.03              57%       54%       51%
    #: ===========  ========  =========  ========
    #:
    #: Raw recall nearly doubles and usable output falls. At 0.03 a COCO YOLO
    #: emits **3.2 "sports ball" candidates per frame** (up to 9), of which at
    #: most one is the ball; the rest are boots, markings and stray white
    #: patches, scattered rather than in one fixed place. Neither per-frame
    #: selection nor the Viterbi trajectory pass in
    #: :func:`~offball.vision.ball.select_ball_trajectory` reliably picks the
    #: right one, because nothing in the geometry distinguishes a small white
    #: blob that moves plausibly from the ball.
    #:
    #: The fix is a detector that knows what a football looks like, not a
    #: threshold. See ``docs/06-roadmap.md``.
    ball_confidence: float = 0.25
    #: NMS IoU threshold.
    iou: float = 0.7
    #: Longest-side input resolution. 1280 rather than the usual 640: the ball
    #: is often under 10 pixels across in a wide broadcast shot, and halving the
    #: input resolution loses it entirely.
    image_size: int = 1280
    device: str | None = None


class YoloDetector:
    """Ultralytics YOLO wrapper.

    Args:
        weights: Path to a ``.pt`` checkpoint, or a model name Ultralytics can
            fetch (e.g. ``"yolov8x.pt"``).
        config: Inference settings.
        class_map: Maps model class names to our labels. The default handles a
            plain COCO model, where ``person`` covers players *and* officials —
            the referee is separated later by kit colour, in
            :mod:`offball.vision.teams`.

    Raises:
        ImportError: if ``ultralytics`` is not installed. Install the optional
            extra with ``pip install 'offball[vision]'``.
    """

    def __init__(
        self,
        weights: str = "yolov8x.pt",
        config: DetectorConfig | None = None,
        class_map: dict[str, str] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "YoloDetector needs the 'vision' extra: pip install 'offball[vision]'"
            ) from exc

        self.config = config or DetectorConfig()
        self.class_map = class_map or {"person": "player", "sports ball": "ball"}
        self._model = YOLO(weights)

    def detect(self, frame) -> list[Detection]:
        # Run at the lower of the two thresholds and filter per class below,
        # so the ball can be kept at a confidence that would flood the frame
        # with spurious players.
        results = self._model.predict(
            frame,
            conf=min(self.config.confidence, self.config.ball_confidence),
            iou=self.config.iou,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False,
        )
        out: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                raw_name = names[int(box.cls)]
                label = self.class_map.get(raw_name)
                if label is None:
                    continue
                confidence = float(box.conf)
                floor = (
                    self.config.ball_confidence
                    if label == "ball"
                    else self.config.confidence
                )
                if confidence < floor:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                out.append(Detection(BBox(x1, y1, x2, y2), confidence, label))
        return out


class ScriptedDetector:
    """Replays a fixed per-frame list of detections.

    Used for deterministic tests and for the demo pipeline, which needs no
    weights and no video file.

    Args:
        script: One list of detections per frame. Frames beyond the end of the
            script yield no detections.
    """

    def __init__(self, script: Sequence[Sequence[Detection]]) -> None:
        self._script = [list(f) for f in script]
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def detect(self, frame=None) -> list[Detection]:
        if self._index >= len(self._script):
            return []
        out = self._script[self._index]
        self._index += 1
        return list(out)
