"""Video input.

Kept separate from the pipeline so that the pipeline never depends on OpenCV.
Frames are yielded lazily; a 90-minute broadcast at 1080p25 is around 400 GB
decoded, so nothing here ever materialises a list of frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

__all__ = ["VideoMeta", "read_frames", "probe"]


@dataclass(frozen=True, slots=True)
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def _open(path: Path):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "video reading needs the 'vision' extra: pip install 'offball[vision]'"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open video: {path}")
    return cap


def probe(path: str | Path) -> VideoMeta:
    """Read video metadata without decoding frames."""
    import cv2

    path = Path(path)
    cap = _open(path)
    try:
        return VideoMeta(
            path=path,
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def read_frames(
    path: str | Path, stride: int = 1, start: int = 0, limit: int | None = None
) -> Iterator:
    """Yield BGR frames from a video file.

    Args:
        path: Video file.
        stride: Emit every Nth frame. Off-the-ball geometry changes slowly, so
            a stride of 5 (5 Hz from 25 fps footage) costs almost no analytical
            fidelity for a 5x speedup. Tracking quality does degrade at high
            strides — the IoU association assumes small frame-to-frame motion —
            so keep it at 1 if you care about identity through congestion.
        start: First frame index to consider.
        limit: Maximum number of frames to emit.
    """
    path = Path(path)
    cap = _open(path)
    emitted = 0
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= start and (index - start) % stride == 0:
                yield frame
                emitted += 1
                if limit is not None and emitted >= limit:
                    break
            index += 1
    finally:
        cap.release()
