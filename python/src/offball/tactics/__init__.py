"""Tactical analysis: off-the-ball metrics and match reporting.

This layer depends only on :mod:`offball.kernels` and :mod:`offball.types`, so
it runs against tracking data from any provider — not just this platform's
vision stage.
"""

from __future__ import annotations

from .offball import FrameScore, OffBallScore, ScoringConfig, score_frame
from .report import MatchReport, PlayerSummary, TeamSummary, build_report

__all__ = [
    "FrameScore",
    "MatchReport",
    "OffBallScore",
    "PlayerSummary",
    "ScoringConfig",
    "TeamSummary",
    "build_report",
    "score_frame",
]
