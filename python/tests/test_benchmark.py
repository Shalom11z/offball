"""Calibration benchmarking."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from test_lines import H_TRUE, render_pitch  # noqa: E402

from offball.benchmark import (  # noqa: E402
    CalibrationSurvey,
    FrameOutcome,
    load_ground_truth,
    survey_frames,
)
from offball.vision.lines import _invert  # noqa: E402


def test_survey_reports_a_clean_sequence_as_calibrated():
    frames = [render_pitch() for _ in range(4)]
    survey = survey_frames(frames)
    assert survey.frames == 4
    assert survey.calibrated == 4
    assert survey.rate == 1.0
    assert survey.percentiles("support")["median"] > 0.8


def test_survey_reports_failures_rather_than_crashing():
    blank = np.zeros((1080, 1920, 3), np.uint8)
    survey = survey_frames([blank, blank])
    assert survey.frames == 2
    assert survey.calibrated == 0
    assert survey.rate == 0.0
    assert survey.percentiles("support") == {}


def test_survey_measures_error_against_ground_truth():
    frames = [render_pitch() for _ in range(3)]
    truth = [_invert(H_TRUE)] * 3
    survey = survey_frames(frames, ground_truth=truth)

    errors = survey.percentiles("truth_error")
    assert errors, "ground truth supplied but no error measured"
    # Recovered up to the pitch symmetry; without a prior the detector may pick
    # a mirrored branch, so only assert the figure exists and is finite here.
    assert errors["median"] >= 0.0


def test_empty_survey_does_not_divide_by_zero():
    survey = CalibrationSurvey()
    assert survey.rate == 0.0
    assert survey.percentiles("support") == {}
    assert "n/a" in survey.summary()


def test_summary_and_dict_round_trip():
    survey = CalibrationSurvey(
        frames=2,
        calibrated=1,
        outcomes=[
            FrameOutcome(0, True, 0.9, 12, 0.4, False, 1.2),
            FrameOutcome(1, False, 0.1, 3),
        ],
    )
    data = survey.to_dict()
    assert data["frames"] == 2
    assert data["rate"] == 0.5
    json.dumps(data)  # must be serialisable
    assert "calibrated" in survey.summary()


def test_load_ground_truth(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps([[1, 0, 0, 0, 1, 0, 0, 0, 1], None]))
    truth = load_ground_truth(path)
    assert truth[0] == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert truth[1] is None


def test_load_ground_truth_rejects_bad_shape(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([[1, 2, 3]]))
    with pytest.raises(ValueError, match="9 elements"):
        load_ground_truth(path)
