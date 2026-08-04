"""Visual debugging renderers.

Images are hard to assert on, so these check the properties that actually
matter: that a correct homography draws ink where the paint is, that a wrong
one does not, and that nothing crashes on the degenerate cases which are
precisely when you reach for a debug view.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from test_lines import H_TRUE, render_pitch  # noqa: E402

from offball.demo import SyntheticMatch, synthetic_frames  # noqa: E402
from offball.types import BBox, FrameState, PlayerObservation, Team  # noqa: E402
from offball.vision.lines import _invert  # noqa: E402
from offball.viz import (  # noqa: E402
    TEAM_COLOURS,
    overlay_calibration,
    render_frame_pair,
    render_pitch_map,
)

TRUE_IMAGE_TO_PITCH = _invert(H_TRUE)


def green_ink(image) -> np.ndarray:
    """Mask of the overlay's drawing colour (0, 240, 0) in BGR."""
    return (
        (image[:, :, 1] > 180) & (image[:, :, 0] < 120) & (image[:, :, 2] < 120)
    )


def test_overlay_returns_a_new_image():
    frame = render_pitch()
    before = frame.copy()
    out = overlay_calibration(frame, TRUE_IMAGE_TO_PITCH)
    assert out.shape == frame.shape
    assert np.array_equal(frame, before), "the input frame must not be modified"


def test_a_correct_homography_draws_onto_the_paint():
    """The core claim: drawn lines land on painted ones when calibration is right."""
    frame = render_pitch()
    out = overlay_calibration(frame, TRUE_IMAGE_TO_PITCH)

    ink = green_ink(out)
    assert ink.sum() > 500, "the overlay drew almost nothing"

    # Where the overlay drew, the original frame should have been near-white
    # paint. Sample the goal/touch lines, which the renderer definitely drew.
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    under = grey[ink]
    on_paint = (under > 150).mean()
    assert on_paint > 0.4, f"only {on_paint:.0%} of drawn ink landed on paint"


def test_a_wrong_homography_draws_off_the_paint():
    """The diagnostic only works if a bad fit looks obviously bad."""
    frame = render_pitch()
    good = np.array(TRUE_IMAGE_TO_PITCH, float)

    correct = overlay_calibration(frame, tuple(good))
    shifted = list(good)
    shifted[2] += 18.0  # translate the pitch 18m
    wrong = overlay_calibration(frame, tuple(shifted))

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    good_hit = (grey[green_ink(correct)] > 150).mean()
    bad_hit = (grey[green_ink(wrong)] > 150).mean()
    assert bad_hit < good_hit / 2, (
        f"a wrong homography still hit {bad_hit:.0%} of paint vs {good_hit:.0%}"
    )


def test_uncalibrated_frames_are_labelled_not_crashed():
    frame = render_pitch()
    out = overlay_calibration(frame, None)
    assert out.shape == frame.shape
    # A red warning is drawn.
    red = (out[:, :, 2] > 180) & (out[:, :, 0] < 90) & (out[:, :, 1] < 90)
    assert red.sum() > 100


def test_singular_homography_is_survived():
    frame = render_pitch()
    out = overlay_calibration(frame, (0.0,) * 9)
    assert out.shape == frame.shape


def test_pitch_map_renders_players_and_ball():
    states = synthetic_frames(SyntheticMatch(frames=20))
    canvas = render_pitch_map(states[10], scale=8)

    assert canvas.shape[:2] == (68 * 8, 105 * 8)
    for team in (Team.HOME, Team.AWAY):
        colour = np.array(TEAM_COLOURS[team])
        close = (np.abs(canvas.astype(int) - colour).sum(axis=2) < 40).sum()
        assert close > 20, f"no {team.value} players drawn"


def test_pitch_map_with_control_shading():
    states = synthetic_frames(SyntheticMatch(frames=20))
    plain = render_pitch_map(states[10], scale=6)
    shaded = render_pitch_map(states[10], scale=6, show_control=True)
    assert shaded.shape == plain.shape
    assert not np.array_equal(plain, shaded), "control shading had no effect"


def test_pitch_map_of_an_empty_frame():
    """A frame with nothing tracked still renders the pitch."""
    empty = FrameState(frame_index=0, timestamp=0.0)
    canvas = render_pitch_map(empty, scale=6, show_control=True)
    assert canvas.shape[:2] == (68 * 6, 105 * 6)


def test_pitch_map_skips_unprojected_players():
    state = FrameState(
        frame_index=0,
        timestamp=0.0,
        players=(PlayerObservation(1, BBox(0, 0, 1, 1), Team.HOME),),  # no pitch_xy
    )
    canvas = render_pitch_map(state, scale=6)
    assert canvas is not None


def test_frame_pair_places_both_views_side_by_side():
    frame = render_pitch()
    states = synthetic_frames(SyntheticMatch(frames=10))
    state = states[5]
    pair = render_frame_pair(frame, state)
    assert pair.shape[1] > frame.shape[1], "the map should be appended to the right"
    assert pair.shape[0] == frame.shape[0]
