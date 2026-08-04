"""Classical pitch-line detection.

The central test renders a pitch through a *known* homography and asserts the
detector recovers it. That is a genuine end-to-end check of masking, Hough
detection, family splitting and template matching together — and it is possible
precisely because the homography being recovered is known ground truth, unlike
on real footage.

Synthetic frames are clean by construction, so passing here does not mean the
detector works on broadcast video. It means the geometry is right.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from offball import kernels  # noqa: E402
from offball.vision.calibration import CalibrationConfig, calibrate_frame  # noqa: E402
from offball.vision.lines import (  # noqa: E402
    TEMPLATE_X,
    TEMPLATE_Y,
    ClassicalKeypointSource,
    Line,
    PitchLineConfig,
    detect_lines,
    line_intersection,
    line_mask,
    pitch_mask,
    split_families,
)
from offball.vision.lines import _invert as invert  # noqa: E402

WIDTH, HEIGHT = 1920, 1080
PITCH_L, PITCH_W = 105.0, 68.0

# Ground-truth pitch -> image homography: a gantry camera at the halfway line.
# The far touchline sits higher in frame and is drawn toward the centre.
H_TRUE = (12.0, 5.3, 250.0, 0.2, -6.5, 980.0, 0.0, 0.004, 1.0)


def to_image(p):
    q = kernels.project(H_TRUE, [p])[0]
    assert q is not None
    return q


def render_pitch(
    homography=H_TRUE, *, noise: float = 0.0, missing: tuple[float, ...] = ()
) -> np.ndarray:
    """Draw a pitch as a camera would see it.

    Args:
        homography: pitch -> image transform.
        noise: standard deviation of additive Gaussian noise.
        missing: template x-lines to omit, simulating worn or occluded paint.
    """
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    frame[:] = (40, 30, 30)  # dark surround: stands, not turf

    def proj(p):
        q = kernels.project(homography, [p])[0]
        return None if q is None else (round(q[0]), round(q[1]))

    corners = [proj(c) for c in ((0, 0), (PITCH_L, 0), (PITCH_L, PITCH_W), (0, PITCH_W))]
    assert all(c is not None for c in corners)
    # BGR: a plausible grass green.
    cv2.fillConvexPoly(frame, np.array(corners, np.int32), (60, 140, 70))

    white = (245, 245, 245)
    for x in TEMPLATE_X:
        if x in missing:
            continue
        a, b = proj((x, 0.0)), proj((x, PITCH_W))
        if a and b:
            cv2.line(frame, a, b, white, 3, cv2.LINE_AA)
    for y in TEMPLATE_Y:
        a, b = proj((0.0, y)), proj((PITCH_L, y))
        if a and b:
            cv2.line(frame, a, b, white, 3, cv2.LINE_AA)

    if noise > 0:
        rng = np.random.default_rng(0)
        noisy = frame.astype(np.float32) + rng.normal(0, noise, frame.shape)
        frame = np.clip(noisy, 0, 255).astype(np.uint8)
    return frame


def crowd_frame() -> np.ndarray:
    """A close-up with no pitch in view."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 90, (HEIGHT, WIDTH, 3), dtype=np.uint8)


# ------------------------------------------------------------------ geometry


def test_line_normalisation_collapses_equivalent_forms():
    a = Line(50.0, 0.3).normalised()
    b = Line(-50.0, 0.3 + math.pi).normalised()
    assert a.rho == pytest.approx(b.rho)
    assert a.theta == pytest.approx(b.theta)
    assert 0 <= a.theta < math.pi


def test_line_intersection_of_axis_aligned_lines():
    vertical = Line(100.0, 0.0)          # x = 100
    horizontal = Line(50.0, math.pi / 2)  # y = 50
    p = line_intersection(vertical, horizontal)
    assert p is not None
    assert p[0] == pytest.approx(100.0)
    assert p[1] == pytest.approx(50.0)


def test_parallel_lines_do_not_intersect():
    assert line_intersection(Line(10.0, 0.4), Line(80.0, 0.4)) is None


def test_split_families_separates_the_two_directions():
    lines = [
        Line(100.0, 0.02, 5), Line(300.0, 0.05, 4), Line(700.0, 0.0, 3),
        Line(400.0, math.pi / 2, 5), Line(600.0, math.pi / 2 + 0.03, 4),
    ]
    a, b = split_families(lines, PitchLineConfig())
    assert {len(a), len(b)} == {2, 3}


def test_split_families_handles_too_few_lines():
    a, b = split_families([Line(1.0, 0.1)], PitchLineConfig())
    assert len(a) == 1 and b == []


# --------------------------------------------------------------- image stages


def test_pitch_mask_finds_the_playing_surface():
    mask = pitch_mask(render_pitch())
    fraction = np.count_nonzero(mask) / mask.size
    # The rendered quad covers roughly a third of the frame.
    assert 0.2 < fraction < 0.6


def test_pitch_mask_rejects_a_crowd_shot():
    mask = pitch_mask(crowd_frame())
    assert np.count_nonzero(mask) / mask.size < 0.25


def test_line_mask_isolates_paint_not_grass():
    frame = render_pitch()
    mask = pitch_mask(frame)
    lines = line_mask(frame, mask)
    fraction = np.count_nonzero(lines) / lines.size
    # Thin markings: present, but a small share of the pitch.
    assert 0.0005 < fraction < 0.08


def test_detect_lines_finds_the_markings():
    frame = render_pitch()
    lines = detect_lines(line_mask(frame, pitch_mask(frame)))
    # 13 template lines are drawn; detection merges and may miss some at the
    # frame edge, but it should find a solid majority.
    assert len(lines) >= 6, f"only found {len(lines)} lines"


# ------------------------------------------------------- end-to-end recovery


# The pitch markings are invariant under these maps, so from lines alone a
# single frame cannot distinguish them. See ClassicalKeypointSource.set_prior.
SYMMETRIES = (
    lambda p: p,
    lambda p: (PITCH_L - p[0], p[1]),
    lambda p: (p[0], PITCH_W - p[1]),
    lambda p: (PITCH_L - p[0], PITCH_W - p[1]),
)

PROBES = [(x, y) for x in (5.0, 30.0, 52.5, 75.0, 100.0) for y in (5.0, 34.0, 63.0)]


def _errors(frame, source: ClassicalKeypointSource) -> list[float] | None:
    """Mean pitch error under each of the pitch's four symmetries, in metres."""
    pairs = source.keypoints(frame)
    if not pairs:
        return None
    cal = calibrate_frame(pairs, CalibrationConfig(ransac_threshold=2.0, min_inliers=4))
    if cal is None:
        return None

    recovered = []
    for p in PROBES:
        got = cal.to_pitch([to_image(p)])[0]
        if got is None:
            return None
        recovered.append(got)

    return [
        sum(math.dist(got, sym(p)) for got, p in zip(recovered, PROBES)) / len(PROBES)
        for sym in SYMMETRIES
    ]


def _best_error(frame, config: PitchLineConfig | None = None) -> float | None:
    """Error under the best-matching symmetry.

    Single-frame line matching recovers the pitch geometry, not which end of it
    is in view; that is resolved by a prior, tested separately below.
    """
    errs = _errors(frame, ClassicalKeypointSource(config))
    return None if errs is None else min(errs)


def test_recovers_pitch_geometry_from_a_rendered_pitch():
    error = _best_error(render_pitch())
    assert error is not None, "detector found no correspondences on a clean pitch"
    assert error < 2.0, f"mean pitch error {error:.2f} m is too large"


def test_survives_image_noise():
    error = _best_error(render_pitch(noise=6.0))
    assert error is not None, "detector failed on a mildly noisy frame"
    assert error < 3.0, f"mean pitch error {error:.2f} m under noise"


def test_survives_a_missing_line():
    # Worn paint or an occluded goal-area line.
    error = _best_error(render_pitch(missing=(5.5, 99.5)))
    assert error is not None, "detector failed with two lines missing"
    assert error < 3.0, f"mean pitch error {error:.2f} m with missing lines"


def test_a_prior_resolves_which_end_of_the_pitch_is_in_view():
    """The symmetry is real; outside information is what breaks it."""
    source = ClassicalKeypointSource()
    # The prior is image -> pitch, matching Calibration.matrix; H_TRUE is
    # pitch -> image, so it has to be inverted.
    source.set_prior(invert(H_TRUE))

    errs = _errors(render_pitch(), source)
    assert errs is not None, "detector failed with a prior set"
    # With the true camera as the prior, the identity symmetry must win
    # outright — not merely tie.
    assert errs[0] < 2.0, f"identity error {errs[0]:.2f} m"
    assert errs[0] == min(errs)


def test_a_wrong_prior_is_rejected_rather_than_forced():
    """A prior far from any real solution yields nothing, not a bad fit."""
    source = ClassicalKeypointSource()
    # A homography describing a completely different camera.
    source.set_prior((1.0, 0.0, 5000.0, 0.0, 1.0, 5000.0, 0.0, 0.0, 1.0))
    assert source.keypoints(render_pitch()) == []


def test_consecutive_frames_stay_locked_to_one_solution():
    """Once locked, the detector must not flip ends between frames."""
    source = ClassicalKeypointSource()
    frames = [render_pitch(), render_pitch(noise=4.0), render_pitch(noise=5.0)]

    picks = []
    for frame in frames:
        errs = _errors(frame, source)
        assert errs is not None, "a frame in the sequence failed"
        picks.append(min(range(4), key=lambda i: errs[i]))

    assert len(set(picks)) == 1, f"solution flipped between frames: {picks}"


def test_ambiguity_is_reported_without_a_prior():
    source = ClassicalKeypointSource()
    source.keypoints(render_pitch())
    # Not asserting it is always True — detection noise can break the tie — but
    # the flag must exist and be a bool for callers to log.
    assert isinstance(source.last_ambiguous, bool)


# --------------------------------------------------------------- abstention


def test_abstains_on_a_crowd_shot():
    source = ClassicalKeypointSource()
    assert source.keypoints(crowd_frame()) == []


def test_abstains_on_a_blank_frame():
    source = ClassicalKeypointSource()
    blank = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    assert source.keypoints(blank) == []


def test_abstains_on_bare_grass_with_no_markings():
    """Green everywhere, no lines: the pitch is visible but unusable."""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    frame[:] = (60, 140, 70)
    source = ClassicalKeypointSource()
    assert source.keypoints(frame) == []


def test_none_frame_is_handled():
    assert ClassicalKeypointSource().keypoints(None) == []


def test_reports_diagnostics_for_tuning():
    source = ClassicalKeypointSource()
    source.keypoints(render_pitch())
    assert source.last_line_count > 0
    assert source.last_support > 0.0
