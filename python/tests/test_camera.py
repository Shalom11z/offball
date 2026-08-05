"""Camera-model calibration for the broadcast centre view."""

from __future__ import annotations

import math

from offball import kernels
from offball.vision.camera import CameraModel, broadcast_prior, fit_camera

W, H = 1280, 720


def truth_camera() -> CameraModel:
    return CameraModel(
        focal=1.45 * W, x=52.5, y=-38.0, z=17.0,
        pan=math.radians(4.0), tilt=math.radians(16.0), roll=math.radians(1.5),
        cx=W / 2, cy=H / 2,
    )


def evidence(camera: CameraModel, circle=True, touch=True, half=True):
    """Image points a frame from this camera would offer."""
    h = camera.homography()

    def proj(p):
        q = kernels.project(h, [p])[0]
        return q if q and 0 < q[0] < W and 0 < q[1] < H else None

    c, t, hw = [], [], []
    if circle:
        for i in range(40):
            a = 2 * math.pi * i / 40
            q = proj((52.5 + 9.15 * math.cos(a), 34 + 9.15 * math.sin(a)))
            if q:
                c.append(q)
    if touch:
        for i in range(30):
            q = proj((10 + 85 * i / 29, 68.0))
            if q:
                t.append(q)
    if half:
        for i in range(20):
            q = proj((52.5, 5 + 58 * i / 19))
            if q:
                hw.append(q)
    return c, t, hw


def recovery_error(homography, camera: CameraModel) -> float:
    h = camera.homography()
    probes = [(x, y) for x in (10.0, 52.5, 95.0) for y in (10.0, 34.0, 58.0)]
    errs = []
    for p in probes:
        img = kernels.project(h, [p])[0]
        got = kernels.project(homography, [img])[0]
        assert got is not None
        errs.append(math.dist(got, p))
    return sum(errs) / len(errs)


def test_homography_round_trips_through_the_camera():
    cam = truth_camera()
    h = cam.homography()
    assert h is not None
    # A pitch point projected and un-projected returns to itself.
    p = (30.0, 20.0)
    img = kernels.project(h, [p])[0]
    from offball.vision.lines import _invert
    back = kernels.project(_invert(h), [img])[0]
    assert math.dist(back, p) < 1e-6


def test_recovers_a_known_camera_from_all_three_sources():
    cam = truth_camera()
    c, t, hw = evidence(cam)
    fit = fit_camera(W, H, c, t, hw)
    assert fit is not None
    assert fit.error < 0.05
    assert recovery_error(fit.homography, cam) < 0.5


def test_recovers_from_circle_and_touchline_only():
    """The halfway line is often missing; circle plus touchline should carry it."""
    cam = truth_camera()
    c, t, _ = evidence(cam, half=False)
    fit = fit_camera(W, H, c, t, [])
    assert fit is not None
    assert recovery_error(fit.homography, cam) < 3.0


def test_a_single_source_is_refused():
    """One source underdetermines the camera; a confident wrong answer is worse."""
    cam = truth_camera()
    c, _, _ = evidence(cam, touch=False, half=False)
    assert fit_camera(W, H, c, [], []) is None
    _, t, _ = evidence(cam, circle=False, half=False)
    assert fit_camera(W, H, [], t, []) is None


def test_no_evidence_at_all():
    assert fit_camera(W, H, [], [], []) is None


def test_rejects_a_fit_worse_than_max_error():
    cam = truth_camera()
    c, t, hw = evidence(cam)
    # Corrupt the circle so no camera explains the evidence.
    bad = [(x + 180 * ((i % 2) - 0.5), y) for i, (x, y) in enumerate(c)]
    assert fit_camera(W, H, bad, t, hw, max_error=0.5) is None


def test_broadcast_prior_is_a_physically_plausible_camera():
    prior = broadcast_prior(W, H)
    assert prior.z > 0, "camera must be above the pitch"
    assert prior.y < 0, "main gantry sits outside the near touchline"
    assert prior.homography() is not None


def test_fit_stays_within_physical_bounds():
    cam = truth_camera()
    c, t, hw = evidence(cam)
    fit = fit_camera(W, H, c, t, hw)
    assert fit is not None
    assert fit.camera.z > 0, "solved camera must not be underground"
    assert fit.camera.focal > 0
