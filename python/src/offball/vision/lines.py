"""Classical pitch-line detection: a homography without a trained model.

This is the no-data path to camera calibration. It finds the painted lines with
ordinary image processing, matches them to the known pitch template, and hands
the resulting correspondences to the existing
:func:`~offball.vision.calibration.calibrate_frame`.

It is **not** as good as a learned keypoint model. It wants a clean, wide
broadcast shot with visible markings; it degrades on worn pitches, hard
shadows, heavy crowd bleed and tight framing. What it buys is an end-to-end
system today, and a baseline to measure a learned model against later. See
``docs/06-roadmap.md``.

## Why match lines rather than corners

A homography maps lines to lines. So the image of the intersection of two pitch
lines is exactly the intersection of their images — whether or not any paint
exists at that point. Matching *lines* and deriving points from them is far more
robust than detecting corners directly, because a line is supported by thousands
of pixels while a corner is supported by a handful, and most useful pitch
intersections (halfway line crossed with a penalty-area edge) are not painted at
all.

## Pipeline

1. Mask the pitch (green, largest connected region).
2. Isolate line pixels inside it (bright, thin — a morphological top-hat).
3. Hough transform for infinite lines; merge near-duplicates.
4. Split into two families: parallel to the touchlines, and parallel to the
   goal lines.
5. Search order-preserving assignments of each family to the template, fit a
   homography from the implied intersections, and score each candidate by how
   well the whole projected template lands on real line pixels.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from ..kernels import fit_dlt, project
from ..types import Point

__all__ = [
    "TEMPLATE_X",
    "TEMPLATE_Y",
    "ClassicalKeypointSource",
    "Line",
    "PitchLineConfig",
    "detect_lines",
    "line_intersection",
    "line_mask",
    "pitch_mask",
    "split_families",
]

# Pitch lines of constant x (parallel to the goal lines), as fractions handled
# in metres for a standard pitch. Distances are fixed by the laws of the game.
TEMPLATE_X: tuple[float, ...] = (0.0, 5.5, 16.5, 52.5, 88.5, 99.5, 105.0)

# Pitch lines of constant y (parallel to the touchlines).
#   penalty area is 40.32m wide -> 34 +/- 20.16
#   goal area is 18.32m wide    -> 34 +/- 9.16
TEMPLATE_Y: tuple[float, ...] = (0.0, 13.84, 24.84, 43.16, 54.16, 68.0)


@dataclass(frozen=True, slots=True)
class Line:
    """An infinite image line in Hough normal form: ``x cosθ + y sinθ = ρ``."""

    rho: float
    theta: float
    #: Hough accumulator votes; used to rank lines, not compared across frames.
    strength: float = 0.0

    def normalised(self) -> Line:
        """Canonical form with ``theta`` in [0, π).

        Hough returns (ρ, θ) and (−ρ, θ+π) for the same line; without
        normalising, merging duplicates silently fails.
        """
        rho, theta = self.rho, self.theta
        theta = theta % (2 * math.pi)
        if theta >= math.pi:
            theta -= math.pi
            rho = -rho
        return Line(rho, theta, self.strength)

    def offset_at(self, cx: float, cy: float) -> float:
        """Signed perpendicular distance from a reference point.

        Used to order the lines within a family, which is what makes the
        order-preserving assignment search valid.
        """
        return self.rho - (cx * math.cos(self.theta) + cy * math.sin(self.theta))


@dataclass(frozen=True, slots=True)
class PitchLineConfig:
    # --- pitch mask -------------------------------------------------------
    #: HSV hue range for grass. OpenCV hue is 0-179.
    hue_range: tuple[int, int] = (30, 90)
    min_saturation: int = 40
    min_value: int = 40
    #: Reject the frame outright if the pitch covers less of it than this.
    #: Filters close-ups and crowd shots before any expensive work.
    min_pitch_fraction: float = 0.25

    # --- line mask --------------------------------------------------------
    #: Structuring-element size for the top-hat that isolates thin bright
    #: features. Must exceed the line width in pixels.
    tophat_kernel: int = 13
    line_threshold: int = 30

    # --- Hough ------------------------------------------------------------
    hough_threshold: int = 120
    hough_rho: float = 1.0
    hough_theta_steps: int = 360
    #: Two lines closer than both tolerances (offset measured from the image
    #: centre) are treated as one. Generous, because anti-aliased paint yields
    #: several Hough peaks per marking: the closest genuinely distinct pitch
    #: lines are 11m apart, which is 100+ px in any usable shot.
    merge_rho: float = 45.0
    merge_theta: float = math.radians(8.0)
    #: Strongest lines kept before the family split, to keep weak spurious
    #: peaks from skewing the angle clustering.
    max_total_lines: int = 16
    #: Strongest N lines kept per family. Raising this grows the assignment
    #: search combinatorially.
    max_lines_per_family: int = 4

    # --- matching ---------------------------------------------------------
    #: A candidate must place this fraction of its sampled template points on
    #: real line pixels to be accepted.
    min_support: float = 0.55
    #: Pixels within which a projected template point counts as supported.
    support_tolerance: float = 12.0
    #: Two candidates within this much support of each other count as a tie.
    ambiguity_margin: float = 0.05
    #: Mean image-space distance, in pixels, within which a candidate is
    #: considered consistent with the prior homography.
    prior_tolerance: float = 120.0
    pitch_length: float = 105.0
    pitch_width: float = 68.0


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "classical line detection needs the 'vision' extra: "
            "pip install 'offball[vision]'"
        ) from exc
    return cv2


def pitch_mask(frame, config: PitchLineConfig | None = None):
    """Binary mask of the playing surface.

    Takes the largest connected green region rather than every green pixel, so
    that grass visible beyond the advertising boards does not join the mask.
    """
    cv2 = _require_cv2()
    import numpy as np

    config = config or PitchLineConfig()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = np.array([config.hue_range[0], config.min_saturation, config.min_value], np.uint8)
    hi = np.array([config.hue_range[1], 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    # Additionally require green to actually dominate in BGR. Hue alone admits
    # desaturated browns and greys that a crowd is full of.
    b, g, r = (frame[:, :, i].astype(np.int16) for i in range(3))
    dominant = ((g > r * 1.05) & (g > b * 1.05)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, dominant)

    # Open first: speckle from a crowd or advertising hoarding is green-ish in
    # places, and without this the close below welds it into a solid region.
    # The pitch is a large contiguous area and barely notices.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    # Then close over players and painted lines so the pitch becomes one region.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest).astype(np.uint8)) * 255


def line_mask(frame, mask=None, config: PitchLineConfig | None = None):
    """Binary mask of painted line pixels within the pitch region.

    A white top-hat keeps features thinner than the structuring element and
    removes the slow brightness gradient across a floodlit pitch, which a plain
    threshold cannot handle.
    """
    cv2 = _require_cv2()

    config = config or PitchLineConfig()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Suppress sensor noise before the top-hat. Without this, per-pixel noise
    # survives the morphology as speckle, and Hough happily finds strong
    # "lines" through it that have nothing to do with the pitch.
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    k = config.tophat_kernel | 1  # must be odd
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)
    _, lines = cv2.threshold(tophat, config.line_threshold, 255, cv2.THRESH_BINARY)
    if mask is not None:
        lines = cv2.bitwise_and(lines, lines, mask=mask)
    return lines


def _merge(
    lines: list[Line], config: PitchLineConfig, cx: float, cy: float
) -> list[Line]:
    """Collapse near-duplicate Hough peaks, keeping the strongest of each.

    Separation is measured from the **image centre**, not from the origin.
    Comparing raw rho does not work: rho is the perpendicular offset from
    (0, 0), so for a line far from the origin a fraction of a degree of theta
    shifts rho by tens of pixels, and genuine duplicates fail to merge. The
    same two lines differ by a few pixels when referenced to the centre of the
    image they were found in.
    """
    span = 0.7 * math.hypot(cx, cy)
    merged: list[Line] = []
    for line in sorted(lines, key=lambda ln: -ln.strength):
        duplicate = False
        for kept in merged:
            dt = abs(line.theta - kept.theta)
            dt = min(dt, math.pi - dt)  # theta wraps at pi
            if dt <= config.merge_theta and _separation(line, kept, span) <= config.merge_rho:
                duplicate = True
                break
        if not duplicate:
            merged.append(line)
    return merged


def _separation(a: Line, b: Line, span: float) -> float:
    """Mean distance between two lines over the visible image extent.

    Comparing a single scalar offset is not enough: two lines can share an
    offset at the reference point and diverge badly across the frame, or differ
    there while being the same marking. Sampling along one line and measuring
    to the other answers the question that actually matters — are these the
    same painted line where we can see it?
    """
    dirx, diry = -math.sin(a.theta), math.cos(a.theta)
    px, py = a.rho * math.cos(a.theta), a.rho * math.sin(a.theta)
    cosb, sinb = math.cos(b.theta), math.sin(b.theta)
    total = 0.0
    offsets = (-span, 0.0, span)
    for t in offsets:
        x, y = px + dirx * t, py + diry * t
        total += abs(x * cosb + y * sinb - b.rho)
    return total / len(offsets)


def detect_lines(lines_mask, config: PitchLineConfig | None = None) -> list[Line]:
    """Hough-transform a line mask into merged infinite lines.

    Standard rather than probabilistic Hough: pitch markings are long and
    straight, and infinite lines are what the intersection step needs.
    """
    cv2 = _require_cv2()

    config = config or PitchLineConfig()
    raw = cv2.HoughLines(
        lines_mask,
        config.hough_rho,
        math.pi / config.hough_theta_steps,
        config.hough_threshold,
    )
    if raw is None:
        return []
    # OpenCV returns peaks in descending vote order; use rank as strength.
    found = [
        Line(float(r[0][0]), float(r[0][1]), strength=float(len(raw) - i)).normalised()
        for i, r in enumerate(raw)
    ]
    height, width = lines_mask.shape[:2]
    merged = _merge(found, config, width / 2.0, height / 2.0)
    # Keep only the strongest: weak peaks are usually anti-aliasing ghosts, and
    # letting them into the family split drags the angle clusters around.
    return sorted(merged, key=lambda line: -line.strength)[: config.max_total_lines]


def line_intersection(a: Line, b: Line) -> Point | None:
    """Intersection of two infinite lines, or ``None`` when near-parallel."""
    det = math.sin(b.theta - a.theta)
    if abs(det) < 1e-6:
        return None
    x = (a.rho * math.sin(b.theta) - b.rho * math.sin(a.theta)) / det
    y = (b.rho * math.cos(a.theta) - a.rho * math.cos(b.theta)) / det
    return (x, y)


def split_families(lines: list[Line], config: PitchLineConfig) -> tuple[list[Line], list[Line]]:
    """Split lines into the two pitch directions by angle.

    1-D k-means on theta, on the circle (theta and theta+pi are the same
    direction), seeded from the two most dissimilar angles so the split is
    deterministic.

    Returns ``(family_a, family_b)``, each ordered by signed offset from the
    image centre. That ordering is what licenses the order-preserving
    assignment search: a projective map of a plane preserves the order of a
    pencil of parallel lines.
    """
    if len(lines) < 2:
        return (list(lines), [])

    def circ_dist(t1: float, t2: float) -> float:
        d = abs(t1 - t2) % math.pi
        return min(d, math.pi - d)

    # Seed with the two most dissimilar directions.
    best = (0, 1)
    best_d = -1.0
    for i, j in itertools.combinations(range(len(lines)), 2):
        d = circ_dist(lines[i].theta, lines[j].theta)
        if d > best_d:
            best_d, best = d, (i, j)
    centres = [lines[best[0]].theta, lines[best[1]].theta]

    labels = [0] * len(lines)
    for _ in range(20):
        changed = False
        for idx, line in enumerate(lines):
            lab = 0 if circ_dist(line.theta, centres[0]) <= circ_dist(line.theta, centres[1]) else 1
            if lab != labels[idx]:
                labels[idx], changed = lab, True
        for c in (0, 1):
            members = [ln.theta for ln, lab in zip(lines, labels, strict=True) if lab == c]
            if members:
                # Circular mean over the half-circle: double the angle, average,
                # halve. Averaging raw angles breaks across the 0/pi seam.
                sx = sum(math.sin(2 * t) for t in members)
                cx = sum(math.cos(2 * t) for t in members)
                centres[c] = (math.atan2(sx, cx) / 2.0) % math.pi
        if not changed:
            break

    fam_a = [ln for ln, lab in zip(lines, labels, strict=True) if lab == 0]
    fam_b = [ln for ln, lab in zip(lines, labels, strict=True) if lab == 1]
    return fam_a, fam_b


def _aligned(lines: list[Line]) -> list[Line]:
    """Give every line in a family a consistently-oriented normal.

    ``normalised()`` puts theta in [0, π), which splits a family that straddles
    the seam: a line at 177° and one at 9.5° are nearly parallel, but their
    normals point in near-opposite directions, so their signed offsets have
    opposite signs and the family cannot be put in spatial order.

    Since (ρ, θ) and (−ρ, θ±π) denote the same line, flipping the odd ones out
    onto the family's mean direction is free, and it is what makes the ordering
    — and therefore the order-preserving assignment search — valid.
    """
    if len(lines) < 2:
        return list(lines)

    # Circular mean over the doubled angle: directions live on the half-circle,
    # so averaging theta directly breaks across the 0/pi seam.
    sx = sum(math.sin(2 * ln.theta) for ln in lines)
    cxs = sum(math.cos(2 * ln.theta) for ln in lines)
    ref = math.atan2(sx, cxs) / 2.0

    out: list[Line] = []
    for ln in lines:
        delta = (ln.theta - ref) % (2 * math.pi)
        if delta > math.pi:
            delta -= 2 * math.pi
        if abs(delta) > math.pi / 2:
            # Antiparallel normal: flip to the equivalent representation.
            theta = ln.theta - math.pi if ln.theta >= math.pi / 2 else ln.theta + math.pi
            out.append(Line(-ln.rho, theta, ln.strength))
        else:
            out.append(ln)
    return out


def _ordered(lines: list[Line], cx: float, cy: float, limit: int) -> list[Line]:
    """Strongest `limit` lines, returned in spatial order."""
    strongest = sorted(lines, key=lambda ln: -ln.strength)[:limit]
    return sorted(_aligned(strongest), key=lambda ln: ln.offset_at(cx, cy))


@dataclass(frozen=True, slots=True)
class _Candidate:
    homography: tuple[float, ...]
    correspondences: list[tuple[Point, Point]]
    support: float


class ClassicalKeypointSource:
    """A :class:`~offball.vision.calibration.KeypointSource` with no model.

    Usage mirrors any other keypoint source::

        from offball.pipeline import Pipeline
        from offball.vision.lines import ClassicalKeypointSource

        pipeline = Pipeline(detector=..., keypoints=ClassicalKeypointSource())

    Returns ``(image_point, pitch_point)`` correspondences, which
    :func:`calibrate_frame` then re-fits under RANSAC with its own quality
    gates. The duplication is deliberate: this class is free to be optimistic,
    and the calibration stage stays the single place quality is enforced.

    Returns an empty list — never a guess — when the pitch is not visible, too
    few lines are found, or no candidate clears ``min_support``.
    """

    def __init__(
        self, config: PitchLineConfig | None = None, prior: tuple[float, ...] | None = None
    ) -> None:
        self.config = config or PitchLineConfig()
        self._prior = prior
        #: Diagnostics for the last call, for tuning on real footage.
        self.last_support: float = 0.0
        self.last_line_count: int = 0
        #: True when more than one candidate scored within `ambiguity_margin`
        #: of the winner — almost always the pitch's own symmetry (see
        #: :meth:`set_prior`). Worth logging when calibration goes wrong.
        self.last_ambiguous: bool = False

    def set_prior(self, homography: tuple[float, ...] | None) -> None:
        """Constrain the search to solutions near a known homography.

        ``homography`` maps **image pixels to pitch metres**, matching
        :attr:`~offball.vision.calibration.Calibration.matrix`, so a previous
        frame's calibration can be fed straight back in.

        **The pitch is symmetric, and this is how that gets resolved.**

        The markings are invariant under ``x -> length - x`` and
        ``y -> width - y``: the two penalty areas are identical, and the
        template line spacings mirror exactly. A homography absorbs the
        resulting rotation perfectly, so *from lines alone in a single frame
        there is no way to tell which end of the pitch is in view*. Every
        candidate in that symmetry group fits the image equally well.

        That ambiguity is a property of the pitch, not a limitation of this
        detector, and no amount of image processing removes it. It is resolved
        with outside information: the previous frame's homography (what
        :class:`~offball.pipeline.Pipeline` supplies), or the known camera side
        and the period's attacking direction.

        Passing ``None`` clears the prior and returns to a free search.
        """
        self._prior = homography

    def reset(self) -> None:
        self._prior = None

    # -- scoring ----------------------------------------------------------

    def _plausible(self, homography: tuple[float, ...], shape) -> bool:
        """Cheap sanity gate before the expensive support scan.

        The assignment search generates thousands of candidates, most of them
        nonsense. Projecting four corners and checking the result still looks
        like a pitch rejects the bulk of them for a fraction of the cost.
        """
        cfg = self.config
        h_inv = _invert(homography)
        if h_inv is None:
            return False
        corners = project(
            h_inv,
            [
                (0.0, 0.0),
                (cfg.pitch_length, 0.0),
                (cfg.pitch_length, cfg.pitch_width),
                (0.0, cfg.pitch_width),
            ],
        )
        if any(c is None for c in corners):
            return False

        height, width = shape[:2]
        # Must be convex and traversed consistently: a valid perspective view
        # of a rectangle never folds over on itself.
        sign = 0
        area = 0.0
        for i in range(4):
            ax, ay = corners[i]
            bx, by = corners[(i + 1) % 4]
            cx2, cy2 = corners[(i + 2) % 4]
            cross = (bx - ax) * (cy2 - by) - (by - ay) * (cx2 - bx)
            if cross == 0:
                return False
            s = 1 if cross > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                return False
            area += ax * by - bx * ay
        area = abs(area) / 2.0

        # The pitch should occupy a sane share of the frame: a candidate that
        # maps it to a sliver or to something vastly larger than the image is
        # a mis-assignment.
        frame_area = float(width * height)
        return 0.05 * frame_area <= area <= 12.0 * frame_area

    def _support(self, homography: tuple[float, ...], distance_map, shape) -> float:
        """Fraction of the projected template that lands on real line pixels.

        Scoring against the dense line mask, rather than against the handful of
        intersections used to fit, is what makes this robust: a wrong
        assignment can fit its own four points perfectly and still put the rest
        of the pitch nowhere near any paint.
        """
        cfg = self.config
        h_inv = _invert(homography)
        if h_inv is None:
            return 0.0

        height, width = shape[:2]
        samples: list[Point] = []
        for x in TEMPLATE_X:
            samples += [(x, cfg.pitch_width * f / 8.0) for f in range(9)]
        for y in TEMPLATE_Y:
            samples += [(cfg.pitch_length * f / 8.0, y) for f in range(9)]

        projected = project(h_inv, samples)
        hits = 0
        visible = 0
        for p in projected:
            if p is None:
                continue
            px, py = p
            if not (0 <= px < width and 0 <= py < height):
                continue
            visible += 1
            if distance_map[int(py), int(px)] <= cfg.support_tolerance:
                hits += 1

        # A candidate that puts almost the whole pitch outside the frame is not
        # well-supported, however good its few visible points look.
        if visible < len(samples) * 0.25:
            return 0.0
        return hits / visible

    # -- matching ---------------------------------------------------------

    def _candidates(
        self, fam_a: list[Line], fam_b: list[Line], distance_map, shape
    ) -> _Candidate | None:
        """Search order-preserving template assignments for the best fit."""
        cfg = self.config
        best: _Candidate | None = None
        scored: list[float] = []

        # split_families returns the two families in arbitrary order, so try
        # each in the const-x role. `a_lines` always matches TEMPLATE_X and
        # `b_lines` always TEMPLATE_Y, which keeps the pitch point below
        # unambiguous — swapping the templates instead silently transposes the
        # coordinates.
        for a_lines, b_lines in ((fam_a, fam_b), (fam_b, fam_a)):
            a_tmpl, b_tmpl = TEMPLATE_X, TEMPLATE_Y
            if len(a_lines) < 2 or len(b_lines) < 2:
                continue

            for a_pick in itertools.combinations(a_tmpl, len(a_lines)):
                for b_pick in itertools.combinations(b_tmpl, len(b_lines)):
                    # Detected order may run either way along the pencil.
                    for a_seq in (a_pick, a_pick[::-1]):
                        for b_seq in (b_pick, b_pick[::-1]):
                            pairs: list[tuple[Point, Point]] = []
                            for la, xa in zip(a_lines, a_seq, strict=True):
                                for lb, yb in zip(b_lines, b_seq, strict=True):
                                    ip = line_intersection(la, lb)
                                    if ip is None:
                                        continue
                                    # The virtual intersection is valid even
                                    # where no paint exists; see module docs.
                                    pairs.append((ip, (xa, yb)))
                            if len(pairs) < 4:
                                continue
                            try:
                                # Every pair is consistent by construction —
                                # they are intersections of the same assigned
                                # lines — so this is a plain least-squares fit.
                                # RANSAC here would be worse than useless: on a
                                # 4x4 grid of intersections a small sample is
                                # frequently collinear, and the fit fails.
                                h = fit_dlt(
                                    [p[0] for p in pairs], [p[1] for p in pairs]
                                )
                            except ValueError:
                                continue
                            if not self._plausible(h, shape):
                                continue
                            if not self._agrees_with_prior(h, shape):
                                continue
                            support = self._support(h, distance_map, shape)
                            scored.append(support)
                            if best is None or support > best.support:
                                best = _Candidate(h, pairs, support)

        if best is None or best.support < cfg.min_support:
            return None

        # Flag near-ties. With no prior these are overwhelmingly the pitch's own
        # symmetry group, and the winner among them is arbitrary.
        rivals = sum(1 for s in scored if s >= best.support - cfg.ambiguity_margin)
        self.last_ambiguous = rivals > 1
        return best

    def _agrees_with_prior(self, homography: tuple[float, ...], shape) -> bool:
        """Whether a candidate is close to the prior, if one is set."""
        if self._prior is None:
            return True
        cfg = self.config
        probes = [
            (0.0, 0.0),
            (cfg.pitch_length, 0.0),
            (cfg.pitch_length, cfg.pitch_width),
            (0.0, cfg.pitch_width),
            (cfg.pitch_length / 2, cfg.pitch_width / 2),
        ]
        a_inv, b_inv = _invert(homography), _invert(self._prior)
        if a_inv is None or b_inv is None:
            return False
        got, want = project(a_inv, probes), project(b_inv, probes)
        total = 0.0
        for p, q in zip(got, want, strict=True):
            if p is None or q is None:
                return False
            total += math.dist(p, q)
        return total / len(probes) <= cfg.prior_tolerance

    # -- public API -------------------------------------------------------

    def keypoints(self, frame) -> list[tuple[Point, Point]]:
        """Find image/pitch correspondences in one frame."""
        cv2 = _require_cv2()
        import numpy as np

        self.last_support = 0.0
        self.last_line_count = 0
        self.last_ambiguous = False
        if frame is None:
            return []

        cfg = self.config
        mask = pitch_mask(frame, cfg)
        if float(np.count_nonzero(mask)) / mask.size < cfg.min_pitch_fraction:
            # A close-up or crowd shot. Report nothing rather than guessing.
            return []

        lines_img = line_mask(frame, mask, cfg)
        lines = detect_lines(lines_img, cfg)
        self.last_line_count = len(lines)
        if len(lines) < 4:
            return []

        fam_a, fam_b = split_families(lines, cfg)
        height, width = lines_img.shape[:2]
        cx, cy = width / 2.0, height / 2.0
        fam_a = _ordered(fam_a, cx, cy, cfg.max_lines_per_family)
        fam_b = _ordered(fam_b, cx, cy, cfg.max_lines_per_family)

        # Distance to the nearest line pixel, so scoring is a lookup.
        distance_map = cv2.distanceTransform(
            cv2.bitwise_not(lines_img), cv2.DIST_L2, 3
        )

        best = self._candidates(fam_a, fam_b, distance_map, lines_img.shape)
        if best is None:
            return []

        self.last_support = best.support
        # Lock onto this solution so the next frame resolves the pitch's
        # symmetry the same way instead of flipping ends.
        self._prior = best.homography
        return best.correspondences


def _invert(h: tuple[float, ...]) -> tuple[float, ...] | None:
    """Invert a flat 3x3 homography, or None if singular."""
    m = h
    c = (
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    )
    det = m[0] * c[0] + m[1] * c[3] + m[2] * c[6]
    if abs(det) < 1e-12:
        return None
    return tuple(v / det for v in c)
