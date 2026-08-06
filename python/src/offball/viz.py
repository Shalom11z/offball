"""Visual debugging: see what the pipeline believes.

A coverage figure tells you *that* calibration failed, never *why*. These
renderers put the model's beliefs back on top of the pixels, which is the only
practical way to tell a wrong homography from a missed detection from a
mis-assigned team.

Two views, answering different questions:

:func:`overlay_calibration`
    Draws the pitch template through the fitted homography onto the frame. If
    the drawn lines sit on the painted ones, calibration is right. This is the
    single most useful diagnostic in the project, and the first thing to look at
    on any new footage.

:func:`render_pitch_map`
    A top-down view of where the model thinks everyone is, optionally shaded by
    pitch control. This is what catches a player projected into the stands, or
    a team assignment that has flipped.

Both return BGR arrays; nothing here writes files or opens windows, so they
compose into video, notebooks or the CLI.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .kernels import ControlGrid, ControlParams, pitch_control
from .types import FrameState, Point, Team
from .vision.lines import _invert

__all__ = [
    "TEAM_COLOURS",
    "overlay_calibration",
    "render_frame_pair",
    "render_pitch_map",
]

#: BGR, chosen to stay distinguishable in greyscale print and to avoid
#: red/green, which is the most common form of colour blindness.
TEAM_COLOURS: dict[Team, tuple[int, int, int]] = {
    Team.HOME: (220, 130, 40),    # blue
    Team.AWAY: (60, 200, 240),    # amber
    Team.REFEREE: (200, 200, 200),
    Team.UNKNOWN: (130, 130, 130),
}
_BALL_COLOUR = (255, 255, 255)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "visualisation needs the 'calibration' extra: "
            "pip install 'offball[vision]'"
        ) from exc
    return cv2


def _segment(a: Point, b: Point, samples: int) -> list[Point]:
    """A straight pitch segment, densely sampled.

    Sampled rather than drawn end to end because the projection of a long line
    can pass near the horizon, where a two-point draw would cut the corner
    badly. Sampling keeps the drawn line on the true projection.
    """
    return [
        (a[0] + (b[0] - a[0]) * i / samples, a[1] + (b[1] - a[1]) * i / samples)
        for i in range(samples + 1)
    ]


def _pitch_polylines(
    pitch_length: float, pitch_width: float, samples: int = 12
) -> list[list[Point]]:
    """The **actual painted markings** of a pitch, as polylines.

    Real markings, not the extended lines the detector matches against. A pitch
    has no line running the full width at x=5.5; drawing one puts green ink
    across grass that has none, and the overlay is only useful if a drawn line
    landing on bare turf means something is wrong.

    Includes the centre circle and both penalty arcs, which the straight-line
    template omits but which are strong visual anchors for judging alignment.
    """
    length, width = pitch_length, pitch_width
    half_w = width / 2.0
    lines: list[list[Point]] = [
        # Touchlines, goal lines, halfway line.
        _segment((0.0, 0.0), (length, 0.0), samples),
        _segment((0.0, width), (length, width), samples),
        _segment((0.0, 0.0), (0.0, width), samples),
        _segment((length, 0.0), (length, width), samples),
        _segment((length / 2, 0.0), (length / 2, width), samples),
    ]

    # Penalty and goal areas: three segments each, at both ends.
    for depth, half_span in ((16.5, 20.16), (5.5, 9.16)):
        for near, far in ((0.0, depth), (length, length - depth)):
            lines.append(_segment((near, half_w - half_span), (far, half_w - half_span), samples))
            lines.append(_segment((near, half_w + half_span), (far, half_w + half_span), samples))
            lines.append(_segment((far, half_w - half_span), (far, half_w + half_span), samples))

    def arc(cx: float, cy: float, radius: float, start: float, end: float) -> list[Point]:
        steps = max(8, samples * 2)
        return [
            (
                cx + radius * math.cos(start + (end - start) * i / steps),
                cy + radius * math.sin(start + (end - start) * i / steps),
            )
            for i in range(steps + 1)
        ]

    lines.append(arc(length / 2, half_w, 9.15, 0.0, 2 * math.pi))

    # Penalty arcs: the part of the 9.15m circle around the penalty spot that
    # falls outside the penalty area.
    for spot_x, direction in ((11.0, 1.0), (length - 11.0, -1.0)):
        inside = math.acos(min(1.0, max(-1.0, (16.5 - 11.0) / 9.15)))
        start = -inside if direction > 0 else math.pi - inside
        end = inside if direction > 0 else math.pi + inside
        lines.append(arc(spot_x, half_w, 9.15, start, end))

    return lines


def overlay_calibration(
    frame,
    homography: Sequence[float] | None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    colour: tuple[int, int, int] = (0, 240, 0),
    thickness: int = 2,
    label: str | None = None,
):
    """Draw the pitch template onto a frame through its homography.

    Args:
        frame: BGR image. Not modified; a copy is returned.
        homography: **image -> pitch** transform, as produced by
            :attr:`~offball.vision.calibration.Calibration.matrix`. ``None``
            marks the frame as uncalibrated rather than raising, because that
            is exactly the case worth looking at.
        label: Optional caption, e.g. the frame index and support score.

    Returns:
        A new BGR image. Where the drawn lines land on the painted ones, the
        calibration is correct.
    """
    cv2 = _require_cv2()
    from .kernels import project

    out = frame.copy()
    height, width = out.shape[:2]

    if homography is None:
        cv2.putText(
            out, "UNCALIBRATED", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        )
        if label:
            cv2.putText(
                out, label, (24, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        return out

    inverse = _invert(tuple(homography))
    if inverse is None:
        return out

    for polyline in _pitch_polylines(pitch_length, pitch_width):
        projected = project(inverse, polyline)
        previous = None
        for point in projected:
            if point is None:
                previous = None
                continue
            current = (round(point[0]), round(point[1]))
            # Segments far outside the frame come from near-horizon points and
            # produce enormous coordinates; drawing them corrupts the image.
            if previous is not None and _on_screen(previous, current, width, height):
                cv2.line(out, previous, current, colour, thickness, cv2.LINE_AA)
            previous = current

    if label:
        cv2.putText(out, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    return out


def _on_screen(a: tuple[int, int], b: tuple[int, int], width: int, height: int) -> bool:
    """Whether a segment is close enough to the frame to be worth drawing."""
    margin = 4 * max(width, height)
    return all(-margin < x < margin and -margin < y < margin for x, y in (a, b))


def render_pitch_map(
    state: FrameState,
    control: ControlGrid | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    scale: int = 10,
    show_control: bool = False,
):
    """Top-down view of one frame's reconstructed positions.

    Args:
        state: The frame to draw.
        control: Precomputed control field. When ``None`` and ``show_control``
            is set, it is computed here.
        scale: Pixels per metre.
        show_control: Shade the pitch by attacking-team control.

    Returns:
        A BGR image ``pitch_length * scale`` by ``pitch_width * scale``.
    """
    cv2 = _require_cv2()
    import numpy as np

    w, h = int(pitch_length * scale), int(pitch_width * scale)
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas[:] = (50, 110, 60)

    def to_px(p: Point) -> tuple[int, int]:
        # Flip y so the image reads like a broadcast view rather than being
        # upside down relative to the pitch coordinate system.
        return (int(p[0] * scale), int((pitch_width - p[1]) * scale))

    if show_control:
        if control is None:
            attackers = (
                state.team(state.attacking_team) if state.attacking_team else ()
            )
            defenders = (
                state.team(state.attacking_team.opponent())
                if state.attacking_team
                else ()
            )
            positions = [p.pitch_xy for p in attackers + defenders]
            velocities = [p.velocity or (0.0, 0.0) for p in attackers + defenders]
            flags = [True] * len(attackers) + [False] * len(defenders)
            if positions:
                control = pitch_control(
                    positions, velocities, flags, pitch_length, pitch_width,
                    52, 34, ControlParams(),
                )
        if control is not None:
            field = np.array(control.values, np.float32).reshape(control.ny, control.nx)
            field = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
            field = np.flipud(field)
            # Blue where the attacking team controls, amber where it does not.
            tint = np.zeros_like(canvas, np.float32)
            tint[..., 0] = 255 * field
            tint[..., 2] = 255 * (1.0 - field)
            canvas = cv2.addWeighted(canvas.astype(np.float32), 0.6, tint, 0.4, 0).astype(
                np.uint8
            )

    # Pitch markings.
    for polyline in _pitch_polylines(pitch_length, pitch_width):
        points = np.array([to_px(p) for p in polyline], np.int32)
        cv2.polylines(canvas, [points], False, (235, 235, 235), 1, cv2.LINE_AA)

    for player in state.players:
        if player.pitch_xy is None:
            continue
        centre = to_px(player.pitch_xy)
        colour = TEAM_COLOURS.get(player.team, TEAM_COLOURS[Team.UNKNOWN])
        cv2.circle(canvas, centre, max(3, scale // 2), colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, centre, max(3, scale // 2), (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(
            canvas, str(player.track_id),
            (centre[0] + 7, centre[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (245, 245, 245), 1, cv2.LINE_AA,
        )
        # Velocity as a one-second lead line: the direction the model thinks
        # the player is heading, which is what the control model acts on.
        if player.velocity and (abs(player.velocity[0]) + abs(player.velocity[1])) > 0.5:
            tip = to_px(
                (
                    player.pitch_xy[0] + player.velocity[0],
                    player.pitch_xy[1] + player.velocity[1],
                )
            )
            cv2.arrowedLine(canvas, centre, tip, colour, 2, cv2.LINE_AA, tipLength=0.3)

    if state.ball_pitch_xy is not None:
        cv2.circle(canvas, to_px(state.ball_pitch_xy), max(3, scale // 3), _BALL_COLOUR, -1)
        cv2.circle(canvas, to_px(state.ball_pitch_xy), max(3, scale // 3), (0, 0, 0), 1)

    caption = f"frame {state.frame_index}  t={state.timestamp:.1f}s"
    if state.attacking_team is not None:
        caption += f"  attacking: {state.attacking_team.value}"
    cv2.putText(
        canvas, caption, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    return canvas


def render_frame_pair(
    frame,
    state: FrameState,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    show_control: bool = False,
):
    """The camera view with its overlay, beside the top-down map.

    The two together answer "is this frame right?" faster than either alone: the
    overlay shows whether the homography is correct, the map shows what that
    homography produced.
    """
    cv2 = _require_cv2()
    import numpy as np

    left = overlay_calibration(
        frame,
        state.homography,
        pitch_length,
        pitch_width,
        label=f"frame {state.frame_index}",
    )
    right = render_pitch_map(
        state, None, pitch_length, pitch_width, show_control=show_control
    )

    height = max(left.shape[0], right.shape[0])
    def pad(img):
        if img.shape[0] == height:
            return img
        out = np.zeros((height, img.shape[1], 3), np.uint8)
        out[: img.shape[0]] = img
        return out

    scale = height / right.shape[0]
    right = cv2.resize(right, (int(right.shape[1] * scale), height))
    return np.hstack([pad(left), pad(right)])
