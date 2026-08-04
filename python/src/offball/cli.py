"""Command-line entry point.

::

    offball demo                      # run the synthetic sequence, print a report
    offball demo --json               # machine-readable
    offball analyse match.mp4         # analyse real footage (needs the vision extra)
    offball serve                     # start the HTTP API
    offball info                      # environment and backend check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .kernels import BACKEND


def _cmd_info(args: argparse.Namespace) -> int:
    print(f"offball {__version__}")
    print(f"kernel backend: {BACKEND}")
    if BACKEND == "python":
        print(
            "  note: the compiled kernels are not installed, so the pitch-control\n"
            "  grid runs in pure Python — correct, but far slower. Build them with:\n"
            "    cd rust/offball-core && maturin develop --release --features python"
        )
    for name, extra in (("cv2", "vision"), ("ultralytics", "vision"), ("fastapi", "api")):
        try:
            __import__(name)
            print(f"  {name}: available")
        except ImportError:
            print(f"  {name}: missing (pip install 'offball[{extra}]')")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import SyntheticMatch, synthetic_frames
    from .tactics.offball import score_frame
    from .tactics.report import build_report

    match = SyntheticMatch(frames=args.frames)
    states = synthetic_frames(match)
    scores = [s for s in (score_frame(st) for st in states) if s is not None]
    report = build_report(scores, len(states), match.fps)

    if args.json:
        json.dump(report.to_dict(), sys.stdout, indent=2)
        print()
        return 0

    print(f"Scored {report.frames_scored}/{report.frames_total} frames "
          f"(coverage {report.coverage:.0%})\n")

    for team in report.teams:
        print(f"[{team.team.value}] in possession for {team.duration:.1f}s")
        print(f"  controlled space      {team.median_controlled_space:8.0f} m^2 (median)")
        print(f"  dangerous space       {team.median_dangerous_space:8.0f}     (threat-weighted)")
        print(f"  attacking block area  {team.median_attacking_hull:8.0f} m^2")
        print(f"  defending block area  {team.median_defending_hull:8.0f} m^2")
        print(f"  passing options       {team.mean_passing_options:8.1f} (mean)")
        print()

    print(f"{'track':>6} {'frames':>7} {'space':>8} {'avail':>7} {'offside':>8} "
          f"{'margin':>8} {'lines':>6} {'press':>6}")
    for p in report.players:
        margin = "  n/a" if p.median_offside_margin is None else f"{p.median_offside_margin:6.1f}m"
        print(
            f"{p.track_id:>6} {p.frames:>7} {p.median_space_owned:7.0f}m² "
            f"{p.availability_rate:6.0%} {p.offside_rate:7.0%} {margin:>8} "
            f"{p.mean_lines_broken:6.2f} {p.mean_pressure:6.2f}"
        )
    print(
        "\nSynthetic scene: these numbers exercise the code path, they do not "
        "measure real-world accuracy."
    )
    return 0


def _cmd_analyse(args: argparse.Namespace) -> int:
    path = Path(args.video)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    print(
        "error: video analysis needs a configured detector and pitch-keypoint model.\n"
        "       Neither ships with this repository yet — see docs/06-roadmap.md.\n"
        "       Run 'offball demo' to exercise the tactics layer in the meantime.",
        file=sys.stderr,
    )
    return 1


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("error: pip install 'offball[api]'", file=sys.stderr)
        return 2
    uvicorn.run("offball.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="offball", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"offball {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show version and backend availability")
    p_info.set_defaults(func=_cmd_info)

    p_demo = sub.add_parser("demo", help="run the synthetic sequence and print a report")
    p_demo.add_argument("--frames", type=int, default=120, help="frames to simulate")
    p_demo.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_demo.set_defaults(func=_cmd_demo)

    p_analyse = sub.add_parser("analyse", help="analyse a match video")
    p_analyse.add_argument("video", help="path to the footage")
    p_analyse.set_defaults(func=_cmd_analyse)

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
