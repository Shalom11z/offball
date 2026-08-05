"""Command-line entry point.

::

    offball demo                      # run the synthetic sequence, print a report
    offball demo --json               # machine-readable
    offball analyse match.mp4         # analyse real footage (needs the vision extra)
    offball benchmark match.mp4       # how often does calibration succeed?
    offball inspect match.mp4         # render overlays to see *why* it failed
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
    """Run the full pipeline over a real video file."""
    path = Path(args.video)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2

    try:
        from .pipeline import Pipeline, PipelineConfig
        from .tactics.offball import ScoringConfig
        from .video import probe, read_frames
        from .vision.broadcast import BroadcastCalibrator
        from .vision.detection import DetectorConfig, YoloDetector
        from .vision.lines import ClassicalKeypointSource
        from .vision.teams import TeamAssigner
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    meta = probe(path)
    fps = meta.fps / args.stride
    print(f"{path.name}: {meta.width}x{meta.height}, {meta.fps:.2f} fps, "
          f"{meta.frame_count} frames ({meta.duration / 60:.1f} min)")
    print(f"analysing every {args.stride} frame(s) -> effective {fps:.1f} fps")
    print(f"detector: {args.weights} | calibrator: {args.calibrator} | kernels: {BACKEND}")
    if args.limit:
        print(f"limited to {args.limit} frames")
    print()

    detector = YoloDetector(
        args.weights, DetectorConfig(confidence=args.confidence, image_size=args.imgsz)
    )

    # Kit colours must be learned before the main pass, from frames sampled
    # across the whole clip: lighting shifts over 90 minutes, and a profile
    # fitted only on the opening minutes degrades badly later. Without this the
    # assigner abstains on every player and nothing can be scored.
    assigner = TeamAssigner()
    if not args.no_teams:
        print("fitting kit colours...", flush=True)
        samples = []
        for frame in read_frames(path, stride=max(1, meta.frame_count // 12), limit=12):
            boxes = [d.bbox for d in detector.detect(frame) if not d.is_ball]
            if boxes:
                samples.append((frame, boxes))
        profile = assigner.fit(samples)
        if profile is None:
            print("  could not learn kit colours; teams will be unassigned")
        else:
            print(f"  separation {profile.separation:.0f} "
                  f"({'reliable' if profile.is_reliable else 'TOO SIMILAR - unreliable'})")

    keypoints = (
        ClassicalKeypointSource() if args.calibrator == "lines" else BroadcastCalibrator()
    )

    pipeline = Pipeline(
        detector=detector,
        keypoints=keypoints,
        team_assigner=assigner,
        config=PipelineConfig(
            fps=fps,
            pitch_length=args.pitch_length,
            pitch_width=args.pitch_width,
            scoring=ScoringConfig(
                pitch_length=args.pitch_length, pitch_width=args.pitch_width
            ),
        ),
    )

    frames = read_frames(path, stride=args.stride, limit=args.limit)
    result = pipeline.run(frames)

    total = len(result.frames)
    print("--- run quality " + "-" * 45)
    print(f"  frames analysed      {total}")
    print(f"  calibrated           {result.calibrated_frames} "
          f"({result.calibrated_frames / max(total, 1):.0%})")
    print(f"  calibrations rejected{result.rejected_calibrations:>5}")
    print(f"  ball detected        {result.ball_detected_frames} "
          f"({result.ball_detected_frames / max(total, 1):.0%})")
    print(f"  ball after repair    {result.ball_recovered_frames} "
          f"({result.ball_recovered_frames / max(total, 1):.0%})")
    print(f"  frames scored        {result.report.frames_scored} "
          f"(coverage {result.report.coverage:.0%})")
    print()

    if result.report.coverage < 0.6:
        print("  NOTE: coverage below 60%. These figures are provisional; the")
        print("  usual cause is calibration failing on close-ups and replays.")
        print()

    if args.json:
        json.dump(result.report.to_dict(), sys.stdout, indent=2, default=str)
        print()
        return 0

    _print_report(result.report)
    return 0


def _print_report(report) -> None:
    """Shared table output for `demo` and `analyse`."""
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
            f"{p.track_id:>6} {p.frames:>7} {p.median_space_owned:7.0f}m\u00b2 "
            f"{p.availability_rate:6.0%} {p.offside_rate:7.0%} {margin:>8} "
            f"{p.mean_lines_broken:6.2f} {p.mean_pressure:6.2f}"
        )


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Measure how well calibration performs on real footage."""
    from .benchmark import evaluate_frames, survey_soccernet

    path = Path(args.source)
    if not path.exists():
        print(f"error: no such file or directory: {path}", file=sys.stderr)
        return 2

    if args.soccernet:
        survey = survey_soccernet(path, limit=args.limit)
    else:
        survey = evaluate_frames(
            path,
            ground_truth=args.ground_truth,
            limit=args.limit,
            stride=args.stride,
        )

    if args.json:
        json.dump(survey.to_dict(), sys.stdout, indent=2)
        print()
        return 0

    print(f"calibration survey: {path.name}\n")
    print(survey.summary())
    print()
    if survey.rate < 0.6:
        print("Calibration rate below 60%: this camera angle or footage quality")
        print("is not usable as-is. Inspect a failing frame before tuning.")
    if not args.ground_truth and not args.soccernet:
        print("No ground truth supplied, so this measures whether calibration")
        print("*succeeds*, not whether it is *correct*. Pass --ground-truth to")
        print("get real error figures.")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Render calibration overlays so failures can be seen, not guessed at."""
    import cv2

    from .benchmark import _iter_images
    from .vision.calibration import CalibrationConfig, calibrate_frame
    from .vision.lines import ClassicalKeypointSource
    from .viz import overlay_calibration

    path = Path(args.source)
    if not path.exists():
        print(f"error: no such file or directory: {path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = ClassicalKeypointSource()
    config = CalibrationConfig()
    written = failed = 0

    for i, frame in enumerate(_iter_images(path, limit=args.limit, stride=args.stride)):
        pairs = source.keypoints(frame)
        cal = calibrate_frame(pairs, config) if pairs else None
        if cal is None:
            failed += 1
        label = (
            f"frame {i}  support={source.last_support:.2f}  lines={source.last_line_count}"
        )
        if cal is not None:
            label += f"  err={cal.error:.2f}m  inliers={cal.inliers}"

        # Failures are the whole point of this command; --failures-only drops
        # the successes so a long clip does not bury them.
        if args.failures_only and cal is not None:
            continue

        image = overlay_calibration(
            frame, cal.matrix if cal else None, label=label
        )
        name = out_dir / f"frame_{i:05d}{'_FAIL' if cal is None else ''}.jpg"
        cv2.imwrite(str(name), image)
        written += 1

    print(f"wrote {written} overlay(s) to {out_dir}")
    print(f"{failed} frame(s) failed to calibrate")
    if failed:
        print("Open a *_FAIL image: if the pitch is clearly visible there, the")
        print("detector is at fault; if it is a close-up or replay, abstaining")
        print("is correct behaviour.")
    return 0


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
    p_analyse.add_argument("--weights", default="yolov8x.pt",
                           help="detector checkpoint (downloaded on first use)")
    p_analyse.add_argument("--stride", type=int, default=5,
                           help="analyse every Nth frame; raises throughput, "
                                "degrades tracking identity")
    p_analyse.add_argument("--limit", type=int, default=None,
                           help="stop after this many analysed frames")
    p_analyse.add_argument("--confidence", type=float, default=0.25)
    p_analyse.add_argument("--imgsz", type=int, default=1280,
                           help="inference resolution; below 1280 the ball is lost")
    p_analyse.add_argument("--pitch-length", type=float, default=105.0)
    p_analyse.add_argument("--pitch-width", type=float, default=68.0)
    p_analyse.add_argument("--calibrator", choices=("camera", "lines"), default="camera",
                           help="'camera' fits a physical camera from the centre "
                                "circle, halfway line and pitch boundary (works on "
                                "broadcast); 'lines' matches straight lines only")
    p_analyse.add_argument("--no-teams", action="store_true",
                           help="skip kit-colour fitting (players stay unassigned)")
    p_analyse.add_argument("--json", action="store_true")
    p_analyse.set_defaults(func=_cmd_analyse)

    p_bench = sub.add_parser(
        "benchmark", help="measure calibration quality on real footage"
    )
    p_bench.add_argument("source", help="video file or directory of frames")
    p_bench.add_argument("--ground-truth", default=None,
                         help="JSON list of per-frame image->pitch homographies")
    p_bench.add_argument("--stride", type=int, default=25,
                         help="sample every Nth frame (default: ~1 per second)")
    p_bench.add_argument("--limit", type=int, default=200,
                         help="stop after this many sampled frames")
    p_bench.add_argument("--soccernet", action="store_true",
                         help="treat SOURCE as a SoccerNet-Calibration split; "
                              "ground truth comes from its line annotations")
    p_bench.add_argument("--json", action="store_true")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_inspect = sub.add_parser(
        "inspect", help="render calibration overlays to look at failures"
    )
    p_inspect.add_argument("source", help="video file or directory of frames")
    p_inspect.add_argument("--out", default="overlays", help="output directory")
    p_inspect.add_argument("--stride", type=int, default=25)
    p_inspect.add_argument("--limit", type=int, default=40)
    p_inspect.add_argument("--failures-only", action="store_true",
                           help="only write frames that failed to calibrate")
    p_inspect.set_defaults(func=_cmd_inspect)

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
