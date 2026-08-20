"""Entry point.

    python main.py process [video] [--zone x y w h] [--no-video]
    python main.py serve   [--host H] [--port P]
    python main.py results [job_id]

`process` runs the whole pipeline synchronously and prints the answer table;
`serve` exposes the same pipeline over REST at /docs.
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from config.settings import get_settings
from core.geometry import NormalizedRect, format_timestamp
from repository.database import init_db, session_scope
from repository.event_repository import EventRepository
from repository.job_repository import JobRepository
from repository.zone_repository import ZoneRepository
from services.pipeline_service import PipelineService


def _bootstrap():
    init_db()
    settings = get_settings()
    return settings


def cmd_process(args: argparse.Namespace) -> int:
    settings = _bootstrap()
    video = settings.resolve(args.video or settings.default_video)
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    settings.write_annotated_video = not args.no_video
    cache = Path(args.cache) if args.cache else None
    if cache is not None and not cache.exists():
        print(f"cache not found: {cache}", file=sys.stderr)
        return 2

    with session_scope() as session:
        zones = ZoneRepository(session)
        if args.zone:
            x, y, w, h = args.zone
            zone = zones.create(args.zone_name, NormalizedRect(x, y, w, h),
                                "created from the command line")
        else:
            zone = zones.ensure_default()
        job = JobRepository(session).create(str(video), zone.id)
        job_id = job.id

    print(f"job {job_id}  video={video.name}  "
          f"zone=({zone.x:.2f},{zone.y:.2f},{zone.w:.2f},{zone.h:.2f})")
    PipelineService(settings).run(job_id, detections_cache=cache)
    _print_results(job_id)

    with session_scope() as session:
        job = JobRepository(session).get(job_id)
        report = (job.artifacts or {}).get("report")
    if report and args.open:
        webbrowser.open(Path(report).resolve().as_uri())
    return 0


def _print_results(job_id: int) -> None:
    with session_scope() as session:
        job = JobRepository(session).get(job_id)
        rows = EventRepository(session).list_for_job(job_id)

    if job is None:
        print("no such job")
        return

    calib = job.calibration or {}
    val = job.validation or {}
    spread = val.get("cross_metric_spread_frames", {})
    naive = val.get("naive_area_baseline_error_frames", {})

    print()
    print(f"{'track':>6} {'type':<11} {'frame':>6} {'timestamp':>11} {'dist m':>7} "
          f"{'bounding box (x1,y1,x2,y2)':<30} {'spread':>6} {'naive d':>8}")
    print("-" * 104)
    for r in rows:
        box = f"{r.x1:.0f},{r.y1:.0f},{r.x2:.0f},{r.y2:.0f}"
        delta = (r.naive_area_frame - r.frame_index) if r.naive_area_frame is not None else 0
        mark = "*" if r.is_emergency else " "
        print(f"{r.track_id:>6}{mark}{r.vehicle_type:<10} {r.frame_index:>6} "
              f"{r.timestamp_hms:>11} {r.distance_m:>7.1f} {box:<30} "
              f"{r.agreement_frames:>6} {delta:>+8d}")

    print()
    print(f"vehicles captured ......... {val.get('vehicles', 0)}")
    by_type = val.get("vehicles_by_type", {})
    if by_type:
        print("  by type ................. " + ", ".join(
            f"{k} {n}" for k, n in sorted(by_type.items(), key=lambda kv: -kv[1])))
    print(f"armed by the zone ......... {val.get('vehicles_triggered_by_zone', 0)}")
    print(f"cross-metric spread ....... median {spread.get('median', 0):.0f} frames, "
          f"p90 {spread.get('p90', 0):.0f}, max {spread.get('max', 0):.0f}")
    print(f"                            {spread.get('within_1_frame_pct', 0)}% within 1 frame, "
          f"{spread.get('within_3_frames_pct', 0)}% within 3")
    depth = val.get("cross_metric_depth_spread", {})
    if depth:
        print(f"  same, as depth ............ median {depth['median_pct']:.2f}% apart, "
              f"p90 {depth['p90_pct']:.2f}%, "
              f"{depth['within_1_pct']}% within 1%")
    if naive.get("median") is not None:
        print(f"naive argmax(area) error .. median {naive['median']:.0f} frames, "
              f"max {naive['max']:.0f}, wrong on {naive['disagreed_pct']}% of vehicles")
    print(f"calibration ............... horizon y={calib.get('horizon_y', 0):.0f} px, "
          f"R2={calib.get('r_squared', 0):.3f}, "
          f"camera height {calib.get('camera_height_m', 0):.1f} m")
    for key, label in (("report", "report"), ("annotated_video", "video"),
                       ("results_csv", "csv"), ("results_json", "json")):
        path = (job.artifacts or {}).get(key)
        if path:
            print(f"{label:.<26} {path}")


def cmd_reanalyse(args: argparse.Namespace) -> int:
    """Re-run the analysis on a finished job's cached detections.

    Detection dominates the runtime; everything after it is milliseconds. This
    makes a different capture zone a sub-second experiment, and guarantees the
    only thing that changed is the analysis.
    """
    settings = _bootstrap()
    cache = settings.output_dir / f"job_{args.job_id:04d}" / "detections.json"
    if not cache.exists():
        print(f"no cached detections for job {args.job_id} ({cache})", file=sys.stderr)
        return 2

    settings.write_annotated_video = not args.no_video
    with session_scope() as session:
        source = JobRepository(session).get(args.job_id)
        if source is None:
            print(f"no such job: {args.job_id}", file=sys.stderr)
            return 2
        zones = ZoneRepository(session)
        if args.zone:
            x, y, w, h = args.zone
            zone = zones.create(args.zone_name, NormalizedRect(x, y, w, h),
                                f"re-analysis of job {args.job_id}")
        else:
            zone = zones.get(source.zone_id) or zones.ensure_default()
        job_id = JobRepository(session).create(source.video_path, zone.id).id

    print(f"job {job_id}  re-analysing job {args.job_id}  "
          f"zone=({zone.x:.2f},{zone.y:.2f},{zone.w:.2f},{zone.h:.2f})")
    PipelineService(settings).run(job_id, detections_cache=cache)
    _print_results(job_id)
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    _bootstrap()
    job_id = args.job_id
    if job_id is None:
        with session_scope() as session:
            jobs = JobRepository(session).list(limit=1)
        if not jobs:
            print("no jobs yet")
            return 1
        job_id = jobs[0].id
    _print_results(job_id)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    settings = _bootstrap()
    import uvicorn

    host = args.host or settings.api_host
    port = args.port or settings.api_port
    print(f"docs on http://{host}:{port}/docs")
    uvicorn.run("api.app:app", host=host, port=port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vehicle-cpa",
        description="Find the frame where each vehicle is closest to the camera.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="run the full pipeline on a clip")
    p.add_argument("video", nargs="?", help="path to the video (default: config)")
    p.add_argument("--zone", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                   help="virtual rectangle in normalised coords, e.g. --zone .15 .55 .85 .45")
    p.add_argument("--zone-name", default="cli-zone")
    p.add_argument("--no-video", action="store_true", help="skip the annotated render")
    p.add_argument("--open", action="store_true", help="open the HTML report when done")
    p.add_argument("--cache", help="reuse a detections.json instead of re-running YOLO")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("reanalyse", help="redo the analysis on cached detections")
    p.add_argument("job_id", type=int)
    p.add_argument("--zone", nargs=4, type=float, metavar=("X", "Y", "W", "H"))
    p.add_argument("--zone-name", default="cli-zone")
    p.add_argument("--no-video", action="store_true")
    p.set_defaults(func=cmd_reanalyse)

    p = sub.add_parser("results", help="print the table for a finished job")
    p.add_argument("job_id", nargs="?", type=int)
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("serve", help="start the REST API")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)
    return parser


if __name__ == "__main__":
    ns = build_parser().parse_args()
    raise SystemExit(ns.func(ns))
