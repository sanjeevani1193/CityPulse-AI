"""Discover, ingest, resume and reconcile historical Citi Bike archives."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path

from citypulse_ai.etl.archives import month_range, probe_boundary_archive, validate_month, write_json
from citypulse_ai.etl.historical import plan_months, process_month, register_existing
from citypulse_ai.etl.history_catalog import build_catalog


@contextmanager
def single_writer(root: Path):
    """Fail visibly instead of racing two commands that update the same manifest."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".ingestion.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another ingestion command owns this output root") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", nargs="+", help="Source archive months as YYYY-MM")
    parser.add_argument("--start", help="Inclusive local date; first of month")
    parser.add_argument("--end", help="Exclusive local date; first of month")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/citibike"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/historical"))
    parser.add_argument("--sample-rows-per-file", type=int)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--memory-limit", default="512MB")
    parser.add_argument("--reserve-gib", type=float, default=5)
    parser.add_argument("--plan", action="store_true", help="Inspect remote ZIP directories and estimate storage; no downloads")
    parser.add_argument("--probe-boundary", action="store_true", help="Read small remote CSV prefixes; never mark the archive processed")
    parser.add_argument("--register-existing", nargs="*", default=[], metavar="MONTH=OUTPUT_DIR")
    parser.add_argument("--reconcile", action="store_true", help="Build new global duplicate-audited start-month snapshot")
    parser.add_argument("--cloud-publish", action="store_true", help="OPT IN: publish and verify completed source artifacts")
    parser.add_argument("--cloud-project")
    parser.add_argument("--cloud-bucket")
    parser.add_argument("--cloud-prefix", default="curated/historical")
    parser.add_argument("--cloud-max-file-mib", type=int, default=100,
                        help="Explicit per-file upload ceiling in MiB (default 100)")
    parser.add_argument("--cloud-upload-timeout", type=float, default=600,
                        help="Per upload HTTP request timeout in seconds (default 600)")
    parser.add_argument("--cloud-upload-attempts", type=int, default=3,
                        help="Bounded create-only upload attempts, 1 through 5 (default 3)")
    parser.add_argument("--publish-existing-only", action="store_true",
                        help="Publish registered processed months without downloads or ETL")
    args = parser.parse_args(argv)
    if args.month and (args.start or args.end):
        parser.error("Use --month OR --start/--end")
    if bool(args.start) != bool(args.end):
        parser.error("Both --start and --end are required")
    if args.reserve_gib < 0 or args.chunk_size <= 0:
        parser.error("Reserve must be nonnegative and chunk size positive")
    if args.sample_rows_per_file is not None and (args.sample_rows_per_file <= 0 or args.reconcile):
        parser.error("Samples must be positive and cannot be reconciled as complete partitions")
    months = month_range(args.start, args.end) if args.start else [validate_month(m) for m in (args.month or [])]
    if len(months) != len(set(months)):
        parser.error("Duplicate source months")
    if args.cloud_publish:
        if (not math.isfinite(args.cloud_upload_timeout) or args.cloud_upload_timeout <= 0
                or not 1 <= args.cloud_upload_attempts <= 5):
            parser.error("Cloud timeout must be finite/positive and attempts between 1 and 5")
        if (not args.cloud_project or not args.cloud_project.strip()
                or not args.cloud_bucket or not args.cloud_bucket.strip()
                or not args.cloud_prefix.strip("/") or args.cloud_max_file_mib <= 0):
            parser.error("Cloud publishing requires project, bucket, nonempty prefix, and positive size limit")
        if not months or args.plan or args.probe_boundary or args.sample_rows_per_file is not None:
            parser.error("Cloud publishing requires full source months, not plans, probes, or samples")
        try:
            from citypulse_ai.cloud.historical import publish_month
        except ImportError as error:
            parser.error(f"Install the cloud extra with pip install -e '.[cloud]': {error}")
    elif (args.publish_existing_only or args.cloud_project or args.cloud_bucket
          or args.cloud_prefix != "curated/historical" or args.cloud_max_file_mib != 100
          or args.cloud_upload_timeout != 600 or args.cloud_upload_attempts != 3):
        parser.error("Cloud options require explicit --cloud-publish")
    if args.publish_existing_only and (args.register_existing or args.reconcile):
        parser.error("--publish-existing-only cannot register or reconcile partitions")
    if args.probe_boundary:
        if not months or args.plan or args.register_existing or args.reconcile or args.sample_rows_per_file:
            parser.error("--probe-boundary requires months and no ingestion/plan modes")
        for month in months:
            report = probe_boundary_archive(month)
            write_json(args.output_root / f"boundary_probe_{month}.json", report)
            print(json.dumps(report, indent=2), flush=True)
        return
    if args.plan:
        if not months or args.register_existing or args.reconcile:
            parser.error("--plan requires months and cannot register/reconcile")
        print(json.dumps(plan_months(months, args.raw_root, args.output_root, int(args.reserve_gib*1024**3)), indent=2), flush=True)
        return
    if not months and not args.register_existing and not args.reconcile:
        parser.error("Specify months, existing outputs, or --reconcile")
    with single_writer(args.output_root):
        for registration in args.register_existing:
            month, directory = registration.split("=", 1)
            result = register_existing(month, Path(directory), args.output_root)
            print(json.dumps({"registered": month, "output_dir": result["output_dir"]}), flush=True)
        for month in months:
            if not args.publish_existing_only:
                print(json.dumps({"month": month, "status": "starting"}), flush=True)
                result = process_month(month, args.raw_root, args.output_root, sample_rows=args.sample_rows_per_file,
                                       chunk_size=args.chunk_size, memory_limit=args.memory_limit,
                                       reserve_bytes=int(args.reserve_gib*1024**3))
                print(json.dumps(result, indent=2), flush=True)
            if args.cloud_publish:
                publication = publish_month(month, args.output_root, project=args.cloud_project,
                                            bucket=args.cloud_bucket, prefix=args.cloud_prefix,
                                            max_file_bytes=args.cloud_max_file_mib * 1024**2,
                                            upload_timeout=args.cloud_upload_timeout,
                                            upload_attempts=args.cloud_upload_attempts)
                print(json.dumps({"month": month, "cloud_publication": publication}, indent=2), flush=True)
        if args.reconcile:
            print(json.dumps(build_catalog(args.output_root, memory_limit=args.memory_limit,
                            chunk_size=args.chunk_size, reserve_bytes=int(args.reserve_gib*1024**3)), indent=2), flush=True)


if __name__ == "__main__":
    main()
