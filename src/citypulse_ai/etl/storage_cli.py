"""Explicit source backup, restore, or offline dry-run retention planning."""

import argparse
import json
import math
from pathlib import Path

from citypulse_ai.etl.historical import load_manifest
from citypulse_ai.etl.historical_cli import single_writer
from citypulse_ai.etl.retention import retention_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['backup-sources', 'restore', 'plan'])
    parser.add_argument('--month')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/historical'))
    parser.add_argument('--raw-root', type=Path, default=Path('data/raw/citibike'))
    parser.add_argument('--project')
    parser.add_argument('--bucket')
    parser.add_argument('--prefix', default='raw/historical')
    parser.add_argument('--max-file-mib', type=int)
    parser.add_argument('--upload-timeout', type=float, default=600)
    parser.add_argument('--upload-attempts', type=int, default=3)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--kind', choices=['outputs', 'sources'], default='outputs')
    parser.add_argument('--verify-local-hashes', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'plan':
        if args.project or args.bucket or args.destination:
            parser.error('Plan is offline; do not supply cloud or restore options')
        print(json.dumps(retention_plan(args.output_root, raw_root=args.raw_root,
                                       verify_local_hashes=args.verify_local_hashes), indent=2))
        return
    if not args.month or not args.project or not args.bucket:
        parser.error('Backup/restore requires month, project and bucket')
    if (not math.isfinite(args.upload_timeout) or args.upload_timeout <= 0
            or not 1 <= args.upload_attempts <= 5):
        parser.error('Upload timeout must be finite/positive and attempts between 1 and 5')
    if args.action == 'backup-sources' and (not args.max_file_mib or args.max_file_mib <= 0):
        parser.error('Source backup requires an explicit positive --max-file-mib')
    if args.action == 'restore' and args.destination is None:
        parser.error('Restore requires an explicit destination')
    from citypulse_ai.cloud.gcs import GCSStorage
    from citypulse_ai.cloud.retention import publish_sources, restore_partition
    with single_writer(args.output_root):
        if args.action == 'backup-sources':
            result = publish_sources(args.month, args.output_root, raw_root=args.raw_root,
                                     project=args.project, bucket=args.bucket, prefix=args.prefix,
                                     max_file_bytes=args.max_file_mib * 1024**2,
                                     upload_timeout=args.upload_timeout, upload_attempts=args.upload_attempts)
        else:
            entry = load_manifest(args.output_root)['archives'][args.month]
            store = GCSStorage(args.project, args.bucket)
            result = {'restored_directory': str(restore_partition(entry, args.destination,
                                               store=store, kind=args.kind))}
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
