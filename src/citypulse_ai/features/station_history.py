"""Read verified, completed station-hour checkpoints; never reconcile or access GCS."""

import json
from pathlib import Path

import duckdb
import pandas as pd

from citypulse_ai.features.station_dataset import utc
from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.etl.storage_state import output_receipts, source_receipts, revision_for


class StationHistory:
    """Immutable accepted-record source, not a certification of trusted zero labels.

    Uses the completed temporary audit because older persistent snapshots may
    omit recent months. Queries are in memory at 512 MB with disk spill disabled.
    """

    def __init__(self, audit_path: Path, manifest_path: Path):
        self.audit_path, self.manifest_path = Path(audit_path), Path(manifest_path)
        self.audit_sha256 = fingerprint(self.audit_path)
        self.manifest_sha256 = fingerprint(self.manifest_path)
        bundle = json.loads(self.audit_path.read_text())
        payload = bundle['payload']
        self.specification = payload['specification']
        self.identity = revision_for(self.specification)
        if (bundle['sha256'] != revision_for(payload) or payload['status'] != 'complete'
                or payload['persistent_catalog_modified'] is not False
                or self.audit_path.stem != self.identity):
            raise ValueError('A complete content-verified temporary audit is required')
        manifest = json.loads(self.manifest_path.read_text())
        if payload['manifest_snapshot'] != manifest:
            raise ValueError('Current manifest differs from completed audit provenance')
        entries = {m: e for m, e in manifest['archives'].items() if e['status'] == 'processed'}
        expected = {'format_version': 1, 'modelling_specification': manifest['modelling_specification'],
                    'inputs': {}}
        self.input_paths = []
        for month, entry in entries.items():
            outputs, sources = output_receipts(entry), source_receipts(entry)
            expected['inputs'][month] = {
                'source': entry['source'], 'input_rows': entry['input_rows'],
                'accepted_rows': entry['accepted_rows'], 'rejected_rows': entry['rejected_rows'],
                'output_sha256': entry['output_sha256'],
                'output_revision': entry['cloud_publication']['revision'],
                'source_revision': entry['source_backup']['revision'],
                'cloud_objects': {kind: {n: {k: r[k] for k in
                    ('cloud_path', 'generation', 'sha256', 'size_bytes')}
                    for n, r in receipts.items()} for kind, receipts in
                    [('cloud_publication', outputs), ('source_backup', sources)]},
            }
            for name in ('validated_trips.parquet', 'rejected_trips.csv'):
                path = Path(entry['output_dir']) / name
                if not path.is_file() or fingerprint(path) != entry['output_sha256'][name]:
                    raise ValueError(f'Required local processed input unavailable or changed: {path}')
                self.input_paths.append(path)
        if expected != self.specification:
            raise ValueError('Audit input identities differ from manifest')
        self.source_months = frozenset(entries)
        self.report = payload['report']
        if (self.report['duplicate_station_hour_keys'] != 0
                or self.report['hourly_ride_starts'] != self.report['accepted_rows']):
            raise ValueError('Completed audit station-hour invariants failed')
        root = Path(payload['execution']['checkpoint_root'])
        if root.name != self.identity or root.is_symlink():
            raise ValueError('Checkpoint identity differs')
        self._assets = []

        def load(stage):
            path = root / f'{stage}.json'
            b = json.loads(path.read_text()); p = b['payload']
            if (path.is_symlink() or b['sha256'] != revision_for(p)
                    or p['identity'] != self.identity or p['stage'] != stage
                    or p['format_version'] != 1):
                raise ValueError(f'Corrupt checkpoint: {stage}')
            self._assets.append((path, fingerprint(path)))
            for receipt in p['files']:
                asset = root / receipt['name']
                if (Path(receipt['name']).name != receipt['name'] or asset.is_symlink()
                        or asset.stat().st_size != receipt['size_bytes']
                        or fingerprint(asset) != receipt['sha256']):
                    raise ValueError(f'Changed checkpoint asset: {asset}')
                self._assets.append((asset, receipt['sha256']))
            return p

        source = load('source-counts')['data']
        for key, field in [('input', 'input_rows'), ('accepted', 'accepted_rows'),
                           ('rejected', 'rejected_rows')]:
            if source[key] != {m: e[field] for m, e in entries.items()}:
                raise ValueError('Incomplete checkpoint source coverage')
        duplicate = load('duplicates')['data']
        if any(duplicate[k] != self.report[k] for k in
               ('duplicate_ride_ids', 'cross_archive_duplicate_ride_ids')):
            raise ValueError('Duplicate checkpoint differs from completed audit')
        days = load('canonical-subtotals')['data']['days']
        self.paths = []
        total_rows = total_starts = 0
        previous_stop = None
        for i, (start, starts) in enumerate(days):
            checkpoint = load(f'hourly-day-{i:05d}')
            data = checkpoint['data']
            begin, stop = pd.Timestamp(data['start']), pd.Timestamp(data['stop'])
            if (data['start'] != start or data['ride_starts'] != starts
                    or len(checkpoint['files']) != 1 or stop - begin != pd.Timedelta(days=1)
                    or (previous_stop is not None and begin < previous_stop)):
                raise ValueError('Hourly partition alignment differs')
            previous_stop = stop
            self.paths.append(str(root / checkpoint['files'][0]['name']))
            total_rows += data['rows']; total_starts += starts
        if (total_rows, total_starts) != (self.report['hourly_rows'], self.report['hourly_ride_starts']):
            raise ValueError('Hourly checkpoint totals differ from completed audit')
        self._unchanged()

    def _unchanged(self):
        if (fingerprint(self.audit_path) != self.audit_sha256
                or fingerprint(self.manifest_path) != self.manifest_sha256):
            raise ValueError('Audit or manifest changed during dataset preparation')
        if any(fingerprint(p) != h for p, h in self._assets):
            raise ValueError('Hourly source assets changed')

    def _query(self, sql, parameters):
        self._unchanged()
        with duckdb.connect(':memory:', config={'memory_limit': '512MB', 'threads': 2,
                                               'max_temp_directory_size': '0B'}) as con:
            con.execute("SET TimeZone='UTC'")
            result = con.execute(sql, [self.paths, *parameters]).df()
        self._unchanged()
        return result

    def verify_local_months(self, months):
        """Adjacent sources establish accepted-record finalization, not zero quality."""
        for month in months:
            following = str(pd.Period(month, freq='M') + 1)
            if month not in self.source_months or following not in self.source_months:
                raise ValueError(f'Missing source/adjacent boundary archive for {month}')
        totals = self._query('SELECT start_month, sum(ride_count)::BIGINT AS starts '
                            'FROM read_parquet(?, hive_partitioning=false) '
                            'WHERE start_month IN (SELECT unnest(?)) GROUP BY 1', [list(months)])
        result = dict(zip(totals.start_month, totals.starts.map(int)))
        if any(result.get(m) != self.report['start_month_counts'].get(m) for m in months):
            raise ValueError('Local-month counts differ from completed audit')
        return {m: result[m] for m in months}

    def activity(self, start, end):
        """Only the requested training interval enters station statistics."""
        start, end = utc(start), utc(end)
        if start >= end: raise ValueError('Ordered time bounds required')
        return self._query("""SELECT start_station_id AS station_id,
            CAST(hour_start AT TIME ZONE 'America/New_York' AS DATE) AS local_date,
            sum(ride_count)::BIGINT AS ride_starts
            FROM read_parquet(?, hive_partitioning=false)
            WHERE hour_start >= ? AND hour_start < ? GROUP BY 1,2 ORDER BY 1,2""", [start, end])

    def hourly(self, stations, start, end):
        start, end = utc(start), utc(end)
        if start >= end: raise ValueError('Ordered time bounds required')
        if not stations or any(not isinstance(s, str) or not s.strip() for s in stations):
            raise ValueError('Explicit exact station IDs required')
        return self._query("""SELECT start_station_id AS station_id,
            hour_start AS hour_start_utc, ride_count AS ride_count_accepted
            FROM read_parquet(?, hive_partitioning=false)
            WHERE hour_start >= ? AND hour_start < ?
              AND start_station_id IN (SELECT unnest(?)) ORDER BY 1,2""", [start, end, list(stations)])
