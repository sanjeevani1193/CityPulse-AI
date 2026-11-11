"""Create a new public-safe local sample; never modify the private source DB."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from .artifacts import ArtifactError, read_json, sha256
from .sql_demo import APPROVED_SQL, DEFAULT_SQL_BUNDLE, SQL_FORMAT, load_sql_bundle, same_rows

ROOT = Path(__file__).resolve().parents[3]
SAMPLE = Path("outputs/historical/samples/2024-01-n1000")
EVALUATION = Path("outputs/sqlpilot/groq-20b-live-evaluation.json")
QUESTIONS = Path("tests/fixtures/sqlpilot_questions.json")


def build_sql_bundle(root=ROOT, destination=DEFAULT_SQL_BUNDLE):
    import duckdb
    from citypulse_ai.sqlpilot.core import CONFIG, TABLES, connect, schema, worker

    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise ArtifactError("SQL demo destination already exists; choose a new directory")
    source = root / SAMPLE / "citypulse.duckdb"
    report = read_json(root / SAMPLE / "report.json")
    specification = read_json(root / QUESTIONS)
    evaluation = read_json(root / EVALUATION)
    if evaluation["question_spec_sha256"] != hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest():
        raise ArtifactError("Original question specification identity differs")
    source_paths = [SAMPLE / n for n in ("citypulse.duckdb", "report.json", "validated_trips.parquet", "hourly_rides.parquet")]
    source_paths += [EVALUATION, QUESTIONS, Path("src/citypulse_ai/sqlpilot/core.py")]
    hashes = {str(p): {"sha256": sha256(root / p), "size_bytes": (root / p).stat().st_size} for p in source_paths}
    queries = {}
    with connect(source) as original, tempfile.TemporaryDirectory(prefix="citypulse-sql-demo-") as scratch:
        source_schema = schema(original)
        if source_schema["validated_trips"]["sample_rows"] != report["accepted_rows"] or source_schema["hourly_rides"]["sample_rows"] != report["hourly_rows"]:
            raise ArtifactError("Sample schema/counts differ from ETL report")
        if (report["accepted_rows"], report["hourly_rows"]) != (1998, 1931):
            raise ArtifactError("Unexpected sample identity")
        target = Path(scratch)
        db = target / "sample.duckdb"
        with duckdb.connect(str(db), config=CONFIG) as public:
            public.execute("SET TimeZone='UTC'")
            for table, columns in TABLES.items():
                # Names/types come only from fixed allowlisted source metadata.
                types = source_schema[table]["columns"]
                allowed_types = {"VARCHAR", "TIMESTAMP WITH TIME ZONE", "BIGINT", "DOUBLE"}
                if not set(types.values()) <= allowed_types:
                    raise ArtifactError("Unexpected source type")
                declarations = ",".join(f'"{c}" {types[c]}' for c in columns)
                public.execute(f'CREATE TABLE "{table}" ({declarations})')
                values = original.execute('SELECT ' + ','.join(columns) + ' FROM ' + table).fetchall()
                public.executemany(f'INSERT INTO "{table}" VALUES (' + ','.join('?' for _ in columns) + ')', values)
        # Runtime is always read-only; the new writer above exists only in export.
        with connect(db) as public:
            if schema(public) != source_schema:
                raise ArtifactError("Public schema differs from permitted original schema")
            for case in specification["cases"]:
                identity = case["id"]
                if identity not in APPROVED_SQL:
                    continue
                sql = APPROVED_SQL[identity]
                if sql != case["reference_sql"]:
                    raise ArtifactError("Reference query changed")
                cursor = original.execute(sql)
                expected = {"columns": [c[0] for c in cursor.description], "rows": [list(r) for r in cursor.fetchall()]}
                result = worker(db, {"sql": sql})
                if result["columns"] != expected["columns"] or not same_rows(result["rows"], expected["rows"]):
                    raise ArtifactError("Projected database query differs from real source")
                queries[identity] = {"question": case["question"], "sql": sql, "verified_result": expected}
            span = original.execute("SELECT min(started_at),max(started_at) FROM validated_trips").fetchone()
        for name, data in [("queries.json", queries), ("evaluation.json", evaluation)]:
            (target / name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
        manifest = {"format_version": SQL_FORMAT, "duckdb_version": duckdb.__version__, "sources": hashes,
                    "sample_rows": {t: r["sample_rows"] for t, r in source_schema.items()}, "schema": source_schema,
                    "source_period_utc": [s.isoformat() for s in span],
                    "original_inputs": [{"filename": Path(i["input_file"]).name, "sha256": i["input_sha256"],
                                         "sampled_rows": i["input_rows"]} for i in report["inputs"]],
                    "scope": "First 1000 records of each of two January 2024 archive members; 2000 source rows, 1998 accepted, 2 rejected. Source-local dates can cross archive boundaries. Not full-history totals or physical demand.",
                    "transformation": "New physical tables containing only the original SQLPilot-permitted columns; values/types preserved. Drop ride IDs, source paths/row IDs, UTC-ns helper columns and unrelated views. No synthetic rows or model results.",
                    "verification": "Six curated queries agree with the original read-only database, including SQL NULL; not a new LLM evaluation.",
                    "outputs": {p.name: {"size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(target.iterdir())}}
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        load_sql_bundle(target)
        if any(sha256(root / p) != v["sha256"] for p, v in hashes.items()):
            raise ArtifactError("A source changed during export")
        destination.mkdir(parents=True)
        for p in target.iterdir():
            with (destination / p.name).open("xb") as stream:
                stream.write(p.read_bytes())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_SQL_BUNDLE)
    args = parser.parse_args()
    result = build_sql_bundle(destination=args.output)
    print(json.dumps({"sample_rows": result["sample_rows"], "outputs": result["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
