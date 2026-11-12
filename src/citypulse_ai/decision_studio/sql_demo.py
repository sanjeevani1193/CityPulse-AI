"""Fixed curated queries and local bundle checks; no LLM, MCP or cloud client."""
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .artifacts import ArtifactError, read_json, sha256

DEFAULT_SQL_BUNDLE = Path(__file__).resolve().parents[3] / "apps/decision_studio/sqlpilot_data"
SQL_FORMAT = "citypulse-curated-sql-v1"
# Exact existing frozen reference SQL; only IDs, never visitor-supplied SQL, enter
# the public execution path. Tests check this registry against the original file.
APPROVED_SQL = {
    "total": "SELECT sum(ride_count) AS starts FROM hourly_rides",
    "top_station": "SELECT start_station_id,sum(ride_count) AS rides FROM hourly_rides GROUP BY 1 ORDER BY rides DESC,start_station_id LIMIT 1",
    "duration": "SELECT avg(duration_seconds) FROM validated_trips",
    "weekday": "SELECT date_part('dow',hour_start AT TIME ZONE 'America/New_York') AS weekday,sum(ride_count) FROM hourly_rides GROUP BY 1 ORDER BY 1",
    "station_filter": "SELECT sum(ride_count) FROM hourly_rides WHERE start_station_id='001'",
    "join": "SELECT count(*) FROM hourly_rides h JOIN validated_trips t ON h.start_station_id=t.start_station_id AND h.hour_start=date_trunc('hour',t.started_at)",
}
SQL_FILES = {"sample.duckdb", "queries.json", "evaluation.json"}


def same_rows(actual, expected):
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if len(a) != len(e):
            return False
        for x, y in zip(a, e):
            if type(x) in (int, float) and type(y) in (int, float):
                if not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12):
                    return False
            elif type(x) is not type(y) or x != y:
                return False
    return True


def explain_result(identity, result):
    """Deterministic descriptions of returned values, not generated narratives."""
    rows = result["rows"]
    if not rows:
        return "No result rows were returned from this sample."
    if identity == "total":
        return f"The approved hourly sample contains {rows[0][0]:,} accepted recorded starts."
    if identity == "top_station":
        return f"Station {rows[0][0]} has the largest sampled total: {rows[0][1]:,} starts. Ties are broken by exact station ID; this is not a citywide ranking."
    if identity == "duration":
        return f"Mean accepted sampled trip duration is {rows[0][0]:.3f} seconds. This describes only the selected sample."
    if identity == "weekday":
        return f"The weekday groups sum to {sum(r[1] for r in rows):,} sampled starts. Sunday is 0; uneven sample coverage does not establish a citywide weekday pattern."
    if identity == "station_filter":
        return "No hourly rows match exact station ID 001 in this sample. SUM returns SQL NULL, not a certified zero count." if rows[0][0] is None else f"The selected station filter returned {rows[0][0]} sampled starts."
    if identity == "join":
        return f"{rows[0][0]:,} accepted sampled trips match a station-hour row on exact station ID and UTC start hour. This checks key alignment, not additional rides."
    raise ArtifactError("Unknown approved query")


def load_sql_bundle(directory=DEFAULT_SQL_BUNDLE):
    """Fixed trusted files only; metadata loading does not execute SQL."""
    directory = Path(directory)
    try:
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 128*1024:
            raise ArtifactError("Invalid SQL demo manifest")
        manifest = read_json(manifest_path)
        if manifest["format_version"] != SQL_FORMAT or set(manifest["outputs"]) != SQL_FILES:
            raise ArtifactError("Unknown SQL demo format/files")
        for name, record in manifest["outputs"].items():
            path = directory / name
            if path.is_symlink() or not 0 < path.stat().st_size <= 2*1024**2:
                raise ArtifactError("SQL demo file missing, unsafe or oversized")
            if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
                raise ArtifactError("SQL demo checksum mismatch")
        queries = read_json(directory / "queries.json")
        if set(queries) != set(APPROVED_SQL):
            raise ArtifactError("Unapproved query registry")
        for identity, record in queries.items():
            if record["sql"] != APPROVED_SQL[identity] or not isinstance(record["question"], str):
                raise ArtifactError("Query differs from approved reference SQL")
            expected = record["verified_result"]
            if not isinstance(expected["columns"], list) or not isinstance(expected["rows"], list) or len(expected["rows"]) > 100:
                raise ArtifactError("Malformed verified result")
            if any(len(row) != len(expected["columns"]) for row in expected["rows"]):
                raise ArtifactError("Malformed result dimensions")
        evaluation = read_json(directory / "evaluation.json")
        cases = evaluation["cases"]
        if (len(cases) != 9 or len(cases) != evaluation["cases_evaluated"] or sum(c["correct"] for c in cases) != evaluation["correct"]
                or evaluation["generation_kind"] != "live" or not evaluation["live_llm_exercised"]):
            raise ArtifactError("Inconsistent saved live evaluation")
        latencies = sorted(c["latency_seconds"] for c in cases)
        if not math.isclose(latencies[len(latencies)//2], evaluation["median_latency_seconds"], abs_tol=1e-12):
            raise ArtifactError("Inconsistent saved evaluation latency")
        if manifest["sample_rows"] != {"hourly_rides": 1931, "validated_trips": 1998}:
            raise ArtifactError("Unexpected analytical sample coverage")
        return {"directory": directory, "manifest": manifest, "queries": queries, "evaluation": evaluation}
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError("Curated SQL bundle is missing or malformed") from error


def run_approved_query(identity, directory=DEFAULT_SQL_BUNDLE, *, timeout=5):
    """Public input is an exact fixed ID. One isolated read-only execution."""
    if not isinstance(identity, str) or identity not in APPROVED_SQL:
        raise ArtifactError("Choose an approved query ID; arbitrary SQL is not accepted")
    if not 0 < timeout <= 5:
        raise ValueError("Query timeout must be positive and at most five seconds")
    load_sql_bundle(directory)
    started = time.monotonic()
    # A fixed script resolves src itself, including on a fresh clone where the
    # project is not pip-installed and the parent has no PYTHONPATH environment.
    worker = Path(__file__).with_name("sql_demo_worker.py")
    process = subprocess.Popen([sys.executable, str(worker), str(Path(directory).absolute()), identity],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    try:
        out, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        process.kill(); process.communicate()
        raise ArtifactError("Approved SQL exceeded its execution deadline") from error
    except BaseException:
        process.kill(); process.communicate()
        raise
    if process.returncode or len(out.encode()) > 128*1024:
        raise ArtifactError("Approved SQL worker failed safely")
    try:
        result = json.loads(out)
        if not isinstance(result, dict) or result.get("error") or result.get("query_id") != identity or result.get("verified_against_source") is not True:
            raise ArtifactError("Approved SQL failed or did not match verified source results")
    except (TypeError, ValueError) as error:
        raise ArtifactError("Approved SQL failed or returned invalid output") from error
    result["latency_seconds"] = time.monotonic() - started
    return result
