"""Owned bounded worker; reuses original SQLPilot AST/readonly execution core."""
from pathlib import Path
import json
import signal
import sys

# This fixed entrypoint also works without an editable install or inherited env.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from citypulse_ai.decision_studio.sql_demo import APPROVED_SQL, load_sql_bundle, same_rows
from citypulse_ai.sqlpilot.core import worker


def execute(directory, identity):
    if identity not in APPROVED_SQL:
        raise ValueError("Unapproved query")
    data = load_sql_bundle(directory)
    result = worker(data["directory"] / "sample.duckdb", {"sql": APPROVED_SQL[identity]})
    expected = data["queries"][identity]["verified_result"]
    if result["truncated"] or result["columns"] != expected["columns"] or not same_rows(result["rows"], expected["rows"]):
        raise ValueError("Query output differs from verified source")
    return {**result, "query_id": identity, "verified_against_source": True,
            "generation_kind": "curated_reference_sql", "live_llm_exercised": False}


if __name__ == "__main__":
    signal.alarm(5)
    try:
        print(json.dumps(execute(Path(sys.argv[1]), sys.argv[2]), allow_nan=False))
    except Exception:
        print(json.dumps({"error": "Approved query could not complete safely"}))
