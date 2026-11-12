"""Real local SQL and UI tests; no LLM, MCP transport, cloud or synthetic answers."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import duckdb
import pytest
from streamlit.testing.v1 import AppTest

from citypulse_ai.decision_studio.artifacts import ArtifactError, read_json, sha256
from citypulse_ai.decision_studio.sql_demo import APPROVED_SQL, DEFAULT_SQL_BUNDLE, explain_result, load_sql_bundle, run_approved_query
from citypulse_ai.sqlpilot.core import QueryRejected, TABLES, connect, schema, validate

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "apps/decision_studio/streamlit_app.py"


@pytest.fixture
def copied(tmp_path):
    import shutil
    target = tmp_path / "sql"
    shutil.copytree(DEFAULT_SQL_BUNDLE, target)
    return target


def resign(directory, name):
    p = directory / name
    manifest = read_json(directory / "manifest.json")
    manifest["outputs"][name] = {"size_bytes": p.stat().st_size, "sha256": sha256(p)}
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_registry_matches_original_frozen_reference_queries_and_evaluation():
    data = load_sql_bundle()
    questions = read_json(ROOT / "tests/fixtures/sqlpilot_questions.json")
    assert APPROVED_SQL == {c["id"]: c["reference_sql"] for c in questions["cases"] if "reference_sql" in c}
    canonical = hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()
    assert data["evaluation"]["question_spec_sha256"] == canonical
    assert data["evaluation"]["correct"] == 6 and data["evaluation"]["cases_evaluated"] == 9
    assert data["evaluation"]["median_latency_seconds"] == pytest.approx(1.9463340410002274)
    assert all("actual_result" not in c and "sql_attempts" not in c for c in data["evaluation"]["cases"])


def test_real_physical_schema_counts_and_no_hidden_columns():
    data = load_sql_bundle()
    with connect(DEFAULT_SQL_BUNDLE / "sample.duckdb") as connection:
        assert schema(connection) == data["manifest"]["schema"]
        assert connection.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1").fetchall() == [("hourly_rides",), ("validated_trips",)]
        for table, columns in TABLES.items():
            actual = connection.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='main' AND table_name=?", [table]).fetchall()
            assert {r[0] for r in actual} == set(columns)
        assert connection.execute("SELECT sum(ride_count) FROM hourly_rides").fetchone()[0] == 1998
        assert connection.execute("SELECT count(*) FROM hourly_rides").fetchone()[0] == 1931


@pytest.mark.parametrize("identity", list(APPROVED_SQL))
def test_all_six_real_worker_results_match_verified_original_source(identity, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)  # Fresh-clone/Streamlit launch path.
    data = load_sql_bundle()
    before = sha256(DEFAULT_SQL_BUNDLE / "sample.duckdb")
    result = run_approved_query(identity)
    assert result["rows"] == data["queries"][identity]["verified_result"]["rows"]
    assert result["columns"] == data["queries"][identity]["verified_result"]["columns"]
    assert result["verified_against_source"] and result["generation_kind"] == "curated_reference_sql"
    assert result["live_llm_exercised"] is False and result["latency_seconds"] >= 0
    assert before == sha256(DEFAULT_SQL_BUNDLE / "sample.duckdb")
    if identity == "station_filter": assert result["rows"] == [[None]]


def test_original_source_projection_and_query_parity_if_sources_available(tmp_path):
    source = ROOT / "outputs/historical/samples/2024-01-n1000/citypulse.duckdb"
    if not source.is_file():
        pytest.skip("Retained original local sample unavailable; no downloads attempted")
    from citypulse_ai.decision_studio.sql_export import build_sql_bundle
    before = sha256(source)
    data = load_sql_bundle()
    assert data["evaluation"] == read_json(ROOT / "outputs/sqlpilot/groq-20b-live-evaluation.json")
    assert data["manifest"]["sources"][str(source.relative_to(ROOT))]["sha256"] == before
    with connect(source) as original, connect(DEFAULT_SQL_BUNDLE / "sample.duckdb") as public:
        for table, columns in TABLES.items():
            projection = ",".join(columns)
            assert sorted(original.execute(f"SELECT {projection} FROM {table}").fetchall(), key=str) == sorted(public.execute(f"SELECT {projection} FROM {table}").fetchall(), key=str)
    output = tmp_path / "new"
    build_sql_bundle(destination=output)
    assert read_json(output / "queries.json") == data["queries"]
    assert sha256(source) == before
    with pytest.raises(ArtifactError, match="already exists"):
        build_sql_bundle(destination=output)


@pytest.mark.parametrize("identity", ["SELECT sum(ride_count) FROM hourly_rides", "total; DELETE FROM hourly_rides", "../sample", "", None])
def test_public_input_accepts_only_query_ids(identity):
    with pytest.raises(ArtifactError, match="approved query ID"):
        run_approved_query(identity)


@pytest.mark.parametrize("sql", ["DELETE FROM hourly_rides", "SELECT 1; SELECT 2",
    "SELECT * FROM read_csv('/etc/passwd')", "SELECT ride_id FROM validated_trips",
    "SELECT count(*) FROM information_schema.tables", "INSTALL httpfs"])
def test_reused_ast_policy_blocks_writes_hidden_columns_and_external_access(sql):
    with connect(DEFAULT_SQL_BUNDLE / "sample.duckdb") as connection:
        with pytest.raises(QueryRejected): validate(sql, connection)


def test_database_is_readonly_and_external_access_disabled():
    with connect(DEFAULT_SQL_BUNDLE / "sample.duckdb") as connection:
        for sql in ["CREATE TABLE forbidden(x INT)", "DELETE FROM hourly_rides", "SELECT * FROM read_csv('/etc/passwd')"]:
            with pytest.raises(duckdb.Error): connection.execute(sql)
        assert connection.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False
        assert connection.execute("SELECT current_setting('threads')").fetchone()[0] == 1


@pytest.mark.parametrize("name", ["manifest.json", "sample.duckdb", "queries.json", "evaluation.json"])
def test_missing_members_fail_closed(copied, name):
    (copied / name).unlink()
    with pytest.raises(ArtifactError): load_sql_bundle(copied)


@pytest.mark.parametrize("name", ["sample.duckdb", "queries.json", "evaluation.json"])
def test_changed_members_fail_integrity(copied, name):
    with (copied / name).open("ab") as stream: stream.write(b"tampered")
    with pytest.raises(ArtifactError, match="checksum"): load_sql_bundle(copied)


def test_even_resigned_registry_cannot_change_approved_sql(copied):
    p = copied / "queries.json"; data = read_json(p)
    data["total"]["sql"] = "SELECT count(*) FROM validated_trips"
    p.write_text(json.dumps(data)); resign(copied, p.name)
    with pytest.raises(ArtifactError, match="approved reference"): load_sql_bundle(copied)


def test_unexpected_worker_results_fail_without_retry(copied):
    p = copied / "queries.json"; data = read_json(p)
    data["total"]["verified_result"]["rows"] = [[6]]  # Synthetic fixture answer is wrong here.
    p.write_text(json.dumps(data)); resign(copied, p.name)
    with pytest.raises(ArtifactError): run_approved_query("total", copied)


def test_corrupt_database_fails_safely_even_with_resigned_receipt(copied):
    p = copied / "sample.duckdb";p.write_bytes(b"not a database")
    resign(copied, p.name)
    with pytest.raises(ArtifactError): run_approved_query("total", copied)


def test_saved_live_evaluation_is_validated_separately(copied):
    p = copied / "evaluation.json"; data = read_json(p)
    data["correct"] = 9
    p.write_text(json.dumps(data)); resign(copied, p.name)
    with pytest.raises(ArtifactError, match="evaluation"): load_sql_bundle(copied)


def test_symlink_manifest_or_database_is_rejected(copied, tmp_path):
    p = copied / "sample.duckdb";outside = tmp_path / "outside.duckdb"
    p.rename(outside);p.symlink_to(outside)
    with pytest.raises(ArtifactError, match="unsafe"): load_sql_bundle(copied)


def test_worker_deadline_kills_and_reaps_without_retry(monkeypatch):
    import citypulse_ai.decision_studio.sql_demo as module
    class Slow:
        def __init__(self): self.killed = False; self.calls = 0
        def communicate(self, timeout=None):
            self.calls += 1
            if timeout is not None: raise subprocess.TimeoutExpired("owned", timeout)
            return "", ""
        def kill(self): self.killed = True
    process = Slow();starts=[]
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: starts.append(kw) or process)
    with pytest.raises(ArtifactError, match="deadline"): run_approved_query("total")
    assert process.killed and process.calls == 2 and len(starts) == 1
    assert set(starts[0]["env"]) == {"PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}


def test_workspace_executes_real_queries_and_clears_stale_display():
    app = AppTest.from_file(str(ENTRY), default_timeout=25).run()
    app.sidebar.radio[0].set_value("SQLPilot").run()
    assert not app.exception and app.title[0].value == "SQLPilot"
    assert len(app.selectbox(key="sql_question").options) == 6
    assert not app.text_input and not app.text_area  # No anonymous free-form SQL/LLM input.
    app.button(key="sql_run").click().run()
    assert not app.exception and app.dataframe[0].value.iloc[0,0] == 1998
    app.selectbox(key="sql_question").set_value("weekday").run()
    assert not app.exception and len(app.dataframe) == 2  # Schema and historical outcomes only.
    app.button(key="sql_run").click().run()
    assert not app.exception and app.dataframe[0].value.iloc[:,1].sum() == 1998
    app.selectbox(key="sql_question").set_value("station_filter").run()
    app.button(key="sql_run").click().run()
    assert not app.exception and any("NULL" in i.value for i in app.info)
    assert app.dataframe[0].value.iloc[0,0] is None


def test_explanations_and_charts_are_grounded_in_verified_results():
    from citypulse_ai.decision_studio.sql_workspace import result_chart
    data = load_sql_bundle()
    for identity in APPROVED_SQL:
        result = data["queries"][identity]["verified_result"]
        assert "sample" in explain_result(identity, result)
    result = data["queries"]["weekday"]["verified_result"]
    chart = result_chart("weekday", result)
    assert sum(chart.data[0].y) == 1998 and len(chart.data[0].x) == 7
    assert "NULL" in explain_result("station_filter", {"rows": [[None]]})
    assert "No result rows" in explain_result("total", {"rows": []})


def test_missing_sql_bundle_does_not_invent_answers(monkeypatch):
    import citypulse_ai.decision_studio.sql_workspace as module
    def missing(): raise ArtifactError("private internal path")
    monkeypatch.setattr(module, "load_sql_bundle", missing)
    app = AppTest.from_file(str(ENTRY), default_timeout=25).run()
    app.sidebar.radio[0].set_value("SQLPilot").run()
    assert not app.exception and any("missing or invalid" in e.value for e in app.error)
    assert not app.button


@pytest.mark.parametrize("result", [{"rows": [], "columns": ["starts"], "latency_seconds": .1}, None])
def test_empty_results_and_execution_failures_are_explicit(monkeypatch, result):
    import citypulse_ai.decision_studio.sql_workspace as module
    def controlled(*a):
        if result is None: raise ArtifactError("private worker failure")
        return deepcopy(result)
    monkeypatch.setattr(module, "run_approved_query", controlled)
    app = AppTest.from_file(str(ENTRY), default_timeout=25).run()
    app.sidebar.radio[0].set_value("SQLPilot").run()
    app.button(key="sql_run").click().run()
    assert not app.exception
    if result is None: assert any("could not complete safely" in e.value for e in app.error)
    else: assert any("No rows matched" in i.value for i in app.info)
