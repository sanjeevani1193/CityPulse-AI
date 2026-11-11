"""Offline artifact, metric, DST and Streamlit integration checks; no models."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import tomllib

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from streamlit.testing.v1 import AppTest

from citypulse_ai.decision_studio.artifacts import (
    ArtifactError, DEFAULT_BUNDLE, MODELS, POINT_COLUMNS, load_bundle, read_json,
    sha256, validate_benchmark, validate_sequences,
)
from citypulse_ai.decision_studio.bundle import ROOT, SOURCE, build_bundle
from citypulse_ai.decision_studio.charts import diagnostic_chart, ranking_chart
from citypulse_ai.decision_studio.metrics import improvement_percent, point_metrics, ranked_rows

ENTRY = ROOT / "apps/decision_studio/streamlit_app.py"


@pytest.fixture
def bundle():
    return load_bundle()


@pytest.fixture
def local_sources():
    # Explicit local-only checks; fresh clones still exercise the bundled app.
    required = [ROOT / SOURCE, ROOT / "outputs/forecasting-v1/artifacts/preprocessing.json",
                ROOT / "outputs/forecasting-v1/diagnostics-final/overall.csv",
                ROOT / "outputs/gcp-pilots/final-december-january-batch-20261023/january-2026/pilot-result.json"]
    if not all(path.is_file() for path in required):
        pytest.skip("Local export reproduction requires retained private source artifacts; no downloads are attempted")


@pytest.fixture
def copy_bundle(tmp_path):
    destination = tmp_path / "bundle"
    shutil.copytree(DEFAULT_BUNDLE, destination)
    return destination


def resign(directory, member):
    manifest = read_json(directory / "manifest.json")
    p = directory / member
    manifest["outputs"][member] = {"size_bytes": p.stat().st_size, "sha256": sha256(p)}
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_real_bundle_scope_schema_and_source_identity(bundle):
    manifest = bundle["manifest"]
    contract = read_json(ROOT / "src/citypulse_ai/api/station_artifacts.json")
    assert manifest["sources"][str(SOURCE)]["sha256"] == contract["diagnostics"]["sha256"]
    assert manifest["sources"][str(SOURCE)]["rows"] == 220800
    assert len(manifest["sources"][str(SOURCE)]["schema"]) == 20
    assert manifest["subset"]["rows"] == 648 and manifest["subset"]["windows"] == 27
    assert len(manifest["subset"]["station_ids"]) == 9
    assert not manifest["subset"]["is_full_evaluation"]
    assert bundle["benchmark"]["benchmark"] == contract["benchmark"]
    assert bundle["benchmark"]["platform"]["final_window_ride_starts"] == 89995902
    assert len(bundle["benchmark"]["station_ids"]) == 100


def test_real_export_preserves_exact_rows_and_excludes_private_fields(bundle, local_sources):
    assert sha256(ROOT / SOURCE) == bundle["manifest"]["sources"][str(SOURCE)]["sha256"]
    subset = pq.read_table(DEFAULT_BUNDLE / "forecast_sequences.parquet").to_pylist()
    desired = {(r["station_id"], r["origin_utc"], r["horizon_step"]): r for r in subset}
    matched = 0
    source = pq.ParquetFile(ROOT / SOURCE)
    stations = pa.array(sorted({r["station_id"] for r in subset}))
    origins = pa.array(sorted({r["origin_utc"] for r in subset}), type=source.schema_arrow.field("origin_utc").type)
    for batch in source.iter_batches(batch_size=8192):
        mask = pc.and_(pc.is_in(batch.column("station_id"), value_set=stations),
                       pc.is_in(batch.column("origin_utc"), value_set=origins))
        for row in batch.filter(mask).to_pylist():
            key = (row["station_id"], row["origin_utc"], row["horizon_step"])
            if key in desired:
                assert row == desired[key]
                matched += 1
    assert matched == 648
    assert pq.read_schema(DEFAULT_BUNDLE / "forecast_sequences.parquet").metadata is None
    text = "\n".join(p.read_text() for p in DEFAULT_BUNDLE.glob("*.json"))
    for prohibited in ("/Users/", "/home/", '"ride_id"', '"api_key"', '"private_key"', "Bearer "):
        assert prohibited not in text


def test_deterministic_create_only_export(tmp_path, local_sources):
    destination = tmp_path / "new"
    build_bundle(destination=destination)
    before = {p.name: sha256(p) for p in destination.iterdir()}
    build_bundle(destination=destination)
    assert before == {p.name: sha256(p) for p in destination.iterdir()}
    (destination / "benchmark.json").write_text("{}")
    with pytest.raises(ArtifactError, match="different bytes"):
        build_bundle(destination=destination)
    assert (destination / "benchmark.json").read_text() == "{}"


def test_wrong_source_hash_fails_before_export(tmp_path, monkeypatch, local_sources):
    import citypulse_ai.decision_studio.bundle as exporter
    changed = tmp_path / "contract.json"
    contract = read_json(ROOT / exporter.CONTRACT)
    contract["diagnostics"]["sha256"] = "0" * 64
    changed.write_text(json.dumps(contract))
    monkeypatch.setattr(exporter, "CONTRACT", changed)
    with pytest.raises(ArtifactError, match="checksum"):
        exporter.build_bundle(destination=tmp_path / "uncreated")
    assert not (tmp_path / "uncreated").exists()


@pytest.mark.parametrize("member", ["manifest.json", "benchmark.json", "diagnostics.json", "forecast_sequences.parquet"])
def test_missing_members_fail_closed(copy_bundle, member):
    (copy_bundle / member).unlink()
    with pytest.raises(ArtifactError):
        load_bundle(copy_bundle)


@pytest.mark.parametrize("member", ["benchmark.json", "diagnostics.json", "forecast_sequences.parquet"])
def test_tampered_members_fail_checksums(copy_bundle, member):
    with (copy_bundle / member).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ArtifactError, match="checksum|size"):
        load_bundle(copy_bundle)


def test_symlink_and_manifest_path_traversal_rejected(copy_bundle, tmp_path):
    p = copy_bundle / "benchmark.json"
    payload = p.read_bytes()
    p.unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(payload)
    p.symlink_to(outside)
    with pytest.raises(ArtifactError, match="unsafe"):
        load_bundle(copy_bundle)
    manifest = read_json(copy_bundle / "manifest.json")
    manifest["outputs"]["../outside.json"] = manifest["outputs"].pop("benchmark.json")
    (copy_bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="files"):
        load_bundle(copy_bundle)


@pytest.mark.parametrize("change", ["missing_model", "nan", "different_targets", "wrong_policy", "rmse", "disclosure", "stations"])
def test_invalid_benchmark_schema(bundle, change):
    data = deepcopy(bundle["benchmark"])
    row = data["benchmark"]["metrics"]["xgboost"]
    if change == "missing_model": del data["benchmark"]["metrics"]["lag-24"]
    elif change == "nan": row["MAE"] = float("nan")
    elif change == "different_targets": row["targets"] = 648
    elif change == "wrong_policy": data["target_policy"] = "physical_demand"
    elif change == "rmse": row["RMSE"] = 0
    elif change == "disclosure": data["benchmark"]["followup_after_prior_test_disclosure"] = False
    else: data["station_ids"] = data["station_ids"][:99]
    with pytest.raises(ArtifactError): validate_benchmark(data)


def test_resigned_bad_diagnostics_rejected(copy_bundle):
    p = copy_bundle / "diagnostics.json"
    diagnostics = read_json(p)
    diagnostics["overall"][0]["MAE"] += 1
    p.write_text(json.dumps(diagnostics)); resign(copy_bundle, p.name)
    with pytest.raises(ArtifactError, match="Diagnostics"):
        load_bundle(copy_bundle)


@pytest.mark.parametrize("change", ["missing_table", "nonfinite", "duplicate_group", "unknown_model"])
def test_malformed_diagnostic_tables(copy_bundle, change):
    p = copy_bundle / "diagnostics.json"
    data = read_json(p)
    if change == "missing_table": del data["hour"]
    elif change == "nonfinite": data["hour"][0]["MAE"] = None
    elif change == "duplicate_group": data["hour"][1]["hour"] = data["hour"][0]["hour"]
    else: data["hour"][0]["model"] = "invented"
    p.write_text(json.dumps(data)); resign(copy_bundle, p.name)
    with pytest.raises(ArtifactError): load_bundle(copy_bundle)


def test_full_metrics_never_replaced_with_demo_metrics(bundle):
    rows = pq.read_table(DEFAULT_BUNDLE / "forecast_sequences.parquet").to_pylist()
    subset = point_metrics([r["actual"] for r in rows], [r["xgb_clipped"] for r in rows])
    full = bundle["benchmark"]["benchmark"]["metrics"]["xgboost"]
    assert subset["targets"] == 648 and full["targets"] == 220800
    assert not math.isclose(subset["MAE"], full["MAE"], abs_tol=1e-8)
    assert full["MAE"] == 3.3663318230240447


def test_dst_fall_back_preserves_two_one_am_hours():
    rows = pq.read_table(DEFAULT_BUNDLE / "forecast_sequences.parquet").to_pylist()
    from zoneinfo import ZoneInfo
    sequence = [r for r in rows if r["station_id"] == rows[0]["station_id"]
                and r["origin_utc"].astimezone(ZoneInfo("America/New_York")).date().isoformat() == "2025-11-02"]
    local = [r["target_hour_utc"].astimezone(ZoneInfo("America/New_York")) for r in sequence]
    assert len(sequence) == 24 and len({r["target_hour_utc"] for r in sequence}) == 24
    assert [h.utcoffset() for h in local if h.hour == 1] == [timedelta(hours=-4), timedelta(hours=-5)]
    assert local[-1].hour == 22


@pytest.mark.parametrize("change", ["missing_hour", "duplicate", "naive", "not_midnight", "wrong_horizon", "untrusted", "nan", "clipping", "numeric_station"])
def test_invalid_forecast_sequences(change):
    rows = deepcopy(pq.read_table(DEFAULT_BUNDLE / "forecast_sequences.parquet").to_pylist()[:24])
    if change == "missing_hour": rows.pop()
    elif change == "duplicate": rows.append(deepcopy(rows[0]))
    elif change == "naive": rows[0]["origin_utc"] = rows[0]["origin_utc"].replace(tzinfo=None)
    elif change == "not_midnight": rows[0]["origin_utc"] += timedelta(hours=1)
    elif change == "wrong_horizon": rows[0]["target_hour_utc"] += timedelta(hours=1)
    elif change == "untrusted": rows[0]["trusted"] = False
    elif change == "nan": rows[0]["actual"] = float("nan")
    elif change == "clipping": rows[0]["xgb_clipped"] += 1
    else: rows[0]["station_id"] = 6140.05
    with pytest.raises(ArtifactError): validate_sequences(rows)


def test_spring_dst_sequence_uses_elapsed_hours():
    rows = deepcopy(pq.read_table(DEFAULT_BUNDLE / "forecast_sequences.parquet").to_pylist()[:24])
    origin = datetime(2025, 3, 9, 5, tzinfo=timezone.utc)
    for i, row in enumerate(rows):
        row.update(origin_utc=origin, target_hour_utc=origin + timedelta(hours=i))
    assert validate_sequences(rows) == 1
    from zoneinfo import ZoneInfo
    assert rows[-1]["target_hour_utc"].astimezone(ZoneInfo("America/New_York")).hour == 0


def test_metric_formulas_and_baseline_improvement():
    result = point_metrics([0, 2, 4], [1, 2, 1])
    assert result["MAE"] == pytest.approx(4/3)
    assert result["RMSE"] == pytest.approx(math.sqrt(10/3))
    assert result["WAPE_percent"] == pytest.approx(100*4/6)
    assert point_metrics([0, 0], [0, 1])["WAPE_percent"] is None
    assert improvement_percent(3, 4) == 25
    assert improvement_percent(5, 4) == -25


@pytest.mark.parametrize("actual,predicted", [([], []), ([1], [1, 2]), ([1], [float('nan')]), ([-1], [1])])
def test_invalid_metric_inputs(actual, predicted):
    with pytest.raises(ValueError): point_metrics(actual, predicted)


def test_rankings_units_and_plot_data(bundle):
    b = bundle["benchmark"]["benchmark"]
    rows = ranked_rows(b, "test", "MAE", list(MODELS))
    assert [r["model_id"] for r in rows] == list(MODELS)
    rmse = ranked_rows(b, "test", "RMSE", list(MODELS))
    assert rmse[2]["model_id"] == "tensorflow-lstm"
    chart = ranking_chart(rows, "MAE", "test")
    assert list(chart.data[0].x) == [r["MAE"] for r in rows]
    assert "station-hour" in chart.layout.xaxis.title.text
    chart = diagnostic_chart(bundle["diagnostics"]["hour"], "hour", ["xgboost"])
    assert len(chart.data) == 1 and len(chart.data[0].x) == 24
    assert "torch" not in sys.modules and "tensorflow" not in sys.modules


def test_streamlit_completed_workspaces_and_interactions():
    app = AppTest.from_file(str(ENTRY), default_timeout=25).run()
    assert not app.exception
    assert len(app.metric) == 4
    assert "90.00M" in [m.value for m in app.metric]
    app.sidebar.radio[0].set_value("Model Comparison").run()
    assert not app.exception and app.title[0].value == "Model Comparison"
    app.selectbox(key="metric").set_value("RMSE").run()
    app.radio(key="split").set_value("validation").run()
    app.multiselect(key="models").set_value(["xgboost", "lag-24"]).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 2
    app.multiselect(key="models").set_value([]).run()
    assert not app.exception and any("Select at least one" in w.value for w in app.warning)


@pytest.mark.parametrize("workspace", ["SQLPilot", "ExperimentLab & Methodology"])
def test_unimplemented_workspaces_are_explicit(workspace):
    app = AppTest.from_file(str(ENTRY), default_timeout=25).run()
    app.sidebar.radio[0].set_value(workspace).run()
    assert not app.exception
    assert any("not implemented" in value.value for value in app.info)
    assert len(app.dataframe) == 0


def test_app_missing_bundle_shows_safe_error(monkeypatch):
    import citypulse_ai.decision_studio.app as module
    def fail():
        raise ArtifactError("Do not display private paths in UI")
    monkeypatch.setattr(module, "load_bundle", fail)
    app = AppTest.from_file(str(ENTRY), default_timeout=25).run()
    assert not app.exception
    assert any("missing or invalid" in e.value for e in app.error)
    assert all("private paths" not in e.value for e in app.error)


def contrast_ratio(foreground, background):
    def luminance(hex_color):
        values = [int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)]
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055)**2.4 for v in values]
        return sum(a*b for a, b in zip(linear, (.2126, .7152, .0722)))
    a, b = sorted((luminance(foreground), luminance(background)))
    return (b + .05) / (a + .05)


def test_native_theme_prevents_mixed_light_dark_surfaces():
    theme = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())["theme"]
    assert theme["base"] == "light"
    for scheme in ("light", "dark"):
        assert theme[scheme]["backgroundColor"] == theme["backgroundColor"]
        assert theme[scheme]["secondaryBackgroundColor"] == "#FFFFFF"
        assert theme[scheme]["textColor"] == theme["textColor"]
    assert contrast_ratio(theme["textColor"], theme["backgroundColor"]) >= 4.5
    sidebar = theme["sidebar"]
    assert contrast_ratio(sidebar["textColor"], sidebar["backgroundColor"]) >= 4.5
    assert contrast_ratio(sidebar["linkColor"], sidebar["backgroundColor"]) >= 4.5
    assert contrast_ratio(sidebar["textColor"], theme["primaryColor"]) >= 4.5
    assert contrast_ratio(theme["dataframeHeaderTextColor"], theme["dataframeHeaderBackgroundColor"]) >= 4.5


def test_scoped_navigation_and_header_styles_preserve_accessibility():
    from citypulse_ai.decision_studio.ui import STYLES
    assert "st-emotion-cache" not in STYLES and ".css-" not in STYLES
    assert 'data-testid="stHeader"' in STYLES
    assert ':has(input:checked)' in STYLES and ':has(input:focus-visible)' in STYLES
    assert "display: none" not in STYLES  # Do not hide navigation/collapse controls.
    assert "prefers-reduced-motion" in STYLES
    # Caption/link overrides must remain readable on the intended surfaces.
    for foreground, background in [("#506676", "#F3F7FA"), ("#CBDCE7", "#102D40"),
                                   ("#9BE6E4", "#102D40"), ("#40596C", "#FFFFFF")]:
        assert contrast_ratio(foreground, background) >= 4.5


def test_plot_theme_stays_legible_without_streamlit_recoloring(bundle):
    rows = ranked_rows(bundle["benchmark"]["benchmark"], "test", "MAE", list(MODELS))
    figure = ranking_chart(rows, "MAE", "test")
    assert figure.layout.paper_bgcolor == figure.layout.plot_bgcolor == "#FFFFFF"
    assert contrast_ratio(figure.layout.font.color, figure.layout.paper_bgcolor) >= 4.5
    assert list(figure.data[0].x) == [r["MAE"] for r in rows]
    assert figure.layout.xaxis.range[1] > max(figure.data[0].x)
    assert not figure.data[0].cliponaxis
