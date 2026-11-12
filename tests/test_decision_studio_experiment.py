"""Saved synthetic evidence, independent inference checks and five-workspace UI."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from citypulse_ai.decision_studio.artifacts import ArtifactError, sha256
from citypulse_ai.decision_studio.experiment_data import (
    DEFAULT_EXPERIMENT, conclusions, load_experiment, p_value_label, station_rows, validate_experiment,
)
from citypulse_ai.decision_studio.experiment_charts import effect_chart, planning_chart, station_chart

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "apps/decision_studio/streamlit_app.py"


@pytest.fixture
def experiment():
    return load_experiment()["experiment"]


def app():
    page = AppTest.from_file(str(ENTRY), default_timeout=25).run()
    page.sidebar.radio[0].set_value("ExperimentLab & Methodology").run()
    assert not page.exception
    return page


def test_bundle_matches_recorded_source_and_tracked_report():
    data = load_experiment()
    source = data["manifest"]["sources"]["outputs/experimentlab/release-20261030-final/experiment.json"]
    assert source == data["manifest"]["outputs"]["experiment.json"]
    assert sha256(DEFAULT_EXPERIMENT / "experiment.json") == "fefe5b941ed2a7618b221cf51db9a3c7fccc195cb36ac0016f5fac01135ba527"
    assert sha256(ROOT / "docs/experimentlab_report.html") == data["manifest"]["sources"]["docs/experimentlab_report.html"]["sha256"]
    assert sum(p.stat().st_size for p in DEFAULT_EXPERIMENT.iterdir()) < 30_000


@pytest.mark.parametrize("fault", ["missing", "corrupt", "symlink", "manifest", "oversize"])
def test_bad_artifacts_fail_closed(tmp_path, fault):
    folder = tmp_path / "experiment"
    shutil.copytree(DEFAULT_EXPERIMENT, folder)
    path = folder / "experiment.json"
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_bytes(path.read_bytes()+b" ")
    elif fault == "symlink":
        path.unlink(); path.symlink_to(DEFAULT_EXPERIMENT / "experiment.json")
    elif fault == "oversize":
        path.write_bytes(b" "*131073)
    else:
        (folder / "manifest.json").write_text("{}")
    with pytest.raises(ArtifactError):
        load_experiment(folder)


@pytest.mark.parametrize("fault", ["missing_source", "different_identity", "bad_hash"])
def test_invalid_provenance_fails_closed(tmp_path, fault):
    folder = tmp_path / "experiment"
    shutil.copytree(DEFAULT_EXPERIMENT, folder)
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    if fault == "missing_source": manifest["sources"].pop("docs/experimentlab_report.html")
    elif fault == "different_identity":
        manifest["sources"]["outputs/experimentlab/release-20261030-final/experiment.json"]["size_bytes"] += 1
    else: manifest["sources"]["docs/experimentlab_report.html"]["sha256"] = "bad"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError): load_experiment(folder)


def test_export_is_create_only_and_byte_identical(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("experiment_export", ROOT / "scripts/build_experiment_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Public bundle/HTML are sufficient fixtures; no private artifacts required.
    sources = tmp_path / "source"
    release = sources / module.SOURCE
    release.mkdir(parents=True)
    (sources / "docs").mkdir()
    shutil.copyfile(DEFAULT_EXPERIMENT / "experiment.json", release / "experiment.json")
    for destination in (release / "report.html", sources / "docs/experimentlab_report.html"):
        shutil.copyfile(ROOT / "docs/experimentlab_report.html", destination)
    (release / "checksums.json").write_text(json.dumps({p.name: sha256(p) for p in release.iterdir()}))
    module.ROOT = sources
    destination = tmp_path / "export"
    module.build(destination)
    assert (destination / "experiment.json").read_bytes() == (DEFAULT_EXPERIMENT / "experiment.json").read_bytes()
    load_experiment(destination)
    with pytest.raises(ArtifactError): module.build(destination)
    (release / "report.html").write_text("changed")
    with pytest.raises(ArtifactError): module.build(tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("field,value", [
    ("data_kind", "historical observations"), ("periods", {}),
    ("station_records", []), ("variance_reduction_fraction", .5),
])
def test_invalid_study_semantics_rejected(experiment, field, value):
    experiment[field] = value
    with pytest.raises(ArtifactError):
        validate_experiment(experiment)


@pytest.mark.parametrize("fault", ["duplicate_id", "assignment", "total", "effect", "pvalue", "nan", "fit_scope"])
def test_corrupt_record_or_inference_rejected(experiment, fault):
    if fault == "duplicate_id": experiment["station_records"][0]["station_id"] = "SIM0001"
    elif fault == "assignment": experiment["station_records"][0]["treated"] = "true"
    elif fault == "total": experiment["station_records"][0]["post_total_starts"] += 1
    elif fault == "effect": experiment["cuped"]["effect"] += .1
    elif fault == "pvalue": experiment["cuped"]["p_value"] = 1.1
    elif fault == "nan": experiment["calibration"]["theta"] = float("nan")
    else: experiment["calibration"]["fit_scope"] = "fit on post outcomes"
    with pytest.raises(ArtifactError): validate_experiment(experiment)


@pytest.mark.parametrize("key,column", [("unadjusted", "observed_post"), ("cuped", "adjusted_post")])
def test_saved_welch_inference_reproduced_without_simulation(experiment, monkeypatch, key, column):
    from citypulse_ai.experiments import redistribution as research
    def forbidden(*args, **kwargs):
        raise AssertionError("No new simulation or calibration fit permitted")
    for name in ("simulate", "simulation_check", "analyze", "fit_cuped"):
        monkeypatch.setattr(research, name, forbidden)
    rows = station_rows(experiment)
    y = np.array([r[column] for r in rows])
    z = np.array([r["arm"] == "Treatment" for r in rows])
    result = research.compare(y, z, experiment["design"]["alpha"])
    for name in ("effect", "standard_error", "ci_low", "ci_high", "p_value", "df", "treatment_mean", "control_mean"):
        assert result[name] == pytest.approx(experiment[key][name], rel=1e-10, abs=1e-12)
    # Separate manual Welch equations and Student-t inversion.
    from scipy.stats import t
    vt, vc = y[z].var(ddof=1)/50, y[~z].var(ddof=1)/50
    se = (vt+vc)**.5
    df = (vt+vc)**2/(vt**2/49+vc**2/49)
    effect = y[z].mean()-y[~z].mean()
    assert 2*t.sf(abs(effect/se), df) == pytest.approx(experiment[key]["p_value"])
    assert effect-t.ppf(.975, df)*se == pytest.approx(experiment[key]["ci_low"])


@pytest.mark.parametrize("key", ["planning_raw", "planning_cuped"])
def test_saved_planning_reproduced_from_frozen_sd(experiment, key):
    from citypulse_ai.experiments.redistribution import planning
    saved = experiment[key]
    result = planning(50, saved["sd"], target_effect=2, target_power=.8, alpha=.05)
    for name in ("power", "mde", "required_stations_per_arm"):
        assert result[name] == pytest.approx(saved[name])


def test_assignment_and_cuped_transform_preserve_recorded_evidence(experiment):
    from citypulse_ai.experiments.redistribution import assign
    records = experiment["station_records"]
    recorded = [r["treated"] for r in records]
    assert list(assign([r["station_id"] for r in records], experiment["design"]["assignment_seed"])) == recorded
    rows = station_rows(experiment)
    assert len(rows) == 100 and sum(r["arm"] == "Treatment" for r in rows) == 50
    assert np.mean([r["adjusted_post"] for r in rows]) == pytest.approx(np.mean([r["observed_post"] for r in rows]))
    assert all(r["post_daily_starts"] == r["post_total_starts"]/28 for r in records)


def test_statistical_and_practical_significance_are_separate(experiment):
    assert conclusions(experiment["cuped"], 2, .05) == {
        "statistically_positive": True, "practical_threshold_supported": False}
    assert conclusions(experiment["unadjusted"], 2, .05)["statistically_positive"] is False
    row = dict(experiment["cuped"], ci_low=2)
    assert not conclusions(row, 2, .05)["practical_threshold_supported"]
    row["ci_low"] = 2.01
    assert conclusions(row, 2, .05)["practical_threshold_supported"]
    assert p_value_label(experiment["cuped"]["p_value"]) == "8.03e-07"
    assert p_value_label(experiment["unadjusted"]["p_value"]) == "0.66666"


def test_charts_use_exact_saved_effects_and_planning(experiment):
    figure = effect_chart(experiment)
    for trace, key in zip(figure.data, ("cuped", "unadjusted")):
        row = experiment[key]
        assert trace.x[0] == row["effect"]
        assert trace.error_x.array[0]+trace.x[0] == pytest.approx(row["ci_high"])
        assert trace.x[0]-trace.error_x.arrayminus[0] == pytest.approx(row["ci_low"])
    assert len(effect_chart(experiment, "unadjusted", False).data) == 1
    rows = station_rows(experiment)
    assert sum(len(t.y) for t in station_chart(rows).data) == 100
    for metric in ("power", "mde"):
        assert list(planning_chart(experiment, metric).data[0].y) == [experiment[k][metric] for k in ("planning_raw", "planning_cuped")]


def test_workspace_disclosure_metrics_and_interactions():
    page = app()
    assert any("Fully synthetic" in w.value for w in page.warning)
    metrics = {m.label: m.value for m in page.metric}
    assert metrics["Effect (starts / station / day)"] == "+2.7531"
    assert metrics["95% Welch confidence interval"] == "[1.7190, 3.7871]"
    assert metrics["Two-sided p-value"] == "8.03e-07"
    assert any("practical threshold uncertain" in i.value for i in page.info)
    page.selectbox[0].set_value("unadjusted").run()
    assert not page.exception
    assert {m.label: m.value for m in page.metric}["Effect (starts / station / day)"] == "+2.8086"
    page.checkbox[0].uncheck().run()
    assert not page.exception
    page.selectbox[1].set_value("Treatment").run()
    assert not page.exception and len(page.dataframe[1].value) == 50
    page.checkbox[1].check().run()
    page.selectbox[2].set_value("mde").run()
    assert not page.exception
    assert any("CUPED remains" in c.value for c in page.caption)


def test_workspace_missing_artifact_is_safe_and_independent(monkeypatch):
    import citypulse_ai.decision_studio.experiment_workspace as module
    def fail(): raise ArtifactError("private details not for display")
    monkeypatch.setattr(module, "load_experiment", fail)
    page = app()
    assert any("missing or invalid" in e.value for e in page.error)
    assert not page.metric
    assert not any("private details" in e.value for e in page.error)


def test_experiment_independent_of_forecast_bundle(monkeypatch):
    import citypulse_ai.decision_studio.app as routing
    def fail(): raise ArtifactError("forecast unavailable")
    monkeypatch.setattr(routing, "load_bundle", fail)
    page = app()
    assert any(m.value == "+2.7531" for m in page.metric)


def test_runtime_loader_has_no_scipy_simulation_or_cloud_imports():
    result = subprocess.run([sys.executable, "-c",
        "import sys; from citypulse_ai.decision_studio.experiment_data import load_experiment; "
        "load_experiment(); assert not any(x in sys.modules for x in "
        "('scipy', 'numpy', 'citypulse_ai.experiments.redistribution', 'google.cloud', 'torch', 'tensorflow'))"],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
