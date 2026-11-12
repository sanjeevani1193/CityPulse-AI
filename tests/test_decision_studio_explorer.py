"""Saved forecast selection and UI checks; no inference, cloud or browser."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from citypulse_ai.decision_studio.artifacts import ArtifactError, MODELS, load_bundle
from citypulse_ai.decision_studio.charts import forecast_chart, horizon_error_chart
from citypulse_ai.decision_studio.forecasts import (
    available_origins, display_rows, select_sequence, station_ids, time_label, window_errors,
)

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "apps/decision_studio/streamlit_app.py"


@pytest.fixture
def bundle():
    return load_bundle()


@pytest.fixture
def sequence(bundle):
    rows = bundle["forecasts"]
    station = station_ids(rows)[0]
    return select_sequence(rows, station, available_origins(rows, station)[0])


@pytest.mark.parametrize("station_index", range(9))
def test_all_nine_stations_have_three_exact_complete_origins(bundle, station_index):
    rows = bundle["forecasts"]
    stations = station_ids(rows)
    assert stations == bundle["manifest"]["subset"]["station_ids"] and len(stations) == 9
    station = stations[station_index]
    origins = available_origins(rows, station)
    assert len(origins) == 3 and origins == sorted(origins)
    for origin in origins:
        result = select_sequence(rows, station, origin.isoformat())
        assert len(result) == 24 and [r["horizon_step"] for r in result] == list(range(1, 25))
        assert {r["station_id"] for r in result} == {station}
        assert all(r["trusted"] is True and r["actual"] >= 0 for r in result)
        assert all(r["target_hour_utc"] == origin + timedelta(hours=r["horizon_step"]-1) for r in result)
        assert result == [r for r in rows if r["station_id"] == station and r["origin_utc"] == origin]


@pytest.mark.parametrize("station,origin", [("missing", "2025-10-01T04:00:00Z"),
    (6140.05, "2025-10-01T04:00:00Z"), ("4395.07", "not-a-time")])
def test_invalid_station_and_origin_fail_closed(bundle, station, origin):
    with pytest.raises(ArtifactError): select_sequence(bundle["forecasts"], station, origin)


def test_unavailable_and_naive_origins_fail_closed(bundle):
    rows = bundle["forecasts"];station = station_ids(rows)[0]
    for origin in ["2025-10-02T04:00:00Z", "2025-10-01T04:00:00", None]:
        with pytest.raises(ArtifactError): select_sequence(rows, station, origin)


def test_selection_is_deterministic_and_does_not_mutate_saved_rows(bundle):
    rows = bundle["forecasts"]; before = deepcopy(rows)
    station = station_ids(rows)[0];origin = available_origins(rows, station)[0]
    first = select_sequence(list(reversed(rows)), station, origin)
    second = select_sequence(rows, station, origin)
    assert first == second
    first[0]["actual"] = -100
    assert rows == before


@pytest.mark.parametrize("damage", ["missing", "duplicate", "untrusted", "bad_quantile", "bad_prediction"])
def test_invalid_sequences_cannot_produce_diagnostics(sequence, damage):
    rows = deepcopy(sequence)
    if damage == "missing": rows.pop()
    elif damage == "duplicate": rows.append(deepcopy(rows[-1]))
    elif damage == "untrusted": rows[0]["trusted"] = False
    elif damage == "bad_quantile":
        rows[0]["tft_q10_raw"] = rows[0]["tft_q10_clipped"] = rows[0]["tft_q90_clipped"] + 1
    else: rows[0]["xgb_clipped"] = float("nan")
    with pytest.raises(ArtifactError, match="Crossed" if damage == "bad_quantile" else None):
        window_errors(rows, ["xgboost"])


def test_error_math_signs_ties_and_target_scope(sequence, bundle):
    rows = deepcopy(sequence)
    for i, row in enumerate(rows):
        row["actual"] = 2.
        row["xgb_raw"] = row["xgb_clipped"] = float(1 + i % 3)
    summaries, details = window_errors(rows, ["xgboost"])
    s = summaries[0]
    assert s["targets"] == 24 and s["MAE"] == pytest.approx(2/3)
    assert s["mean_error"] == 0 and s["overprediction_hours"] == s["underprediction_hours"] == s["exact_hours"] == 8
    assert s["largest_absolute_error"] == 1 and s["largest_error_step"] == 1
    assert [r["error"] for r in details[:3]] == [-1, 0, 1]
    assert bundle["benchmark"]["benchmark"]["metrics"]["xgboost"]["targets"] == 220800


def test_unknown_or_duplicate_models_are_rejected(sequence):
    for models in [["not-a-model"], ["xgboost", "xgboost"]]:
        with pytest.raises(ArtifactError): window_errors(sequence, models)
    assert window_errors(sequence, []) == ([], [])


def test_fall_back_utc_identity_local_labels_and_chart_band(bundle):
    rows = bundle["forecasts"];station = station_ids(rows)[0]
    result = select_sequence(rows, station, "2025-11-02T04:00:00Z")
    labels = display_rows(result, ["pytorch-tft"])
    assert "01:00-04:00" in labels[1]["Target New York"] and "EDT" in labels[1]["Target New York"]
    assert "01:00-05:00" in labels[2]["Target New York"] and "EST" in labels[2]["Target New York"]
    assert labels[1]["Target UTC"] != labels[2]["Target UTC"]
    assert "22:00-05:00" in labels[-1]["Target New York"]
    figure = forecast_chart(result, ["pytorch-tft"])
    assert len(figure.data) == 4  # q10, q90 band, q50, actual.
    assert figure.data[1].fill == "tonexty"
    assert list(figure.data[0].y) == [r["tft_q10_clipped"] for r in result]
    assert list(figure.data[1].y) == [r["tft_q90_clipped"] for r in result]
    assert len(set(figure.data[-1].x)) == 24 and figure.layout.xaxis.type == "date"
    assert "01:00-04:00" in figure.data[-1].customdata[1][0]
    assert "01:00-05:00" in figure.data[-1].customdata[2][0]
    assert figure.data[2].customdata[0][5] == result[0]["tft_q10_clipped"]
    assert figure.data[2].customdata[0][6] == result[0]["tft_q90_clipped"]


def test_all_model_charts_use_authentic_points_and_signed_errors(sequence):
    from citypulse_ai.decision_studio.artifacts import POINT_COLUMNS
    models = list(MODELS)
    figure = forecast_chart(sequence, models, zone="UTC", show_band=False)
    assert len(figure.data) == 7
    for trace, model in zip(figure.data, models):
        assert list(trace.y) == [r[POINT_COLUMNS[model]] for r in sequence]
        assert len(trace.x) == 24
    assert all("UTC" in label for label in figure.layout.xaxis.ticktext)
    summaries, details = window_errors(sequence, models)
    errors = horizon_error_chart(details, models)
    assert len(errors.data) == len(summaries) == 6
    assert list(errors.data[0].x) == list(range(1, 25))
    assert list(errors.data[0].y) == [r["xgb_clipped"] - r["actual"] for r in sequence]
    assert errors.layout.shapes[0].y0 == 0


def test_timestamp_labels_reject_ambiguous_display_inputs():
    assert time_label(datetime(2025, 10, 1, 4, tzinfo=timezone.utc), "UTC") == "2025-10-01T04:00Z"
    with pytest.raises(ArtifactError): time_label("2025-11-02T01:00:00")
    with pytest.raises(ValueError): time_label("2025-11-02T04:00:00Z", "unknown")


def test_explorer_renders_selection_changes_and_empty_models():
    app = AppTest.from_file(str(ENTRY), default_timeout=25).run()
    app.sidebar.radio[0].set_value("Forecast Explorer").run()
    assert not app.exception and app.title[0].value == "Forecast Explorer"
    assert len(app.selectbox(key="forecast_station").options) == 9
    assert len(app.selectbox(key="forecast_origin").options) == 3
    app.selectbox(key="forecast_origin").set_value("2025-11-02T04:00:00+00:00").run()
    assert not app.exception and any("daylight-saving" in i.value for i in app.info)
    app.multiselect(key="forecast_models").set_value(list(MODELS)).run()
    assert not app.exception and len(app.dataframe[0].value) == 6
    app.radio(key="forecast_zone").set_value("UTC").run()
    app.checkbox(key="forecast_band").uncheck().run()
    other = app.selectbox(key="forecast_station").options[-1]
    app.selectbox(key="forecast_station").set_value(other).run()
    assert not app.exception
    app.multiselect(key="forecast_models").set_value([]).run()
    assert not app.exception and any("Only actual" in w.value for w in app.warning)
    assert len(app.dataframe) == 1  # Details only, never invented model scores.


def test_explorer_missing_sequence_has_safe_error(bundle, monkeypatch):
    import citypulse_ai.decision_studio.app as module
    damaged = deepcopy(bundle)
    damaged["forecasts"].pop(0)
    monkeypatch.setattr(module, "load_bundle", lambda: damaged)
    app = AppTest.from_file(str(ENTRY), default_timeout=25).run()
    app.sidebar.radio[0].set_value("Forecast Explorer").run()
    assert not app.exception and any("missing or invalid" in e.value for e in app.error)
