"""Fail-closed loading of a small, hash-verified, public-safe offline bundle."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

FORMAT = "citypulse-decision-studio-v1"
DEFAULT_BUNDLE = Path(__file__).resolve().parents[3] / "apps/decision_studio/data"
MODELS = {"xgboost": "XGBoost", "pytorch-mlp": "MLP", "pytorch-tft": "TFT",
          "tensorflow-lstm": "LSTM", "lag-24": "Lag-24", "lag-168": "Lag-168"}
POINT_COLUMNS = {"xgboost": "xgb_clipped", "pytorch-mlp": "mlp_clipped",
                 "pytorch-tft": "tft_q50_clipped", "tensorflow-lstm": "lstm_clipped",
                 "lag-24": "lag24", "lag-168": "lag168"}
FORECAST_COLUMNS = {"station_id", "origin_utc", "target_hour_utc", "horizon_step", "actual", "trusted",
                    "tft_q10_raw", "tft_q50_raw", "tft_q90_raw", "lag24", "lag168", "xgb_raw",
                    "xgb_clipped", "tft_q10_clipped", "tft_q50_clipped", "tft_q90_clipped",
                    "mlp_raw", "mlp_clipped", "lstm_raw", "lstm_clipped"}
FILES = {"benchmark.json", "diagnostics.json", "forecast_sequences.parquet"}


class ArtifactError(ValueError):
    """Missing, corrupt or incompatible bundle; never silently invent results."""


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def invalid(value):
        raise ArtifactError("Non-finite JSON value")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def validate_benchmark(data):
    try:
        b = data["benchmark"]
        if data["format_version"] != FORMAT or data["target_policy"] != "accepted_recorded_starts_v1":
            raise ArtifactError("Unsupported benchmark policy/version")
        if b["targets"] != 220800 or not b["followup_after_prior_test_disclosure"]:
            raise ArtifactError("Evaluation scope or follow-up disclosure differs")
        for key in ("metrics", "validation_metrics"):
            if set(b[key]) != set(MODELS):
                raise ArtifactError("Require all six frozen methods")
            for row in b[key].values():
                if row["targets"] != b["targets"]:
                    raise ArtifactError("Methods do not share the frozen target count")
                for metric in ("MAE", "RMSE", "WAPE_percent"):
                    value = row[metric]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ArtifactError("Invalid benchmark metric")
                if row["RMSE"] + 1e-12 < row["MAE"]:
                    raise ArtifactError("RMSE cannot be less than MAE")
        coverage = b["TFT_uncertainty"]["interval_coverage"]
        if not isinstance(coverage, (int, float)) or not 0 <= coverage <= 1:
            raise ArtifactError("Invalid interval coverage")
        stations = data["station_ids"]
        if len(stations) != 100 or len(set(stations)) != 100 or any(not isinstance(s, str) or not s for s in stations):
            raise ArtifactError("Invalid frozen station vocabulary")
    except (KeyError, TypeError) as error:
        raise ArtifactError("Malformed benchmark schema") from error


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ArtifactError("Forecast timestamps must be explicit UTC instants")
    return value


def validate_sequences(rows):
    """24 elapsed hours, exact IDs, midnight origins, no duplicate forecast keys."""
    groups = {}
    last_key = None
    try:
        if not rows:
            raise ArtifactError("Empty forecast subset")
        for row in rows:
            if set(row) != FORECAST_COLUMNS:
                raise ArtifactError("Unexpected forecast schema")
            station = row["station_id"]
            if not isinstance(station, str) or not station:
                raise ArtifactError("Station IDs must remain nonempty strings")
            origin, target = _utc(row["origin_utc"]), _utc(row["target_hour_utc"])
            local = origin.astimezone(ZoneInfo("America/New_York"))
            if (local.hour, local.minute, local.second, local.microsecond) != (0, 0, 0, 0):
                raise ArtifactError("Forecast origin is not NYC-local midnight")
            step = row["horizon_step"]
            if type(step) is not int or not 1 <= step <= 24 or target != origin + timedelta(hours=step-1):
                raise ArtifactError("Invalid elapsed-hour horizon")
            if row["trusted"] is not True or not math.isfinite(row["actual"]) or row["actual"] < 0 or row["actual"] % 1:
                raise ArtifactError("Untrusted/invalid actual count")
            for col in FORECAST_COLUMNS - {"station_id", "origin_utc", "target_hour_utc", "horizon_step", "trusted"}:
                if not math.isfinite(row[col]):
                    raise ArtifactError("Non-finite forecast value")
            for col in POINT_COLUMNS.values():
                if row[col] < 0:
                    raise ArtifactError("Fixed clipped forecasts cannot be negative")
            for prefix in ("xgb", "mlp", "lstm", "tft_q10", "tft_q50", "tft_q90"):
                if row[prefix + "_clipped"] != max(0, row[prefix + "_raw"]):
                    raise ArtifactError("Clipping differs from frozen rule")
            if not row["tft_q10_clipped"] <= row["tft_q50_clipped"] <= row["tft_q90_clipped"]:
                raise ArtifactError("Crossed TFT quantiles")
            steps = groups.setdefault((station, origin), set())
            if step in steps:
                raise ArtifactError("Duplicate forecast key")
            key = (station, origin, step)
            if last_key is not None and key <= last_key:
                raise ArtifactError("Forecast keys must be in canonical UTC order")
            last_key = key
            steps.add(step)
        if any(steps != set(range(1, 25)) for steps in groups.values()):
            raise ArtifactError("Incomplete 24-hour sequence")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError("Malformed forecast sequence") from error
    return len(groups)


def validate_diagnostics(diagnostics, benchmark):
    dimensions = {"overall": None, "station_id": "station_id", "hour": "hour", "day_type": "day_type",
                  "month": "month", "activity": "activity", "period": "period", "station_distribution": None}
    if set(diagnostics) != set(dimensions):
        raise ArtifactError("Missing diagnostic tables")
    names = {name: key for key, name in MODELS.items()}
    for table, dimension in dimensions.items():
        records = diagnostics[table]
        if not isinstance(records, list) or not records:
            raise ArtifactError("Empty diagnostic table")
        for name, model_id in names.items():
            group = [r for r in records if r["model"] == name]
            if not group:
                raise ArtifactError("Missing diagnostic method")
            if table == "station_distribution":
                if len(group) != 1 or group[0]["stations"] != 100:
                    raise ArtifactError("Invalid station distribution")
            else:
                if sum(r["targets"] for r in group) != benchmark["benchmark"]["targets"]:
                    raise ArtifactError("Diagnostic cohort differs")
                if dimension:
                    values = [r[dimension] for r in group]
                    if len(set(values)) != len(values):
                        raise ArtifactError("Duplicate diagnostic group")
                    if table == "station_id" and set(values) != set(benchmark["station_ids"]):
                        raise ArtifactError("Diagnostic station vocabulary differs")
                elif len(group) != 1:
                    raise ArtifactError("Duplicate overall method")
                for row in group:
                    if type(row["targets"]) is not int or row["targets"] <= 0:
                        raise ArtifactError("Invalid diagnostic target count")
                    if not math.isfinite(row["MAE"]) or row["MAE"] < 0 or not math.isfinite(row["mean_signed_error"]):
                        raise ArtifactError("Invalid diagnostic error")
            if table == "overall":
                expected = benchmark["benchmark"]["metrics"][model_id]
                for key in ("MAE", "RMSE", "WAPE_percent"):
                    if not math.isclose(group[0][key], expected[key], rel_tol=0, abs_tol=1e-12):
                        raise ArtifactError("Diagnostics differ from frozen full-test metrics")
        if any(r["model"] not in names for r in records):
            raise ArtifactError("Unknown diagnostic method")


def load_bundle(directory=DEFAULT_BUNDLE):
    """Only fixed local files; no download, credential lookup or model imports."""
    import pyarrow.parquet as pq

    directory = Path(directory)
    try:
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 128 * 1024:
            raise ArtifactError("Invalid bundle manifest")
        manifest = read_json(manifest_path)
        if manifest["format_version"] != FORMAT or set(manifest["outputs"]) != FILES:
            raise ArtifactError("Unknown bundle version/files")
        if manifest["evaluation_scope"] != "frozen_full_test" or manifest["subset"]["is_full_evaluation"]:
            raise ArtifactError("Full metrics and demonstration scope must remain separate")
        for name, expected in manifest["outputs"].items():
            path = directory / name
            if path.is_symlink() or not 0 < path.stat().st_size <= 5 * 1024**2:
                raise ArtifactError("Bundle member is unsafe or oversized")
            if path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
                raise ArtifactError("Bundle checksum or size mismatch: " + name)
        data = read_json(directory / "benchmark.json")
        validate_benchmark(data)
        diagnostics = read_json(directory / "diagnostics.json")
        table = pq.read_table(directory / "forecast_sequences.parquet")
        if table.num_rows != manifest["subset"]["rows"] or table.num_rows > 10000:
            raise ArtifactError("Unexpected demo row count")
        rows = table.to_pylist()
        windows = validate_sequences(rows)
        if windows != manifest["subset"]["windows"]:
            raise ArtifactError("Unexpected sequence count")
        subset_stations = sorted({r["station_id"] for r in rows})
        if subset_stations != manifest["subset"]["station_ids"] or not set(subset_stations) <= set(data["station_ids"]):
            raise ArtifactError("Subset station identity differs")
        validate_diagnostics(diagnostics, data)
        return {"manifest": manifest, "benchmark": data, "diagnostics": diagnostics}
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError("Demo bundle missing or malformed; regenerate from verified local sources") from error
