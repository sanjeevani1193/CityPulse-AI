"""Select and describe frozen forecast sequences; never generate predictions."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .artifacts import ArtifactError, MODELS, POINT_COLUMNS, validate_sequences
from .metrics import point_metrics

NYC = ZoneInfo("America/New_York")


def utc_origin(value):
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("Require an explicit UTC origin")
        return value.astimezone(timezone.utc)
    except (TypeError, ValueError) as error:
        raise ArtifactError("Require a valid UTC forecast origin") from error


def station_ids(rows):
    result = sorted({row["station_id"] for row in rows})
    if not result or any(not isinstance(s, str) or not s for s in result):
        raise ArtifactError("No valid station IDs available")
    return result


def available_origins(rows, station):
    if not isinstance(station, str) or station not in station_ids(rows):
        raise ArtifactError("Station is not in the verified demonstration subset")
    return sorted({utc_origin(row["origin_utc"]) for row in rows if row["station_id"] == station})


def select_sequence(rows, station, origin):
    origin = utc_origin(origin)
    if origin not in available_origins(rows, station):
        raise ArtifactError("Forecast origin is not available for this station")
    sequence = sorted((dict(r) for r in rows if r["station_id"] == station and utc_origin(r["origin_utc"]) == origin),
                      key=lambda r: r["horizon_step"])
    if validate_sequences(sequence) != 1:
        raise ArtifactError("Require exactly one complete trusted forecast sequence")
    return sequence


def time_label(value, zone="New York"):
    value = utc_origin(value)
    if zone == "UTC":
        return value.isoformat(timespec="minutes").replace("+00:00", "Z")
    if zone != "New York":
        raise ValueError("Unknown display timezone")
    local = value.astimezone(NYC)
    return local.isoformat(timespec="minutes") + " (" + local.tzname() + ")"


def display_rows(sequence, models):
    return [{"Horizon": r["horizon_step"], "Target UTC": time_label(r["target_hour_utc"], "UTC"),
             "Target New York": time_label(r["target_hour_utc"]), "Actual starts": r["actual"],
             **{MODELS[m]: r[POINT_COLUMNS[m]] for m in models},
             **({"TFT q10": r["tft_q10_clipped"], "TFT q90": r["tft_q90_clipped"]} if "pytorch-tft" in models else {})}
            for r in sequence]


def window_errors(sequence, models):
    """Signed error = saved prediction - trusted actual; no official metrics edits."""
    if validate_sequences(sequence) != 1:
        raise ArtifactError("Window diagnostics require one complete trusted sequence")
    if len(set(models)) != len(models) or any(m not in MODELS for m in models):
        raise ArtifactError("Unknown or duplicate forecasting method")
    details, summaries = [], []
    actual = [r["actual"] for r in sequence]
    for model in models:
        predicted = [r[POINT_COLUMNS[model]] for r in sequence]
        errors = [p - a for p, a in zip(predicted, actual)]
        largest = max(range(24), key=lambda i: abs(errors[i]))  # Earliest step wins ties.
        summaries.append({"model_id": model, "targets": len(actual),
                          "MAE": point_metrics(actual, predicted)["MAE"],
                          "mean_error": sum(errors) / len(errors),
                          "overprediction_hours": sum(e > 0 for e in errors),
                          "underprediction_hours": sum(e < 0 for e in errors),
                          "exact_hours": sum(e == 0 for e in errors),
                          "largest_absolute_error": abs(errors[largest]),
                          "largest_error_step": sequence[largest]["horizon_step"],
                          "largest_error_actual": actual[largest], "largest_error_prediction": predicted[largest]})
        details.extend({"model_id": model, "horizon_step": row["horizon_step"],
                        "target_hour_utc": row["target_hour_utc"], "actual": a, "prediction": p,
                        "error": e, "absolute_error": abs(e)}
                       for row, a, p, e in zip(sequence, actual, predicted, errors))
    return summaries, details
