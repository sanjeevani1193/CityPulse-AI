"""Small, framework-independent metric calculations in original count units."""
import math


def point_metrics(actual, predicted):
    """Paired finite observations only; undefined WAPE is None, never zero."""
    actual, predicted = list(actual), list(predicted)
    if not actual or len(actual) != len(predicted):
        raise ValueError("Require nonempty, equally sized paired observations")
    if any(isinstance(x, bool) or not math.isfinite(x) for x in actual + predicted):
        raise ValueError("Observations must be finite numbers")
    if any(x < 0 for x in actual):
        raise ValueError("Recorded counts cannot be negative")
    errors = [p - a for a, p in zip(actual, predicted)]
    absolute = math.fsum(abs(x) for x in errors)
    total = math.fsum(actual)
    return {"targets": len(actual), "MAE": absolute / len(actual),
            "RMSE": math.sqrt(math.fsum(x*x for x in errors) / len(actual)),
            "WAPE_percent": 100 * absolute / total if total else None}


def improvement_percent(value, baseline):
    if not math.isfinite(value) or not math.isfinite(baseline) or value < 0 or baseline <= 0:
        raise ValueError("Require nonnegative error and positive baseline")
    return 100 * (baseline - value) / baseline


def ranked_rows(benchmark, split, metric, selected):
    if split not in ("test", "validation") or metric not in ("MAE", "RMSE", "WAPE_percent"):
        raise ValueError("Unknown split or metric")
    source = benchmark["metrics" if split == "test" else "validation_metrics"]
    rows = [{"model_id": name, **source[name]} for name in selected]
    rows.sort(key=lambda r: (r[metric], r["model_id"]))
    return [{"rank": i + 1, **row} for i, row in enumerate(rows)]
