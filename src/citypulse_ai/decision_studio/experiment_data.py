"""Verify saved synthetic evidence; no simulation, SciPy or cloud at runtime."""
import math
from pathlib import Path
import statistics

from .artifacts import ArtifactError, read_json, sha256

DEFAULT_EXPERIMENT = Path(__file__).resolve().parents[3] / "apps/decision_studio/experiment_data"
FORMAT = "citypulse-experiment-studio-v1"
ESTIMATORS = {"cuped": "CUPED primary", "unadjusted": "Unadjusted sensitivity"}


def station_rows(experiment):
    """Display-only CUPED transform using the saved, pre-fitted coefficient."""
    records = experiment["station_records"]
    center = statistics.fmean(r["baseline_daily_starts"] for r in records)
    theta = experiment["calibration"]["theta"]
    return [{"station_id": r["station_id"], "arm": "Treatment" if r["treated"] else "Control",
             "baseline": r["baseline_daily_starts"], "observed_post": r["post_daily_starts"],
             "adjusted_post": r["post_daily_starts"] - theta*(r["baseline_daily_starts"]-center),
             "post_total_starts": r["post_total_starts"]} for r in records]


def welch_moments(values, arms):
    """Independent algebra check, not a new p-value/CI calculation."""
    treatment = [v for v, arm in zip(values, arms) if arm == "Treatment"]
    control = [v for v, arm in zip(values, arms) if arm == "Control"]
    if len(values) != len(arms) or len(treatment)+len(control) != len(values) or min(len(treatment), len(control)) < 2:
        raise ArtifactError("Invalid experiment groups")
    vt, vc = statistics.variance(treatment)/len(treatment), statistics.variance(control)/len(control)
    if vt+vc <= 0:
        raise ArtifactError("Degenerate experiment variance")
    return {"treatment_mean": statistics.fmean(treatment), "control_mean": statistics.fmean(control),
            "effect": statistics.fmean(treatment)-statistics.fmean(control),
            "standard_error": math.sqrt(vt+vc),
            "df": (vt+vc)**2/(vt**2/(len(treatment)-1)+vc**2/(len(control)-1))}


def conclusions(row, threshold, alpha):
    """The saved primary result remains primary regardless of UI selection."""
    return {"statistically_positive": row["p_value"] < alpha and row["ci_low"] > 0,
            "practical_threshold_supported": row["ci_low"] > threshold}


def p_value_label(value):
    return f"{value:.2e}" if value < .001 else f"{value:.5f}"


def _same(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ArtifactError("Saved experiment algebra differs: " + label)


def validate_experiment(data):
    try:
        if data["version"] != "experimentlab-redistribution-v1" or data["data_kind"] != "fully synthetic; no historical observations":
            raise ArtifactError("Require the saved synthetic study, not historical outcomes")
        design, records = data["design"], data["station_records"]
        if design["stations"] != 100 or design["days"] != 28 or len(records) != 100:
            raise ArtifactError("Unexpected saved study size")
        if not 0 < design["alpha"] < 1 or design["practical_threshold"] <= 0:
            raise ArtifactError("Invalid saved inference thresholds")
        if {r["station_id"] for r in records} != {f"SIM{i:04d}" for i in range(100)}:
            raise ArtifactError("Synthetic station IDs must be exact and unique")
        if any(type(r["treated"]) is not bool for r in records) or sum(r["treated"] for r in records) != 50:
            raise ArtifactError("Require the recorded balanced 50/50 assignment")
        for r in records:
            if any(not math.isfinite(r[k]) or r[k] < 0 for k in ("baseline_daily_starts", "post_daily_starts", "post_total_starts")):
                raise ArtifactError("Invalid synthetic observations")
            if type(r["post_total_starts"]) is not int:
                raise ArtifactError("Synthetic post-period totals must be integers")
            _same(r["post_daily_starts"], r["post_total_starts"]/design["days"], "daily outcome")
        if data["periods"] != {"calibration_x": [0,28], "calibration_y": [28,56], "baseline": [56,84], "post": [84,112]}:
            raise ArtifactError("Calibration/baseline/post periods must be ordered and disjoint")
        if data["calibration"]["fit_scope"] != "pre_period_1 -> pre_period_2; before assignment and outcome":
            raise ArtifactError("CUPED coefficient must be pre-calibrated")
        if not all(math.isfinite(data["calibration"][k]) for k in ("theta", "calibration_sd", "calibration_adjusted_sd")):
            raise ArtifactError("Non-finite calibration values")
        rows = station_rows(data)
        for key, column in (("unadjusted", "observed_post"), ("cuped", "adjusted_post")):
            row = data[key]
            if not all(math.isfinite(row[k]) for k in ("treatment_mean", "control_mean", "treatment_sd", "control_sd",
                                                       "effect", "standard_error", "ci_low", "ci_high", "p_value", "df")):
                raise ArtifactError("Non-finite saved inference")
            moments = welch_moments([r[column] for r in rows], [r["arm"] for r in rows])
            for name, value in moments.items():
                _same(row[name], value, name)
            if row["treatment_n"] != 50 or row["control_n"] != 50:
                raise ArtifactError("Inferential sample sizes differ")
            if not 0 <= row["p_value"] <= 1 or not row["ci_low"] < row["effect"] < row["ci_high"]:
                raise ArtifactError("Invalid saved p-value or confidence interval")
            _same(row["ci_high"]-row["effect"], row["effect"]-row["ci_low"], "two-sided interval")
            if row["significant"] is not (row["p_value"] < design["alpha"]):
                raise ArtifactError("Significance flag differs from saved alpha")
        _same(data["variance_reduction_fraction"], 1-(data["cuped"]["standard_error"]/data["unadjusted"]["standard_error"])**2,
              "effect-estimator variance reduction")
        for key in ("planning_raw", "planning_cuped"):
            plan = data[key]
            if plan["sd_source"] != "pre-treatment calibration only" or plan["n_per_arm"] != 50 or not 0 <= plan["power"] <= 1:
                raise ArtifactError("Invalid recorded planning scope")
            if plan["mde"] <= 0 or plan["sd"] <= 0 or plan["target_effect"] != design["practical_threshold"]:
                raise ArtifactError("Invalid recorded planning values")
        return data
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError("Malformed saved experiment") from error


def load_experiment(directory=DEFAULT_EXPERIMENT):
    try:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 128*1024:
            raise ArtifactError("Invalid experiment manifest")
        manifest = read_json(manifest_path)
        if manifest["format_version"] != FORMAT or set(manifest["outputs"]) != {"experiment.json"}:
            raise ArtifactError("Unknown experiment bundle/version")
        sources = manifest["sources"]
        if set(sources) != {"docs/experimentlab_report.html",
                            "outputs/experimentlab/release-20261030-final/experiment.json",
                            "outputs/experimentlab/release-20261030-final/checksums.json"}:
            raise ArtifactError("Missing canonical source identities")
        if sources["outputs/experimentlab/release-20261030-final/experiment.json"] != manifest["outputs"]["experiment.json"]:
            raise ArtifactError("Copied artifact differs from canonical source identity")
        for source in sources.values():
            if (not isinstance(source["sha256"], str) or len(source["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in source["sha256"])
                or type(source["size_bytes"]) is not int or source["size_bytes"] <= 0):
                raise ArtifactError("Malformed source identity")
        path = directory / "experiment.json"
        record = manifest["outputs"][path.name]
        if path.is_symlink() or not 0 < path.stat().st_size <= 128*1024:
            raise ArtifactError("Unsafe or oversized experiment artifact")
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ArtifactError("Experiment artifact checksum differs")
        return {"manifest": manifest, "experiment": validate_experiment(read_json(path))}
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError("Saved synthetic experiment is missing or malformed") from error
