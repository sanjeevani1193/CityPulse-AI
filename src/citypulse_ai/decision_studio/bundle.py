"""Reproducible local-only export; never performs inference or accesses GCP."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import tempfile
from zoneinfo import ZoneInfo

from .artifacts import (ArtifactError, DEFAULT_BUNDLE, FORMAT, FORECAST_COLUMNS,
                        MODELS, read_json, sha256, validate_sequences)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path("outputs/forecasting-v1/saved-test-predictions.parquet")
CONTRACT = Path("src/citypulse_ai/api/station_artifacts.json")
PREPROCESSING = Path("outputs/forecasting-v1/artifacts/preprocessing.json")
PROOF = Path("outputs/gcp-pilots/final-december-january-batch-20261023/january-2026/pilot-result.json")
DIAGNOSTICS = Path("outputs/forecasting-v1/diagnostics-final")
# Rank positions within the frozen training-mean order, never test-error ranking.
RANKS = (0, 16, 33, 34, 50, 66, 67, 83, 99)
DATES = ("2025-10-01", "2025-11-02", "2025-12-01")
TABLES = ("overall", "station_id", "hour", "day_type", "month", "activity", "period", "station_distribution")
TEXT_COLUMNS = {"model", "station_id", "day_type", "month", "activity", "period"}
INTEGER_COLUMNS = {"targets", "stations", "hour"}


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_bundle(root=ROOT, destination=DEFAULT_BUNDLE):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    root, destination = Path(root), Path(destination)
    sources = {}

    def record(relative):
        path = root / relative
        sources[str(relative)] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        return path

    spec = read_json(record(CONTRACT))
    source = record(SOURCE)
    if sha256(source) != spec["diagnostics"]["sha256"]:
        raise ArtifactError("Saved prediction checksum differs from frozen contract")
    prep_path = record(PREPROCESSING)
    if sha256(prep_path) != spec["artifacts"]["preprocessing.json"]["sha256"]:
        raise ArtifactError("Training-only preprocessing identity differs")
    prep = read_json(prep_path)
    order = sorted(prep["station_codes"], key=lambda s: (prep["scalers"][s]["mean"], s))
    if len(order) != 100:
        raise ArtifactError("Require exactly 100 frozen stations")
    stations = [order[i] for i in RANKS]
    origins = {datetime.fromisoformat(d).replace(tzinfo=ZoneInfo("America/New_York")) for d in DATES}
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != spec["benchmark"]["targets"] or set(parquet.schema_arrow.names) != FORECAST_COLUMNS:
        raise ArtifactError("Frozen full prediction schema/count differs")
    sources[str(SOURCE)]["rows"] = parquet.metadata.num_rows
    sources[str(SOURCE)]["schema"] = [{"name": f.name, "type": str(f.type)} for f in parquet.schema_arrow]
    selected = []
    station_values = pa.array(stations, type=pa.string())
    origin_values = pa.array(sorted(origins), type=parquet.schema_arrow.field("origin_utc").type)
    # One 8,192-row batch at a time. No models, cloud calls or all-tensor arrays.
    for batch in parquet.iter_batches(batch_size=8192):
        mask = pc.and_(pc.is_in(batch.column("station_id"), value_set=station_values),
                       pc.is_in(batch.column("origin_utc"), value_set=origin_values))
        selected.extend(batch.filter(mask).to_pylist())
    selected.sort(key=lambda r: (r["station_id"], r["origin_utc"], r["horizon_step"]))
    if validate_sequences(selected) != len(stations) * len(origins):
        raise ArtifactError("Some predeclared demo sequences are unavailable")
    diagnostics = {}
    for name in TABLES:
        with record(DIAGNOSTICS / (name + ".csv")).open() as stream:
            diagnostics[name] = [{key: value if key in TEXT_COLUMNS else int(value) if key in INTEGER_COLUMNS else float(value)
                                  for key, value in row.items()} for row in csv.DictReader(stream)]
    diagnostics_proof = read_json(record(DIAGNOSTICS / "provenance.json"))
    if (diagnostics_proof["source_sha256"] != spec["diagnostics"]["sha256"]
            or diagnostics_proof["paired_key_sha256"] != spec["benchmark"]["paired_key_sha256"]):
        raise ArtifactError("Diagnostics source/key identity differs")
    for row in diagnostics["overall"]:
        key = next(k for k, v in MODELS.items() if v == row["model"])
        frozen = spec["benchmark"]["metrics"][key]
        if any(abs(row[m] - frozen[m]) > 1e-12 for m in ("MAE", "RMSE", "WAPE_percent", "targets")):
            raise ArtifactError("Saved diagnostics differ from frozen full-test metrics")
    proof = read_json(record(PROOF))["final_24_month_proof"]
    platform = {key: proof[key] for key in ("final_window_ride_starts", "final_window_unique_station_hours",
                                          "source_archives", "primary_archives", "duplicate_ride_ids", "duplicate_station_hour_keys")}
    benchmark = {"format_version": FORMAT, "target_policy": spec["target_policy"],
                 "benchmark": spec["benchmark"], "platform": platform,
                 "dataset_manifest_sha256": spec["dataset_manifest_sha256"],
                 "station_ids": sorted(prep["station_codes"]),
                 "periods": {"train": ["2024-01-01", "2025-07-01"],
                             "validation": ["2025-07-01", "2025-10-01"], "test": ["2025-10-01", "2026-01-01"]},
                 "encoder_hours": 336, "forecast_hours": 24}
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as error:
        raise ArtifactError("Record the source Git revision before export") from error
    with tempfile.TemporaryDirectory(prefix="citypulse-offline-bundle-") as scratch:
        temporary = Path(scratch)
        (temporary / "benchmark.json").write_bytes(_json_bytes(benchmark))
        (temporary / "diagnostics.json").write_bytes(_json_bytes(diagnostics))
        pq.write_table(pa.Table.from_pylist(selected, schema=parquet.schema_arrow.remove_metadata()),
                       temporary / "forecast_sequences.parquet", compression="zstd", version="2.6")
        manifest = {"format_version": FORMAT, "source_git_revision": revision,
                    "exporter_sha256": sha256(__file__), "pyarrow_version": pa.__version__,
                    "evaluation_scope": "frozen_full_test", "sources": sources,
                    "subset": {"is_full_evaluation": False, "rows": len(selected), "windows": len(stations)*len(origins),
                               "station_ids": sorted(stations), "local_origin_dates": list(DATES),
                               "selection_rule": "Training-mean ascending station ranks (zero-based) " + str(list(RANKS)) +
                                                 "; one fixed origin per test month, including November DST fall-back; no error-based selection",
                               "horizon_rule": "steps 1..24; target UTC = origin UTC + (step-1) elapsed hours"},
                    "transformation": "Exact selected source rows/types, canonical row order; strip pandas metadata; no rounding, inference, refitting or new evaluation",
                    "outputs": {p.name: {"size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(temporary.iterdir())}}
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        from .artifacts import load_bundle
        load_bundle(temporary)
        # Idempotent on identical bytes; changed exports need a new destination.
        if destination.exists():
            if {p.name for p in destination.iterdir()} != {p.name for p in temporary.iterdir()} or any(
                    sha256(destination / p.name) != sha256(p) for p in temporary.iterdir()):
                raise ArtifactError("Destination exists with different bytes; choose a new directory")
        else:
            destination.mkdir(parents=True)
            for p in temporary.iterdir():
                with (destination / p.name).open("xb") as stream:
                    stream.write(p.read_bytes())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    result = build_bundle(destination=args.output)
    print(json.dumps({"rows": result["subset"]["rows"], "windows": result["subset"]["windows"],
                      "outputs": result["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
