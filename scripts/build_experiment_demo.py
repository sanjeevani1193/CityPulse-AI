"""Package existing synthetic evidence unchanged; no simulation or inference."""
import argparse
import json
from pathlib import Path

from citypulse_ai.decision_studio.artifacts import ArtifactError, read_json, sha256
from citypulse_ai.decision_studio.experiment_data import DEFAULT_EXPERIMENT, FORMAT, validate_experiment

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("outputs/experimentlab/release-20261030-final")


def build(destination=DEFAULT_EXPERIMENT):
    destination = Path(destination)
    if destination.exists():
        raise ArtifactError("Destination already exists; choose a new directory")
    checks = read_json(ROOT / SOURCE / "checksums.json")
    for name in ("experiment.json", "report.html"):
        if sha256(ROOT / SOURCE / name) != checks[name]:
            raise ArtifactError("Canonical ExperimentLab checksum differs")
    report = ROOT / "docs/experimentlab_report.html"
    if sha256(report) != checks["report.html"]:
        raise ArtifactError("Committed HTML does not match canonical report")
    source = ROOT / SOURCE / "experiment.json"
    data = validate_experiment(read_json(source))
    paths = [SOURCE / "experiment.json", SOURCE / "checksums.json", Path("docs/experimentlab_report.html")]
    manifest = {"format_version": FORMAT,
                "sources": {str(p): {"sha256": sha256(ROOT/p), "size_bytes": (ROOT/p).stat().st_size} for p in paths},
                "transformation": "Byte-identical copy of the existing fully synthetic experiment.json. No simulation rerun, coefficient refit or historical observations.",
                "verification": "Canonical checksum manifest and committed HTML match; saved unit records independently reproduce effect/SE/df algebra. P-values/CIs and planning are preserved, not recalculated at app runtime.",
                "limitations": "Earlier calibration-period unit records and original implementation hashes are absent from this JSON. Frozen theta/SD/fit scope and saved runtime/seeds are preserved; their original fit is not reconstructed.",
                "synthetic_stations": len(data["station_records"]),
                "outputs": {"experiment.json": {"sha256": sha256(source), "size_bytes": source.stat().st_size}}}
    destination.mkdir(parents=True)
    with (destination / "experiment.json").open("xb") as stream:
        stream.write(source.read_bytes())
    with (destination / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True); stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPERIMENT)
    print(json.dumps(build(parser.parse_args().output), indent=2))
