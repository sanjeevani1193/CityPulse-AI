"""Command-line entry point for a local Citi Bike CSV."""

import argparse
import json

from citypulse_ai.etl.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and aggregate Citi Bike trips")
    parser.add_argument("--input", required=True, help="Local CSV path")
    parser.add_argument("--output-dir", required=True, help="Directory for the four run outputs")
    args = parser.parse_args()
    try:
        report = run_pipeline(args.input, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"ETL failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
