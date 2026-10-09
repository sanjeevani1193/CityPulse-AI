"""Command-line entry point for a local Citi Bike CSV."""

import argparse
import json

import duckdb

from citypulse_ai.etl.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and aggregate Citi Bike trips")
    parser.add_argument("--input", required=True, nargs="+", help="One or more local CSV paths")
    parser.add_argument("--output-dir", required=True, help="Directory for the four run outputs")
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Maximum pandas batch rows")
    parser.add_argument("--max-rows-per-file", type=int, help="Validate only the first N rows per file")
    parser.add_argument("--memory-limit", default="512MB", help="DuckDB working memory limit")
    args = parser.parse_args()
    try:
        report = run_pipeline(
            args.input, args.output_dir, chunk_size=args.chunk_size,
            max_rows_per_file=args.max_rows_per_file, memory_limit=args.memory_limit,
        )
    except (ValueError, OSError, duckdb.Error) as exc:
        parser.exit(2, f"ETL failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
