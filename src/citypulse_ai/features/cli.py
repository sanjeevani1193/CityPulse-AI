"""Run Phase 2.1 preparation without changing the Phase 1 ETL."""

import argparse
import json

from citypulse_ai.features.preparation import run_preparation


def main():
    parser = argparse.ArgumentParser(description="Prepare reviewed February hourly time series")
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confirm-complete-source", action="store_true",
                        help="Acknowledge reconciled, complete monthly source before assuming zeros")
    parser.add_argument("--station-zero-horizon", type=int,
                        help="OPT IN: assume zero within N hours after activity at training-seen stations")
    args = parser.parse_args()
    print(json.dumps(run_preparation(args.warehouse, args.output_dir,
                     source_complete=args.confirm_complete_source,
                     station_zero_horizon=args.station_zero_horizon), indent=2))


if __name__ == "__main__":
    main()
