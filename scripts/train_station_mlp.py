"""Station MLP development only; no test evaluation or legacy artifact loading."""
import argparse
import json
from pathlib import Path

from citypulse_ai.forecasting.station_mlp import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tft-reference', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=9000)
    args = parser.parse_args()
    report = train(args.dataset, args.output_dir, args.tft_reference, max_seconds=args.max_seconds,
                   progress=lambda stage: print(stage, flush=True))
    print(json.dumps({k: report[k] for k in ['status', 'epochs_completed', 'best', 'parameters',
                                           'stop_reason', 'validation', 'test_evaluated']}, indent=2))
