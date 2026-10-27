"""Explicitly authorized follow-up only; no training or cloud lifecycle actions."""
import argparse
import json
from pathlib import Path

from citypulse_ai.evaluation.station_followup import freeze, mlp_phase, lstm_phase, summarize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--artifacts-json', type=Path, required=True,
                        help='Trusted operator mapping of frozen artifact names to local paths')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['freeze', 'mlp', 'lstm', 'summarize'], required=True)
    args = parser.parse_args()
    artifacts = json.loads(args.artifacts_json.read_text())
    if args.phase == 'freeze':
        result = freeze(args.dataset, artifacts, args.output_dir)
    elif args.phase == 'summarize':
        result = summarize(artifacts, args.output_dir)
    else:
        function = mlp_phase if args.phase == 'mlp' else lstm_phase
        result = function(args.dataset, artifacts, args.output_dir,
                          progress=lambda stage: print(stage, flush=True))
    print(json.dumps(result, indent=2))
