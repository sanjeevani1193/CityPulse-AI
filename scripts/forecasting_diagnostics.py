"""Post-hoc analysis of pinned saved forecasts only; no inference or tuning."""
import argparse
import json
from pathlib import Path
from citypulse_ai.evaluation.forecasting_diagnostics import analyze

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, default=Path('outputs/forecasting-v1/saved-test-predictions.parquet'))
    p.add_argument('--preprocessing', type=Path, default=Path('outputs/forecasting-v1/artifacts/preprocessing.json'))
    p.add_argument('--output', type=Path, default=Path('outputs/forecasting-v1/diagnostics'))
    p.add_argument('--figures', type=Path, default=Path('docs/assets/forecasting-v1'))
    a = p.parse_args(); summary, _ = analyze(a.predictions, a.preprocessing, a.output, a.figures)
    print(json.dumps(summary, indent=2))
