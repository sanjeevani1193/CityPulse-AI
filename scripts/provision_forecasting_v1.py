"""Read-only GCS provisioning with existing ADC; never start compute or fit models."""
import argparse
import json
from pathlib import Path
from citypulse_ai.cloud.forecasting_release import provision

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts-dir', type=Path, default=Path('outputs/forecasting-v1/artifacts'))
    p.add_argument('--demo', action='store_true')
    p.add_argument('--diagnostics', action='store_true')
    a = p.parse_args()
    print(json.dumps(provision(a.artifacts_dir, demo=a.demo, diagnostics=a.diagnostics), indent=2))
