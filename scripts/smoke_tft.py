"""Three CPU optimizer steps, two existing stations; architecture diagnostics only."""

import argparse
import io
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

import pandas as pd
import torch

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.station_dataset import NYC, build_station_panel, make_window
from citypulse_ai.features.station_history import StationHistory
from citypulse_ai.features.station_labels import POLICY, review_positive_labels
from citypulse_ai.forecasting.tft import TFTConfig, TemporalFusionTransformer, inverse_quantiles, training_step
from citypulse_ai.forecasting.tft_preparation import (
    fit_training_scalers, load_preprocessing, save_preprocessing, windows_to_batch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-review', type=Path, required=True)
    parser.add_argument('--cohort-report', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/historical/manifest.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a new diagnostic directory; existing artifacts are preserved')
    started, free = time.perf_counter(), shutil.disk_usage('.').free
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    review = json.loads(args.label_review.read_text())
    cohort = json.loads(args.cohort_report.read_text())
    pinned_reports = {str(p): fingerprint(p) for p in (args.label_review, args.cohort_report)}
    code_paths = [Path(__file__), Path('src/citypulse_ai/forecasting/tft.py'),
                  Path('src/citypulse_ai/forecasting/tft_preparation.py')]
    code_hashes = {str(p): fingerprint(p) for p in code_paths}
    if (review['policy'] != POLICY or review['source_identity'] != cohort['source_revision']
            or review['frozen_cohort_report_sha256'] != pinned_reports[str(args.cohort_report)]
            or cohort['selection']['train_end_utc_exclusive'] != '2024-07-01T04:00:00+00:00'):
        raise ValueError('Development cohort or label-policy identity differs')
    codes = review['station_codes']
    if codes != dict(list(cohort['selected_codes'].items())[:2]):
        raise ValueError('Require the existing two-station development subset')
    audit = Path(review['evidence_reviews'][0]['audit_path'])
    history = StationHistory(audit, args.manifest)
    if (history.identity != review['source_identity'] or history.audit_sha256 != review['source_audit_sha256']
            or history.manifest_sha256 != review['manifest_sha256']):
        raise ValueError('Historical evidence changed')
    existing = {e['start_utc']: e['identity'] for e in review['evidence_reviews']}
    provenance = []

    def panel_for(date):
        origin = pd.Timestamp(date, tz=NYC).tz_convert('UTC')
        start, end = origin-pd.Timedelta(hours=336), origin+pd.Timedelta(hours=24)
        result = review_positive_labels(history, codes, start=start, end=end)
        if result.evidence['identity'] != existing.get(start.isoformat()):
            raise ValueError('Re-adjudicated label evidence differs from prior review')
        panel = build_station_panel(result.observed, codes, start=start, end=end, quality=result.quality)
        assert panel.loc[panel.label_observed_mask, 'ride_count'].gt(0).all()
        assert not panel.availability_known_mask.any()
        provenance.append({'origin_utc': origin.isoformat(), 'label_review_identity': result.evidence['identity'],
            'trusted_hours': int(panel.label_observed_mask.sum()), 'hours': len(panel),
            'mask_reasons': result.decisions.loc[~result.decisions.count_complete, 'reason'].value_counts().to_dict()})
        return panel, origin, end

    # Fixed dates/steps/seed; July cannot select parameters or influence scalers.
    train, origin, end = panel_for('2024-06-01')
    scaler_start = train.hour_start_utc.min()
    scalers = fit_training_scalers(train, train_start=scaler_start, train_end=end)
    split_start = pd.Timestamp('2024-01-01', tz=NYC).tz_convert('UTC')
    split_end = pd.Timestamp('2024-07-01', tz=NYC).tz_convert('UTC')
    train_windows = [make_window(train, s, origin, split_start=split_start, split_end=split_end,
                                 scalers=scalers, split_role='training') for s in codes]
    batch = windows_to_batch(train_windows, scalers)
    if not batch.target_mask.any() or not all(w.input_eligible for w in train_windows):
        raise ValueError('Sparse real data cannot support this diagnostic training step')
    config = TFTConfig(stations=100, hidden=16, heads=2, dropout=.1)
    model = TemporalFusionTransformer(config).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    fit_started = time.perf_counter()
    steps = []
    for _ in range(3):
        result = training_step(model, batch, optimizer)
        if result is None:
            raise ValueError('No trusted training targets; no labels are manufactured')
        steps.append(result)
    training_seconds = time.perf_counter()-fit_started
    assert any(not torch.equal(p, before[n]) for n,p in model.named_parameters())
    # July is inference only, loaded after all optimizer steps.
    holdout, july, july_end = panel_for('2024-07-01')
    holdout_windows = [make_window(holdout, s, july, split_start=july, split_end=july_end,
                                  scalers=scalers) for s in codes]
    evaluation = windows_to_batch(holdout_windows, scalers)
    model.eval()
    with torch.inference_mode():
        prediction = model(*evaluation.inputs())
        assert torch.equal(prediction, model(*evaluation.inputs()))
        counts = inverse_quantiles(prediction, evaluation.means, evaluation.scales)
    assert torch.isfinite(counts).all()
    assert (counts[...,0] <= counts[...,1]).all() and (counts[...,1] <= counts[...,2]).all()
    buffer = io.BytesIO()
    torch.save({'format_version': 1, 'config': model.artifact_config(), 'state_dict': model.state_dict()}, buffer)
    if len(buffer.getvalue()) > 1024**2:
        raise ValueError('One-MiB checkpoint budget exceeded')
    buffer.seek(0)
    saved = torch.load(buffer, map_location='cpu', weights_only=True)
    restored = TemporalFusionTransformer(TFTConfig(**saved['config'])).cpu()
    restored.load_state_dict(saved['state_dict']); restored.eval()
    with torch.inference_mode():
        assert torch.equal(prediction, restored(*evaluation.inputs()))
    history._unchanged()
    if any(fingerprint(Path(p)) != h for p,h in {**pinned_reports, **code_hashes}.items()):
        raise ValueError('Code or report changed during smoke test')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    preprocessing = args.output_dir/'preprocessing.json'
    save_preprocessing(preprocessing, scalers, codes)
    loaded, loaded_codes = load_preprocessing(preprocessing)
    assert loaded == scalers and loaded_codes == codes
    checkpoint = args.output_dir/'diagnostic_state.pt'
    with checkpoint.open('xb') as stream:
        stream.write(buffer.getvalue())
    # Check the actual file as well as the in-memory serialized state.
    disk = torch.load(checkpoint, map_location='cpu', weights_only=True)
    restored.load_state_dict(disk['state_dict'])
    with torch.inference_mode():
        assert torch.equal(prediction, restored(*evaluation.inputs()))
    report = {'scope': 'architecture_smoke_only_not_accuracy_or_final_training', 'seed': 42,
        'cpu_threads': 1, 'config': model.artifact_config(), 'parameters': sum(p.numel() for p in model.parameters()),
        'python': sys.version, 'torch': torch.__version__, 'pandas': pd.__version__,
        'source_identity': history.identity, 'manifest_sha256': history.manifest_sha256,
        'source_audit_sha256': history.audit_sha256, 'input_report_sha256': pinned_reports,
        'code_sha256': code_hashes, 'label_policy': POLICY, 'provenance': provenance,
        'training_origin_utc': origin.isoformat(), 'training_windows': len(train_windows),
        'training_target_count': int(batch.target_mask.sum()), 'training_target_grid': batch.target_mask.numel(),
        'unique_scaler_training_hours': {s: v.sample_count for s,v in scalers.items()},
        'scaler_start_utc': scaler_start.isoformat(), 'scaler_end_utc_exclusive': end.isoformat(),
        'optimizer': 'Adam lr=0.001; gradient norm clipping=1', 'optimizer_steps': steps,
        'training_seconds': training_seconds, 'july_used_for_training_or_preprocessing': False,
        'inference_origin_utc': july.isoformat(), 'inference_target_mask_count': int(evaluation.target_mask.sum()),
        'forecast_shape': list(counts.shape), 'forecasts_original_units': counts.tolist(),
        'repeated_inference_exact': True, 'serialization_reload_exact': True, 'quantile_crossings': 0,
        'model_training_kind': 'exactly_three_CPU_steps_on_two_fixed_windows',
        'zero_labels_certified': 0, 'all_hour_benchmark_eligible': False,
        'artifact_sha256': {p.name: fingerprint(p) for p in (preprocessing, checkpoint)},
        'artifact_bytes': {p.name: p.stat().st_size for p in (preprocessing, checkpoint)},
        'runtime_seconds_before_report': time.perf_counter()-started,
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*
            (1 if platform.system()=='Darwin' else 1024),
        'host_free_before_bytes': free, 'host_free_after_bytes': shutil.disk_usage('.').free}
    payload = (json.dumps(report, indent=2, allow_nan=False)+'\n').encode()
    if len(payload)+sum(report['artifact_bytes'].values()) > 2*1024**2:
        raise ValueError('Two-MiB diagnostic output budget exceeded')
    with (args.output_dir/'report.json').open('xb') as stream:
        stream.write(payload)
    print(json.dumps({k: report[k] for k in ['parameters','training_target_count','unique_scaler_training_hours',
        'optimizer_steps','training_seconds','forecast_shape','serialization_reload_exact',
        'runtime_seconds_before_report','process_peak_rss_bytes','artifact_bytes']}, indent=2))


if __name__ == '__main__':
    main()
