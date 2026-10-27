"""Synthetic frozen MLP inference; run in the PyTorch environment."""
import json

import pandas as pd
import pytest
import torch

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.evaluation.station_followup import mlp_phase, keys_and_labels
from citypulse_ai.forecasting.mlp import DemandMLP
from tests.test_final_station_dataset import saved
from tests.test_station_followup import phase_plan


def test_saved_mlp_inference_only_and_once(saved, tmp_path, monkeypatch):
    root, _, _, _, _, *_ = saved
    prep = tmp_path/'mlp-preprocessing.json'
    prep.write_text(json.dumps({'station_codes': {'exact.00_': 1, '001': 2}, 'numeric_mean': [0.]*47,
        'numeric_scale': [1.]*47, 'target_mean': 0., 'target_scale': 1.}))
    model = DemandMLP(96); checkpoint = tmp_path/'synthetic.pt'
    torch.save({'epoch': 10, 'state_dict': model.state_dict(), 'preprocessing_sha256': fingerprint(prep)}, checkpoint)
    report = tmp_path/'report.json'; report.write_text('{}')
    artifacts = {'mlp_checkpoint': checkpoint, 'mlp_preprocessing': prep, 'mlp_report': report}
    out = tmp_path/'evaluation'; contract = phase_plan(root, artifacts, out)
    from citypulse_ai.forecasting import station_mlp
    def forbidden(*a, **kw): raise AssertionError('Training called during test evaluation')
    monkeypatch.setattr(station_mlp, 'train', forbidden)
    original = fingerprint(checkpoint)
    mlp_phase(root, artifacts, out)
    result = pd.read_parquet(out/'mlp-predictions.parquet')
    assert keys_and_labels(result, contract) == contract['paired_key_sha256']
    assert len(result) == 480 and (result.mlp_clipped >= 0).all()
    assert fingerprint(checkpoint) == original
    with pytest.raises(FileExistsError): mlp_phase(root, artifacts, out)
