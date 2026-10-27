"""Synthetic frozen LSTM inference; run in the isolated TensorFlow environment."""
import numpy as np
import pandas as pd
import pytest

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.evaluation.station_followup import lstm_phase, keys_and_labels
from citypulse_ai.features.station_dataset import make_window
from citypulse_ai.forecasting.station_lstm import build_model
from tests.test_final_station_dataset import saved
from tests.test_station_followup import phase_plan


def test_saved_lstm_inference_matches_canonical_window_and_once(saved, tmp_path, monkeypatch):
    root, panel, scalers, _, splits, *_ = saved
    model = build_model(); model.optimizer.build(model.trainable_variables)
    checkpoint = tmp_path/'synthetic.keras'; model.save(checkpoint)
    report = tmp_path/'report.json'; report.write_text('{}')
    artifacts = {'lstm_checkpoint': checkpoint, 'lstm_preprocessing': root/'preprocessing.json', 'lstm_report': report}
    out = tmp_path/'evaluation'; contract = phase_plan(root, artifacts, out)
    from citypulse_ai.forecasting import station_lstm
    def forbidden(*a, **kw): raise AssertionError('Training called during test evaluation')
    monkeypatch.setattr(station_lstm, 'train', forbidden)
    original = fingerprint(checkpoint)
    lstm_phase(root, artifacts, out)
    result = pd.read_parquet(out/'lstm-predictions.parquet')
    assert keys_and_labels(result, contract) == contract['paired_key_sha256']
    assert len(result) == 480 and (result.lstm_clipped >= 0).all()
    first = result.iloc[0]; origin = first.origin_utc; station = first.station_id
    w = make_window(panel, station, origin, split_start=splits['test'][0], split_end=splits['test'][1], scalers=scalers)
    x = {'history': np.concatenate([w.encoder_known, w.encoder_observed], axis=-1)[None].astype('float32'),
         'station': w.static_station_code.reshape(1, 1), 'calendar': w.decoder_known[None].astype('float32')}
    expected = np.asarray(model(x, training=False)).reshape(-1)*scalers[station].scale+scalers[station].mean
    np.testing.assert_allclose(result.lstm_raw.iloc[:24], expected, atol=1e-5, rtol=1e-5)
    assert fingerprint(checkpoint) == original and int(model.optimizer.iterations) == 0
    with pytest.raises(FileExistsError): lstm_phase(root, artifacts, out)
