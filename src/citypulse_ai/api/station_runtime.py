"""Trusted frozen artifacts and lazy CPU adapters; TensorFlow stays in its worker."""
import hashlib
import json
import os
from pathlib import Path
from threading import RLock

import numpy as np

from citypulse_ai.api.station_contract import MODELS, context, response, tabular
from citypulse_ai.api.station_catalog import catalog
from citypulse_ai.features.final_station_dataset import load_preprocessing
from citypulse_ai.forecasting.station_xgboost import FEATURES, FEATURE_TYPES


class ArtifactStore:
    def __init__(self, root=None, specification=None):
        self.root = Path(root or os.environ.get('CITYPULSE_STATION_ARTIFACT_DIR', 'outputs/forecasting-v1/artifacts'))
        self.specification = specification or catalog()
        self.scalers, self.codes = load_preprocessing(self.file('preprocessing.json'))
        if len(self.codes) != 100 or set(self.codes.values()) != set(range(1, 101)):
            raise ValueError('Frozen station vocabulary differs')

    def file(self, name):
        expected = self.specification['artifacts'][name]; path = self.root/name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != expected['size_bytes']:
            raise ValueError('Trusted station artifact missing or invalid: '+name)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''): digest.update(block)
        if digest.hexdigest() != expected['sha256']:
            raise ValueError('Trusted station artifact checksum mismatch: '+name)
        return path

    def version(self, model):
        names = {'xgboost': 'xgboost/model.ubj', 'pytorch-mlp': 'mlp/epoch-010.pt',
                 'pytorch-tft': 'tft/epoch-006.pt', 'tensorflow-lstm': 'lstm/epoch-018.keras'}
        return ('elapsed-'+model+'-v1') if model not in names else names[model].split('/')[-1]+':'+self.specification['artifacts'][names[model]]['sha256'][:12]


class ModelUnavailable(RuntimeError):
    pass


class StationRuntime:
    def __init__(self, root=None):
        self.store = ArtifactStore(root); self.models = {}; self.lock = RLock(); self.errors = {}

    def load(self, model):
        with self.lock:
            if model in self.models: return self.models[model]
            try:
                store = self.store
                if model == 'xgboost':
                    import xgboost as xgb
                    booster = xgb.Booster(); booster.load_model(store.file('xgboost/model.ubj'))
                    prep = json.loads(store.file('xgboost/preprocessing.json').read_text())
                    if (booster.feature_names != list(FEATURES) or booster.feature_types != list(FEATURE_TYPES)
                            or booster.num_boosted_rounds() != 300 or prep['station_codes'] != store.codes
                            or prep['feature_names'] != list(FEATURES)):
                        raise ValueError('Frozen XGBoost schema differs')
                    booster.set_param({'device': 'cpu', 'nthread': 1}); value = booster
                elif model in ('pytorch-mlp', 'pytorch-tft'):
                    import torch
                    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
                    if model == 'pytorch-mlp':
                        from citypulse_ai.forecasting.station_mlp import restore
                        cp, network, prep = restore(store.file('mlp/epoch-010.pt'), store.file('mlp/preprocessing.json'))
                        if cp['epoch'] != 10 or prep['station_codes'] != store.codes or prep['feature_names'] != list(FEATURES):
                            raise ValueError('Frozen MLP contract differs')
                        value = network, prep
                    else:
                        from citypulse_ai.forecasting.tft import TemporalFusionTransformer, TFTConfig
                        cp = torch.load(store.file('tft/epoch-006.pt'), map_location='cpu', weights_only=True)
                        if cp['epoch'] != 6 or cp['dataset_manifest_sha256'] != store.specification['dataset_manifest_sha256']:
                            raise ValueError('Frozen TFT checkpoint differs')
                        value = TemporalFusionTransformer(TFTConfig(**cp['model_config'])).cpu()
                        value.load_state_dict(cp['state_dict']); value.eval()
                else:
                    raise ValueError('Unsupported local model')
                self.errors.pop(model, None)
                self.models[model] = value
                return value
            except Exception as error:
                self.errors[model] = 'Trusted model runtime unavailable'
                raise ModelUnavailable(self.errors[model]) from error

    def listing(self):
        names = {'xgboost': ['xgboost/model.ubj', 'xgboost/preprocessing.json'],
            'pytorch-mlp': ['mlp/epoch-010.pt', 'mlp/preprocessing.json'], 'pytorch-tft': ['tft/epoch-006.pt']}
        values = []
        for model in MODELS:
            available = model != 'tensorflow-lstm' and model not in self.errors
            try:
                for name in names.get(model, []): self.store.file(name)
            except ValueError:
                available = False
            loaded = model in self.models or model.startswith('lag-')
            values.append({'id': model, 'model_version': self.store.version(model), 'available': available,
                           'loaded': loaded, 'ready': available and loaded,
                           'state': 'ready' if available and loaded else 'verified-not-loaded' if available else 'unavailable',
                           'quantiles': model == 'pytorch-tft'})
        return values

    def predict(self, request):
        with self.lock:
            frame, window = context(request, self.store.codes, self.store.scalers)
            model = request.model; quantiles = None
            if model.startswith('lag-'):
                lag = int(model.split('-')[1]); rows = frame.iloc[336-lag:360-lag]
                if not rows.label_observed_mask.all():
                    raise ValueError('The selected baseline needs all 24 trusted lag observations')
                prediction = rows.ride_count.to_numpy(dtype=float)
            elif model == 'xgboost':
                prediction = self.load(model).inplace_predict(tabular(frame, self.store.codes[request.station_id]))
            elif model == 'pytorch-mlp':
                network, prep = self.load(model)
                from citypulse_ai.forecasting.station_mlp import predict
                prediction = predict(network, tabular(frame, self.store.codes[request.station_id]), prep)
            elif model == 'pytorch-tft':
                import torch
                from citypulse_ai.forecasting.tft_preparation import windows_to_batch
                from citypulse_ai.forecasting.tft import inverse_quantiles
                batch = windows_to_batch([window], self.store.scalers)
                with torch.inference_mode():
                    raw = inverse_quantiles(self.load(model)(*batch.inputs()), batch.means, batch.scales).numpy()[0]
                quantiles = {name: raw[:, i] for i, name in enumerate(('q10', 'q50', 'q90'))}
                prediction = raw[:, 1]
            else:
                raise ValueError('TensorFlow requests require the isolated worker')
            return response(request, prediction, self.store.version(model), quantiles)
