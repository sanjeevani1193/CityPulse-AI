"""Pooled station LSTM development. No test targets and no legacy weights."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from citypulse_ai.etl.archives import fingerprint
from citypulse_ai.features.final_station_dataset import LazyStationWindows
from citypulse_ai.features.station_dataset import KNOWN_FEATURES, utc
from citypulse_ai.forecasting.station_xgboost import paired_metrics, write_json


@dataclass(frozen=True)
class Config:
    stations: int = 100
    embedding: int = 8
    units: int = 32
    head_units: int = 16
    batch_size: int = 64
    learning_rate: float = .001
    max_epochs: int = 20
    patience: int = 4
    min_delta: float = .0001
    seed: int = 42


def configure_cpu(seed=42):
    """Same deterministic CPU policy as the original LSTM, without importing it."""
    tf.config.set_visible_devices([], 'GPU')
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass  # A caller may already have initialized this isolated CPU runtime.
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_model(config=Config()):
    configure_cpu(config.seed)
    history = keras.Input((336, 16), name='history')
    station = keras.Input((1,), dtype='int32', name='station')
    future = keras.Input((24, 11), name='calendar')
    embedding = keras.layers.Flatten()(keras.layers.Embedding(config.stations+1, config.embedding)(station))
    context = keras.layers.LSTM(config.units)(keras.layers.Concatenate()([
        history, keras.layers.RepeatVector(336)(embedding)]))
    decoder = keras.layers.Concatenate()([
        keras.layers.RepeatVector(24)(context), keras.layers.RepeatVector(24)(embedding), future])
    decoder = keras.layers.Dense(config.head_units, activation='relu')(decoder)
    output = keras.layers.Reshape((24,))(keras.layers.Dense(1)(decoder))
    model = keras.Model({'history': history, 'station': station, 'calendar': future}, output)
    model.compile(optimizer=keras.optimizers.Adam(config.learning_rate, global_clipnorm=1.), loss='mse')
    return model


class DevelopmentData(LazyStationWindows):
    """Train/validation only, one station frame and one sequence minibatch in RAM."""
    def __init__(self, root, split):
        if split not in ('train', 'validation'):
            raise ValueError('Test split is not authorized')
        super().__init__(root, split)
        self.index = self.index.sort_values(['station_id', 'origin_utc']).reset_index(drop=True)
        right = utc(self.manifest['splits'][split][1])
        self.paths = [p for p in self.paths if self.panel_start(p.name) < right]
        a, b = map(utc, self.manifest['splits']['train'])
        if any(s.train_start_utc != a or s.train_end_utc != b for s in self.scalers.values()):
            raise ValueError('Frozen scaler training period differs')
        if self.index.empty or not self.index.training_eligible.all():
            raise ValueError('Unusable frozen windows')

    @staticmethod
    def panel_start(name):
        return pd.Timestamp(name.removeprefix('panel-').removesuffix('.parquet')+'-01',
                            tz='America/New_York').tz_convert('UTC')

    def verify(self):
        right = utc(self.manifest['splits']['validation'][1])
        names = {'preprocessing.json', 'windows.parquet'} | {
            r['name'] for r in self.manifest['panels'] if self.panel_start(r['name']) < right}
        receipts = {r['name']: r for r in self.manifest['artifacts']}
        for name in names:
            p = self.root/name; r = receipts[name]
            if (p.name != name or p.is_symlink() or p.stat().st_size != r['size_bytes']
                    or fingerprint(p) != r['sha256']):
                raise ValueError('Development artifact changed: '+name)

    def batches(self, batch_size, *, epoch=None):
        groups = list(self.index.groupby('station_id', sort=True))
        rng = np.random.default_rng(42 if epoch is None else 42+epoch)
        order = np.arange(len(groups))
        if epoch is not None:
            rng.shuffle(order)
        left, right = map(utc, self.manifest['splits'][self.index.split.iloc[0]])
        for gi in order:
            station, group = groups[gi]
            begin = utc(group.origin_utc.min())-pd.Timedelta(hours=336)
            end = utc(group.origin_utc.max())+pd.Timedelta(hours=24)
            if begin < utc(self.manifest['splits']['train'][0]) or end > right:
                raise ValueError('Window crosses frozen boundaries')
            frame = pd.read_parquet(self.paths, filters=[('station_id', '==', station),
                ('hour_start_utc', '>=', begin), ('hour_start_utc', '<', end)]).sort_values('hour_start_utc')
            hours = pd.DatetimeIndex(frame.hour_start_utc)
            if not hours.equals(pd.date_range(begin, end, freq='h', inclusive='left')):
                raise ValueError('Incomplete UTC station grid')
            if not frame.station_id.eq(station).all() or not frame.station_code.eq(self.codes[station]).all():
                raise ValueError('Station identity differs')
            counts = frame.ride_count.to_numpy(dtype=np.float64, na_value=np.nan)
            mask = frame.label_observed_mask.to_numpy(dtype=bool)
            if not np.isfinite(counts[mask]).all() or (counts[mask] < 0).any():
                raise ValueError('Invalid trusted counts')
            known = frame[list(KNOWN_FEATURES[:-1])].to_numpy(dtype=np.float32)
            if not np.isfinite(known).all():
                raise ValueError('Invalid known calendar')
            statuses = frame.station_available.to_numpy(dtype=np.float32, na_value=np.nan)
            status_base = frame.availability_known_mask.to_numpy(dtype=bool)
            published = pd.to_datetime(frame.status_available_at_utc, utc=True).array.asi8
            scaler = self.scalers[station]
            records = group.to_records(index=False)
            positions = np.arange(len(records))
            if epoch is not None:
                rng.shuffle(positions)
            for start in range(0, len(records), batch_size):
                rows = group.iloc[positions[start:start+batch_size]]
                origins = pd.DatetimeIndex(rows.origin_utc)
                if (origins.min() < left or (origins+pd.Timedelta(hours=24)).max() > right
                        or not (origins.tz_convert('America/New_York').hour == 0).all()):
                    raise ValueError('Invalid issuance or split boundary')
                p = ((origins-begin)/pd.Timedelta(hours=1)).to_numpy(dtype=int)
                past = p[:, None]+np.arange(-336, 0); target = p[:, None]+np.arange(24)
                observed = mask[past]
                values = np.where(observed, (counts[past]-scaler.mean)/scaler.scale, 0)
                previous = np.maximum.accumulate(np.where(observed, np.arange(336), -336), axis=1)
                ages = np.minimum(336, np.arange(336)-previous)/336
                status_mask = status_base[past] & (published[past] != pd.NaT.value) & (
                    published[past] <= origins.asi8[:, None])
                if not np.isin(statuses[past][status_mask], [0, 1]).all():
                    raise ValueError('Invalid trusted status')
                obs = np.stack([values, observed, ages, np.where(status_mask, statuses[past], 0), status_mask], axis=-1)
                past_known = np.concatenate([known[past], np.broadcast_to(
                    (np.arange(336)/359)[None, :, None], (len(p), 336, 1))], axis=-1)
                future_known = np.concatenate([known[target], np.broadcast_to(
                    (np.arange(336, 360)/359)[None, :, None], (len(p), 24, 1))], axis=-1)
                inputs = {'history': np.concatenate([past_known, obs], axis=-1).astype('float32'),
                          'calendar': future_known.astype('float32'),
                          'station': np.full((len(p), 1), self.codes[station], dtype='int32')}
                target_mask = mask[target]
                actual = np.where(target_mask, counts[target], 0)
                y = np.where(target_mask, (actual-scaler.mean)/scaler.scale, 0).astype('float32')
                yield inputs, y, target_mask.astype('float32'), {
                    'station_id': station, 'origins': origins, 'actual': actual,
                    'target_hours': hours.to_numpy()[target], 'mean': scaler.mean, 'scale': scaler.scale,
                    'lag24': counts[p[:, None]+np.arange(-24, 0)],
                    'lag168': counts[p[:, None]+np.arange(-168, -144)]}


def masked_mse(actual, prediction, mask):
    """Zero contribution from unobserved targets, including NaN placeholders."""
    mask = tf.cast(mask, tf.bool)
    residual = tf.where(mask, prediction-actual, tf.zeros_like(prediction))
    return tf.math.divide_no_nan(tf.reduce_sum(tf.square(residual)), tf.reduce_sum(tf.cast(mask, prediction.dtype)))


def optimizer_step(model):
    @tf.function(reduce_retracing=True)
    def step(inputs, y, mask):
        with tf.GradientTape() as tape:
            loss = masked_mse(y, model(inputs, training=True), mask)
        gradients = tape.gradient(loss, model.trainable_variables)
        tf.debugging.assert_all_finite(loss, 'Nonfinite loss')
        for grad in gradients:
            tf.debugging.assert_all_finite(grad, 'Nonfinite gradient')
        def update():
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        return tf.cond(tf.reduce_sum(mask) > 0, update, lambda: loss)
    return step


def evaluate(model, data, batch_size, *, destination=None):
    records = []; squared = 0.; valid = 0; digest = hashlib.sha256()
    for inputs, y, mask, meta in data.batches(batch_size):
        prediction = np.asarray(model(inputs, training=False))
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite validation forecast')
        squared += float(np.square(prediction-y)[mask.astype(bool)].sum(dtype=np.float64)); valid += int(mask.sum())
        if not mask.all():
            raise ValueError('Paired final validation requires all 24 targets')
        raw = prediction.astype(float)*meta['scale']+meta['mean']
        for i, origin in enumerate(meta['origins']):
            targets = pd.DatetimeIndex(meta['target_hours'][i])
            for lead, hour in enumerate(targets):
                digest.update(f"{meta['station_id']}|{origin.isoformat()}|{hour.isoformat()}|{lead+1}\n".encode())
            records.append(pd.DataFrame({'station_id': meta['station_id'], 'origin_utc': origin,
                'target_hour_utc': targets, 'horizon_step': np.arange(1,25), 'actual': meta['actual'][i],
                'prediction_raw': raw[i], 'prediction_clipped': np.maximum(raw[i],0),
                'lag24': meta['lag24'][i], 'lag168': meta['lag168'][i]}))
    if not valid:
        raise ValueError('No trusted validation targets')
    frame = pd.concat(records, ignore_index=True)
    if frame.duplicated(['station_id','origin_utc','target_hour_utc','horizon_step']).any():
        raise ValueError('Duplicate validation keys')
    if destination is not None:
        frame.to_parquet(destination, index=False, compression='zstd')
    return {'mse_scaled': squared/valid, 'forecast_windows': len(frame)//24, 'targets': valid,
            'paired_key_sha256': digest.hexdigest(),
            'raw': paired_metrics(frame.actual, frame.prediction_raw),
            'clipped': paired_metrics(frame.actual, frame.prediction_clipped),
            'baselines': {f'lag_{lag}': paired_metrics(frame.actual, frame[f'lag{lag}']) for lag in (24,168)}}


def verify_reference(root, data, reference):
    fixed = json.loads(Path(reference).read_text())
    if fixed['test_evaluated'] or fixed['dataset_manifest_sha256'] != fingerprint(Path(root)/'dataset.json'):
        raise ValueError('Frozen development reference differs')
    if len(data['train']) != fixed['train_windows'] or len(data['validation']) != fixed['validation']['forecast_windows']:
        raise ValueError('Frozen window counts differ')
    # Verify keys without generating a forecast or opening target/test panels.
    digest = hashlib.sha256()
    for row in data['validation'].index.itertuples(index=False):
        origin = utc(row.origin_utc)
        for lead in range(24):
            hour = origin+pd.Timedelta(hours=lead)
            digest.update(f'{row.station_id}|{origin.isoformat()}|{hour.isoformat()}|{lead+1}\n'.encode())
    if digest.hexdigest() != fixed['validation']['paired_key_sha256']:
        raise ValueError('Paired validation keys differ')
    if len(data['validation'])*24 != fixed['validation']['targets']:
        raise ValueError('Paired target count differs')
    return fixed


def probe(root, out, *, config=Config(), batches=12):
    """Training only, disposable fresh model; never a selected checkpoint."""
    configure_cpu(config.seed); out = Path(out); out.mkdir(parents=True, exist_ok=False)
    data = DevelopmentData(root, 'train'); model = build_model(config); step = optimizer_step(model)
    started = time.monotonic(); times = []; sizes = []
    for inputs, y, mask, _ in data.batches(config.batch_size, epoch=0):
        began = time.monotonic(); loss = float(step(inputs,y,mask)); times.append(time.monotonic()-began)
        sizes.append(len(y))
        if len(times) >= batches:
            break
    result = {'training_only': True, 'probe_batches': len(times), 'windows': sum(sizes), 'parameters': model.count_params(),
        'loss_finite': bool(np.isfinite(loss)), 'compile_first_batch_seconds': times[0],
        'step_seconds_after_compile': float(np.mean(times[1:])), 'elapsed_seconds': time.monotonic()-started,
        'estimated_train_epoch_seconds': float(np.mean(times[1:]))*sum(
            int(np.ceil(len(group)/config.batch_size)) for _,group in data.index.groupby('station_id')),
        'config': asdict(config), 'tensorflow': tf.__version__, 'keras': keras.__version__,
        'test_evaluated': False, 'weights_discarded_before_actual_training': True}
    write_json(out/'probe.json', result); return result


def train(root, out, reference, *, config=Config(), max_seconds=10000, publish=None,
          progress=lambda stage:None, resume=None):
    configure_cpu(config.seed)
    root, out = Path(root), Path(out)
    if resume is None:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        raise ValueError('Resume run missing')
    data = {s: DevelopmentData(root,s) for s in ('train','validation')}
    fixed = verify_reference(root,data,reference)
    if len(data['train'].codes) != config.stations:
        raise ValueError('Station count differs')
    contract = {'config': asdict(config), 'dataset_manifest_sha256': fingerprint(root/'dataset.json'),
        'preprocessing_sha256': fingerprint(root/'preprocessing.json'), 'reference_sha256': fingerprint(Path(reference)),
        'source_sha256': fingerprint(Path(__file__)), 'selection': 'minimum validation station-standardized MSE; earlier exact ties',
        'training_loss': 'trusted-target masked station-standardized MSE', 'test_evaluated': False,
        'fresh_initialization': True, 'optimizer': 'Adam global_clipnorm=1', 'splits':data['train'].manifest['splits']}
    preprocessing = out/'preprocessing.json'
    if resume is None:
        preprocessing.write_bytes((root/'preprocessing.json').read_bytes())
        write_json(out/'configuration.json',contract)
        model = build_model(config); history = []; best = None; stale = 0; patience_score = float('inf')
        initial_sha = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
    else:
        state = json.loads((out/(Path(resume).stem+'.json')).read_text())
        if (state['configuration'] != contract or fingerprint(out/Path(resume).name) != state['checkpoint_sha256']
                or fingerprint(preprocessing) != contract['preprocessing_sha256']):
            raise ValueError('Resume identity differs')
        model = keras.models.load_model(out/Path(resume).name)
        history, best, stale, patience_score = state['history'],state['best'],state['stale'],state['patience_score']
        initial_sha = state['initial_state_sha256']
    started=time.monotonic(); deadline=started+max_seconds; step=optimizer_step(model); stop='fixed_epoch_budget'
    for epoch in range(len(history)+1,config.max_epochs+1):
        estimate = history[-1]['epoch_seconds']*1.35 if history else 1800
        if time.monotonic()+estimate+900 > deadline:
            stop='runtime_before_next_epoch';break
        progress(f'train_epoch_{epoch}'); began=time.monotonic(); total=0.; valid=0; windows=0
        for inputs,y,mask,_ in data['train'].batches(config.batch_size,epoch=epoch):
            if time.monotonic()>deadline-900:
                stop='partial_epoch_discarded';break
            loss=float(step(inputs,y,mask)); n=int(mask.sum());total+=loss*n;valid+=n;windows+=len(y)
        if stop=='partial_epoch_discarded':break
        if not valid or windows!=len(data['train']):raise ValueError('Incomplete training epoch')
        progress(f'validate_epoch_{epoch}')
        scores=evaluate(model,data['validation'],config.batch_size)
        for lag in (24,168):
            for metric in ('MAE','RMSE','WAPE_percent','targets'):
                if not np.isclose(scores['baselines'][f'lag_{lag}'][metric],fixed['validation']['baselines'][f'lag_{lag}'][metric],rtol=0,atol=1e-9):
                    raise ValueError('Paired baseline differs')
        if scores['paired_key_sha256']!=fixed['validation']['paired_key_sha256']:raise ValueError('Validation keys changed')
        history.append({'epoch':epoch,'training_mse_scaled':total/valid,'training_windows':windows,
                        'training_targets':valid,'validation':scores,'epoch_seconds':time.monotonic()-began})
        score=scores['mse_scaled']; name=f'epoch-{epoch:03d}.keras'
        if best is None or score<best['validation_mse_scaled']:best={'epoch':epoch,'checkpoint':name,'validation_mse_scaled':score}
        if score<patience_score-config.min_delta:stale=0;patience_score=score
        else:stale+=1
        model.save(out/name)
        state={'configuration':contract,'history':history,'best':best,'stale':stale,'patience_score':patience_score,
               'initial_state_sha256':initial_sha,'checkpoint_sha256':fingerprint(out/name),
               'optimizer_iterations':int(model.optimizer.iterations.numpy())}
        write_json(out/(Path(name).stem+'.json'),state);write_json(out/'history.json',history)
        if publish:publish(out/name);publish(out/(Path(name).stem+'.json'))
        if stale>=config.patience:stop='validation_early_stopping';break
    if best is None:raise RuntimeError('No completed epoch')
    selected=keras.models.load_model(out/best['checkpoint'])
    scores=evaluate(selected,data['validation'],config.batch_size,destination=out/'validation-predictions.parquet')
    recorded=history[best['epoch']-1]['validation']
    if scores!=recorded:raise ValueError('Selected checkpoint validation changed on restore')
    report={**contract,'status':'station_lstm_validation_complete','best':best,'history':history,
        'epochs_completed':len(history),'training_complete':stop in ('fixed_epoch_budget','validation_early_stopping'),
        'stop_reason':stop,'parameters':selected.count_params(),'tensorflow':tf.__version__,'keras':keras.__version__,
        'training_windows':len(data['train']),'validation':scores,'elapsed_seconds':time.monotonic()-started,
        'initial_state_sha256':initial_sha,'checkpoint_sha256':fingerprint(out/best['checkpoint']),
        'optimizer_iterations':int(selected.optimizer.iterations.numpy())}
    write_json(out/'experiment.json',report);return report
