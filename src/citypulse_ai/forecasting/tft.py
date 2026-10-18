"""Small native PyTorch TFT; inputs never include decoder demand or target masks."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


QUANTILES = (0.1, 0.5, 0.9)


class GateAddNorm(nn.Module):
    def __init__(self, hidden, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden, 2 * hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, value, residual):
        return self.norm(residual + F.glu(self.gate(self.dropout(value)), dim=-1))


class GatedResidualNetwork(nn.Module):
    """ELU transform, optional static context, GLU gate, residual and layer norm."""

    def __init__(self, input_size, hidden, output_size=None, context=False, dropout=0.):
        super().__init__()
        output_size = output_size or hidden
        self.first = nn.Linear(input_size, hidden)
        self.context = nn.Linear(hidden, hidden, bias=False) if context else None
        self.second = nn.Linear(hidden, output_size)
        self.skip = nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size)
        self.finish = GateAddNorm(output_size, dropout)

    def forward(self, value, context=None):
        transformed = self.first(value)
        if self.context is not None:
            if context is None:
                raise ValueError('Static context required')
            while context.ndim < transformed.ndim:
                context = context.unsqueeze(1)
            transformed = transformed + self.context(context)
        return self.finish(self.second(F.elu(transformed)), self.skip(value))


class VariableSelection(nn.Module):
    """Transform each variable separately, then learn contextual softmax weights."""

    def __init__(self, variables, hidden, *, context=True, dropout=0.):
        super().__init__()
        self.variables = variables
        self.transforms = nn.ModuleList([
            GatedResidualNetwork(hidden, hidden, dropout=dropout) for _ in range(variables)])
        self.weights = GatedResidualNetwork(variables * hidden, hidden, variables,
                                            context=context, dropout=dropout)

    def forward(self, embeddings, context=None):
        transformed = torch.stack([layer(embeddings[..., i, :])
                                   for i, layer in enumerate(self.transforms)], dim=-2)
        weights = self.weights(embeddings.flatten(-2), context).softmax(dim=-1)
        return (transformed * weights.unsqueeze(-1)).sum(dim=-2), weights


class ScalarEmbeddings(nn.Module):
    def __init__(self, variables, hidden):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, hidden) for _ in range(variables)])

    def forward(self, values):
        return torch.stack([layer(values[..., i:i+1]) for i, layer in enumerate(self.layers)], dim=-2)


class InterpretableAttention(nn.Module):
    """Per-head Q/K, a shared value projection, and mean (not concatenated) heads."""

    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, self.width, bias=False)
        self.output = nn.Linear(self.width, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, encoder_hours):
        batch, length, _ = sequence.shape
        q = self.query(sequence[:, encoder_hours:]).reshape(batch, -1, self.heads, self.width).transpose(1, 2)
        k = self.key(sequence).reshape(batch, length, self.heads, self.width).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.width)
        query_position = torch.arange(encoder_hours, length, device=sequence.device)
        future = torch.arange(length, device=sequence.device)[None, :] > query_position[:, None]
        weights = scores.masked_fill(future[None, None], float('-inf')).softmax(dim=-1)
        heads = self.dropout(weights) @ self.value(sequence).unsqueeze(1)
        return self.output(heads.mean(dim=1)), weights.mean(dim=1)


@dataclass(frozen=True)
class TFTConfig:
    stations: int = 100
    hidden: int = 16
    heads: int = 2
    dropout: float = 0.1
    encoder_hours: int = 336
    horizon: int = 24
    known_features: int = 11
    observed_features: int = 5


class TemporalFusionTransformer(nn.Module):
    """Station-conditioned TFT with ordered q10/q50/q90 standardized outputs.

    One categorical static variable, continuous/mask temporal variables, one
    LSTM layer and decoder-query-only attention keep the implementation small.
    """

    def __init__(self, config=TFTConfig()):
        super().__init__()
        if (min(config.stations, config.hidden, config.heads, config.known_features,
                config.observed_features) < 1 or config.hidden % config.heads
                or config.hidden < 2 or not 0 <= config.dropout < 1
                or (config.encoder_hours, config.horizon) != (336, 24)):
            raise ValueError('Positive dimensions, divisible attention heads and 336/24 contract required')
        self.config = config
        h, d = config.hidden, config.dropout
        self.station = nn.Embedding(config.stations + 1, h, padding_idx=0)
        self.static_selection = VariableSelection(1, h, context=False, dropout=d)
        self.contexts = nn.ModuleDict({name: GatedResidualNetwork(h, h, dropout=d)
                                     for name in ('selection', 'enrichment', 'hidden', 'cell')})
        self.known_embeddings = ScalarEmbeddings(config.known_features, h)
        self.observed_embeddings = ScalarEmbeddings(config.observed_features, h)
        self.encoder_selection = VariableSelection(config.known_features + config.observed_features, h, dropout=d)
        self.decoder_selection = VariableSelection(config.known_features, h, dropout=d)
        self.encoder = nn.LSTM(h, h, batch_first=True)
        self.decoder = nn.LSTM(h, h, batch_first=True)
        self.local_gate = GateAddNorm(h, d)
        self.enrichment = GatedResidualNetwork(h, h, context=True, dropout=d)
        self.attention = InterpretableAttention(h, config.heads, d)
        self.attention_gate = GateAddNorm(h, d)
        self.positionwise = GatedResidualNetwork(h, h, dropout=d)
        self.final_gate = GateAddNorm(h, d)
        self.quantile_head = nn.Linear(h, 3)

    def forward(self, static_station_code, encoder_known, encoder_observed, decoder_known,
                *, return_diagnostics=False):
        c = self.config
        if static_station_code.ndim != 2 or static_station_code.shape[1] != 1:
            raise ValueError('Station codes must have shape (batch, 1)')
        batch = len(static_station_code)
        if static_station_code.dtype != torch.long or batch < 1:
            raise ValueError('Nonempty integer station codes required')
        if ((static_station_code < 1) | (static_station_code > c.stations)).any():
            raise ValueError('Station code outside frozen vocabulary')
        for value, shape in [(encoder_known, (batch, c.encoder_hours, c.known_features)),
                             (encoder_observed, (batch, c.encoder_hours, c.observed_features)),
                             (decoder_known, (batch, c.horizon, c.known_features))]:
            if tuple(value.shape) != shape or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError('Ordered finite feature tensors with the configured shapes required')
        # Enforce placeholder semantics even if a caller changes masked payloads.
        observed = encoder_observed.clone()
        if c.observed_features != 5:
            raise ValueError('Observed contract is count/mask/age/status/status-mask')
        for index in (1, 4):
            if not ((observed[..., index] == 0) | (observed[..., index] == 1)).all():
                raise ValueError('Observation/status masks must be binary')
        observed[..., 0] = observed[..., 0].masked_fill(observed[..., 1] == 0, 0)
        observed[..., 3] = observed[..., 3].masked_fill(observed[..., 4] == 0, 0)
        static, sw = self.static_selection(self.station(static_station_code))
        contexts = {name: layer(static) for name, layer in self.contexts.items()}
        past, pw = self.encoder_selection(torch.cat([
            self.known_embeddings(encoder_known), self.observed_embeddings(observed)], dim=-2), contexts['selection'])
        future, fw = self.decoder_selection(self.known_embeddings(decoder_known), contexts['selection'])
        past_output, states = self.encoder(past, (contexts['hidden'].unsqueeze(0), contexts['cell'].unsqueeze(0)))
        future_output, _ = self.decoder(future, states)
        local = self.local_gate(torch.cat([past_output, future_output], dim=1), torch.cat([past, future], dim=1))
        enriched = self.enrichment(local, contexts['enrichment'])
        attended, aw = self.attention(enriched, c.encoder_hours)
        attended = self.attention_gate(attended, enriched[:, c.encoder_hours:])
        features = self.final_gate(self.positionwise(attended), local[:, c.encoder_hours:])
        raw = self.quantile_head(features)
        median = raw[..., 1]
        prediction = torch.stack([median - F.softplus(raw[..., 0]), median,
                                  median + F.softplus(raw[..., 2])], dim=-1)
        if return_diagnostics:
            return prediction, {'static_weights': sw, 'encoder_weights': pw,
                'decoder_weights': fw, 'attention_weights': aw, 'static_representation': static}
        return prediction

    def artifact_config(self):
        return asdict(self.config)


def masked_pinball_loss(prediction, target, mask):
    """Mean across quantiles, sum across valid hours / valid hour count.

    Return None for an empty batch so the caller skips backward AND optimizer
    step (including weight decay). Masked NaN targets never enter arithmetic.
    """
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError('Predictions must have shape (batch, horizon, 3)')
    if target.shape != prediction.shape[:2] + (1,) or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError('Targets and boolean masks must have shape (batch, horizon, 1)')
    if not torch.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    valid = mask.squeeze(-1)
    if not valid.any():
        return None
    y = target.squeeze(-1)[valid]
    if not torch.isfinite(y).all():
        raise ValueError('Trusted targets must be finite')
    error = y.unsqueeze(-1) - prediction[valid]
    q = prediction.new_tensor(QUANTILES)
    return torch.maximum(q * error, (q - 1) * error).mean(dim=-1).sum() / valid.sum()


def inverse_quantiles(prediction, means, scales):
    """Invert frozen per-station affine scaling; no clipping or refitting."""
    if (means.shape != (len(prediction),) or scales.shape != means.shape
            or not torch.isfinite(means).all() or not torch.isfinite(scales).all() or not (scales > 0).all()):
        raise ValueError('Finite station means and positive scales required')
    return prediction * scales[:, None, None] + means[:, None, None]


def training_step(model, batch, optimizer):
    """One explicit step; empty loss skips optimizer updates, including decay."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = masked_pinball_loss(model(*batch.inputs()), batch.target, batch.target_mask)
    if loss is None:
        return None
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite model parameter after update')
    return {'loss': float(loss.detach()), 'gradient_norm_before_clipping': float(norm),
            'valid_targets': int(batch.target_mask.sum())}
