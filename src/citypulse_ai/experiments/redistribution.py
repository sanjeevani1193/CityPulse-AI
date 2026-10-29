"""Synthetic station-randomized redistribution experiment; no historical input.

One station = one unit. CUPED theta is frozen from two earlier pre-treatment
periods, never fitted to experimental outcomes or assignment. No model training.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy import optimize, stats


@dataclass(frozen=True)
class Design:
    stations: int = 100
    days: int = 28
    effect: float = 3.0
    practical_threshold: float = 2.0
    seed: int = 2026
    assignment_seed: int = 42
    alpha: float = 0.05

    def validate(self):
        if not isinstance(self.stations, int) or self.stations < 8 or self.stations % 2:
            raise ValueError('Require an even number of at least eight stations')
        if not isinstance(self.days, int) or self.days < 1:
            raise ValueError('Days must be a positive integer')
        if not math.isfinite(self.effect) or self.effect < 0 or not np.isclose(self.effect*self.days, round(self.effect*self.days)):
            raise ValueError('Effect must be nonnegative with an integer period-start increment')
        if not 0 < self.alpha < 1 or not math.isfinite(self.practical_threshold) or self.practical_threshold <= 0:
            raise ValueError('Invalid significance/practical threshold')


def vector(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Require finite one-dimensional observations, at least two')
    return values


def assign(station_ids, seed):
    """Balanced complete randomization; exact string IDs, no duplicates."""
    ids = list(station_ids)
    if len(ids) < 8 or len(ids) % 2 or len(set(ids)) != len(ids) or not all(isinstance(s, str) and s for s in ids):
        raise ValueError('Require unique nonempty station IDs and even n >= 8')
    assignment = np.zeros(len(ids), dtype=bool)
    assignment[np.random.default_rng(seed).permutation(len(ids))[:len(ids)//2]] = True
    return assignment


def fit_cuped(earlier_x, earlier_y):
    """Only pre-treatment calibration inputs belong here; caller records periods."""
    x, y = vector(earlier_x), vector(earlier_y)
    if len(x) < 3 or len(x) != len(y) or np.var(x, ddof=1) <= 0:
        raise ValueError('Calibration covariate must vary and align with outcome')
    theta = float(np.cov(x, y, ddof=1)[0, 1]/np.var(x, ddof=1))
    residual_sd = float(np.std(y-theta*(x-x.mean()), ddof=2))
    return {'theta': theta, 'calibration_sd': float(np.std(y, ddof=1)),
            'calibration_adjusted_sd': residual_sd,
            'fit_scope': 'pre_period_1 -> pre_period_2; before assignment and outcome'}


def simulate(design=Design()):
    design.validate()
    rng = np.random.default_rng(design.seed)
    latent_daily_rate = rng.lognormal(np.log(80)-0.35**2/2, 0.35, design.stations)
    # Nonoverlapping 28-day periods: calibration1, calibration2, baseline, post.
    totals = rng.poisson(latent_daily_rate[:, None]*design.days, size=(design.stations, 4))
    calibration = fit_cuped(totals[:, 0]/design.days, totals[:, 1]/design.days)
    ids = [f'SIM{i:04d}' for i in range(design.stations)]
    treated = assign(ids, design.assignment_seed)
    post_total = totals[:, 3] + treated.astype(int)*round(design.effect*design.days)
    return {'station_ids': ids, 'treated': treated, 'pre': totals[:, 2]/design.days,
            'outcome': post_total/design.days, 'post_total': post_total,
            'untreated_outcome': totals[:, 3]/design.days, 'calibration': calibration}


def compare(outcome, treated, alpha=0.05):
    y = vector(outcome)
    z = np.asarray(treated)
    if z.dtype != np.bool_ or z.shape != y.shape or min(z.sum(), (~z).sum()) < 2 or not 0 < alpha < 1:
        raise ValueError('Aligned boolean assignment and two stations per arm required')
    a, b = y[z], y[~z]
    se2 = float(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
    if se2 <= 0:
        raise ValueError('Inference requires positive sampling variance')
    test = stats.ttest_ind(a, b, equal_var=False)
    ci = test.confidence_interval(confidence_level=1-alpha)
    return {'treatment_n': len(a), 'control_n': len(b),
            'treatment_mean': float(a.mean()), 'control_mean': float(b.mean()),
            'treatment_sd': float(a.std(ddof=1)), 'control_sd': float(b.std(ddof=1)),
            'effect': float(a.mean()-b.mean()), 'standard_error': math.sqrt(se2),
            'ci_low': float(ci.low), 'ci_high': float(ci.high),
            'p_value': float(test.pvalue), 'df': float(test.df),
            'significant': bool(test.pvalue < alpha)}


def power(n_per_arm, effect, sd, alpha=0.05):
    """Planning approximation: equal arms/variance, two-sided noncentral t."""
    if not isinstance(n_per_arm, int) or n_per_arm < 2 or not (math.isfinite(sd) and sd > 0) or not math.isfinite(effect) or effect < 0 or not 0 < alpha < 1:
        raise ValueError('Invalid power inputs')
    df = 2*n_per_arm-2
    critical = stats.t.ppf(1-alpha/2, df)
    noncentrality = effect/(sd*math.sqrt(2/n_per_arm))
    # Symmetry avoids SciPy returning NaN in the tiny negative CDF tail
    # for moderately large positive noncentrality. Both terms are exact tails.
    return float(stats.nct.sf(critical, df, noncentrality)+stats.nct.sf(critical, df, -noncentrality))


def planning(n_per_arm, sd, target_effect=2.0, target_power=0.8, alpha=0.05):
    if not 0 < target_power < 1 or target_power <= alpha or target_effect <= 0:
        raise ValueError('Require positive planning effect and alpha < target power < 1')
    achieved = power(n_per_arm, target_effect, sd, alpha)
    mde = optimize.brentq(lambda e: power(n_per_arm, e, sd, alpha)-target_power, 0, sd*100)
    low, high = 2, 100000
    if power(high, target_effect, sd, alpha) < target_power:
        raise ValueError('Required sample exceeds planning bound')
    while low < high:
        middle = (low+high)//2
        if power(middle, target_effect, sd, alpha) >= target_power: high = middle
        else: low = middle+1
    return {'sd_source': 'pre-treatment calibration only', 'sd': sd, 'n_per_arm': n_per_arm,
            'target_effect': target_effect, 'target_power': target_power, 'power': achieved,
            'mde': float(mde), 'required_stations_per_arm': low,
            'assumption': 'equal independent station variances; approximate noncentral t planning, not observed power'}


def analyze(design=Design()):
    data = simulate(design)
    y, x, z = data['outcome'], data['pre'], data['treated']
    calibration = data['calibration']
    adjusted = y-calibration['theta']*(x-x.mean())
    raw, cuped = compare(y, z, design.alpha), compare(adjusted, z, design.alpha)
    balance = compare(x, z, design.alpha)
    pooled_sd = math.sqrt((x[z].var(ddof=1)+x[~z].var(ddof=1))/2)
    balance['standardized_mean_difference'] = balance['effect']/pooled_sd
    raw['relative_lift_percent'] = 100*raw['effect']/raw['control_mean']
    # Predeclared primary estimator: CUPED; raw is sensitivity, not another win criterion.
    conclusion = ('supports practical benefit in this simulation' if cuped['ci_low'] > design.practical_threshold
                  else 'statistically positive but practical threshold uncertain' if cuped['ci_low'] > 0
                  else 'inconclusive about positive benefit')
    return {'version': 'experimentlab-redistribution-v1', 'data_kind': 'fully synthetic; no historical observations',
            'design': asdict(design), 'primary_estimator': 'pre-calibrated CUPED Welch difference in station means',
            'estimand': 'station-average intention-to-treat effect in starts/station/local-calendar-day over fixed post period',
            'periods': {'calibration_x': [0, 28], 'calibration_y': [28, 56], 'baseline': [56, 84], 'post': [84, 112]} if design.days == 28 else
                       {k: [i*design.days, (i+1)*design.days] for i,k in enumerate(('calibration_x','calibration_y','baseline','post'))},
            'calibration': calibration, 'balance': balance, 'unadjusted': raw, 'cuped': cuped,
            'variance_reduction_fraction': 1-(cuped['standard_error']/raw['standard_error'])**2,
            'planning_raw': planning(design.stations//2, calibration['calibration_sd'], design.practical_threshold, alpha=design.alpha),
            'planning_cuped': planning(design.stations//2, calibration['calibration_adjusted_sd'], design.practical_threshold, alpha=design.alpha),
            'conclusion': conclusion,
            'station_records': [{'station_id': station, 'treated': bool(z[i]), 'baseline_daily_starts': float(x[i]),
                                 'post_total_starts': int(data['post_total'][i]), 'post_daily_starts': float(y[i])}
                                for i,station in enumerate(data['station_ids'])],
            'implementation_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                      for name in ('redistribution.py', 'report.py')},
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}}


def simulation_check(repetitions=200):
    if not isinstance(repetitions, int) or not 20 <= repetitions <= 1000:
        raise ValueError('Use 20–1000 independent simulation replicates')
    rows = []
    for effect in (0., 3.):
        raw_reject = cuped_reject = coverage = 0
        estimates = []; raw_estimates = []
        for i in range(repetitions):
            d = Design(effect=effect, seed=10000+i, assignment_seed=20000+i)
            data = simulate(d); x, y, z = data['pre'], data['outcome'], data['treated']
            r = compare(y, z); c = compare(y-data['calibration']['theta']*(x-x.mean()), z)
            raw_reject += r['significant']; cuped_reject += c['significant']
            coverage += c['ci_low'] <= effect <= c['ci_high']
            estimates.append(c['effect']);raw_estimates.append(r['effect'])
        rejection_ci = stats.binomtest(cuped_reject, repetitions).proportion_ci(method='exact')
        coverage_ci = stats.binomtest(coverage, repetitions).proportion_ci(method='exact')
        rows.append({'cuped_rejection_ci': [float(rejection_ci.low), float(rejection_ci.high)],
                     'cuped_coverage_ci': [float(coverage_ci.low), float(coverage_ci.high)],
                     'true_effect': effect, 'replicates': repetitions,
                     'raw_rejection_rate': raw_reject/repetitions, 'cuped_rejection_rate': cuped_reject/repetitions,
                     'cuped_ci_coverage': coverage/repetitions, 'cuped_mean_estimate': float(np.mean(estimates)),
                     'raw_mean_estimate': float(np.mean(raw_estimates)),
                     'rejection_rate_mc_se': math.sqrt((cuped_reject/repetitions)*(1-cuped_reject/repetitions)/repetitions)})
    return rows


def main():
    import argparse
    from citypulse_ai.experiments.report import render
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/experimentlab/redistribution-v1'))
    parser.add_argument('--simulation-replicates', type=int, default=200)
    args = parser.parse_args()
    result = analyze();result['simulation_checks'] = simulation_check(args.simulation_replicates)
    # Create-only run directory: never overwrite a previously saved experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(result, indent=2, allow_nan=False).encode()
    (args.output/'experiment.json').write_bytes(payload)
    (args.output/'report.html').write_text(render(result))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())}
    (args.output/'checksums.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'output': str(args.output), 'raw': result['unadjusted'], 'cuped': result['cuped'],
                      'variance_reduction_fraction': result['variance_reduction_fraction'],
                      'planning_raw': result['planning_raw'], 'planning_cuped': result['planning_cuped'],
                      'simulation_checks': result['simulation_checks']}, indent=2))


if __name__ == '__main__': main()
