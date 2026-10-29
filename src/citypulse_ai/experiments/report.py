"""Self-contained HTML/SVG report; no frontend framework or external assets."""
from html import escape


def number(value):
    return f'{value:.4f}'


def render(result):
    raw, adjusted = result['unadjusted'], result['cuped']
    low = min(0, raw['ci_low'], adjusted['ci_low'])-1
    high = max(raw['ci_high'], adjusted['ci_high'], result['design']['effect'])+1
    def scale(x): return 100+(x-low)/(high-low)*560
    svg = ['<svg viewBox="0 0 760 190" role="img" aria-label="Station treatment effects with 95 percent confidence intervals">']
    for label, value, y, color in [('Raw', raw, 60, '#526777'), ('CUPED', adjusted, 110, '#007c78')]:
        svg += [f'<text x="8" y="{y+5}">{label}</text>',
                f'<line x1="{scale(value["ci_low"])}" x2="{scale(value["ci_high"])}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="5"/>',
                f'<circle cx="{scale(value["effect"])}" cy="{y}" r="6" fill="{color}"><title>{label}: {value["effect"]:.4f}</title></circle>']
    for value, label, color in [(0, 'Null', '#526777'), (result['design']['effect'], 'Simulated effect', '#b34a00')]:
        svg += [f'<line x1="{scale(value)}" x2="{scale(value)}" y1="20" y2="140" stroke="{color}" stroke-dasharray="4 4"/>',
                f'<text x="{scale(value)}" y="{160 if value == 0 else 180}" text-anchor="middle">{label}: {value:g}</text>']
    svg.append('</svg>')
    rows = ''
    for name, v in [('Unadjusted sensitivity', raw), ('CUPED primary', adjusted)]:
        rows += '<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in [name, number(v['control_mean']), number(v['treatment_mean']), number(v['effect']),
                            f'[{v["ci_low"]:.4f}, {v["ci_high"]:.4f}]', f'{v["p_value"]:.4g}'])+'</tr>'
    planning = ''.join(f'<tr><td>{name}</td><td>{v["sd"]:.3f}</td><td>{100*v["power"]:.2f}%</td><td>{v["mde"]:.3f}</td><td>{v["required_stations_per_arm"]}</td></tr>'
                       for name,v in [('Raw',result['planning_raw']),('CUPED',result['planning_cuped'])])
    checks = ''.join(f'<tr><td>{v["true_effect"]:g}</td><td>{v["replicates"]}</td><td>{v["raw_rejection_rate"]:.3f}</td><td>{v["cuped_rejection_rate"]:.3f} [{v["cuped_rejection_ci"][0]:.3f}, {v["cuped_rejection_ci"][1]:.3f}]</td><td>{v["cuped_ci_coverage"]:.3f}</td><td>{v["cuped_mean_estimate"]:.3f}</td></tr>' for v in result.get('simulation_checks', []))
    records = result['station_records']
    maximum = max(max(r['baseline_daily_starts'],r['post_daily_starts']) for r in records)*1.05
    scatter = ['<svg viewBox="0 0 500 300" role="img" aria-label="Synthetic station baseline versus post starts; treatment teal, control gray">',
               '<text x="70" y="16">Post starts/day</text><text x="250" y="292">Baseline starts/day</text>',
               '<line x1="55" y1="255" x2="460" y2="255" stroke="#526777"/><line x1="55" y1="255" x2="55" y2="25" stroke="#526777"/>']
    for r in records:
        x=55+400*r['baseline_daily_starts']/maximum;y=255-220*r['post_daily_starts']/maximum
        color='#007c78' if r['treated'] else '#526777'
        scatter.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"><title>{escape(r["station_id"])}: baseline {r["baseline_daily_starts"]:.2f}, post {r["post_daily_starts"]:.2f}</title></circle>')
    for v in (0,maximum/2,maximum):
        scatter.append(f'<text x="{55+400*v/maximum:.2f}" y="275">{v:.0f}</text><text x="10" y="{255-220*v/maximum:.2f}">{v:.0f}</text>')
    scatter.append('</svg>')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CityPulse AI ExperimentLab — synthetic station experiment</title>
<style>body{{max-width:960px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui;color:#172b37;background:#fafcfc}}h1,h2{{line-height:1.2}}.notice{{border-left:5px solid #b34a00;padding:14px;background:#fff2df}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #cad8dc}}svg{{width:100%;max-height:330px;background:white}}code{{overflow-wrap:anywhere}}.scroll{{overflow:auto}}</style>
<h1>ExperimentLab: bike redistribution</h1><p class="notice"><b>Fully simulated experiment — no real intervention or historical data.</b>
These results demonstrate methodology and cannot establish that redistribution works in NYC.</p>
<h2>Predeclared design</h2><p>Does hypothetical redistribution increase accepted recorded starts?
Randomize {result['design']['stations']} synthetic stations equally, with one fixed {result['design']['days']}-day post metric per station.
Primary estimand: station-average intention-to-treat difference in starts/station/day. H₀: effect = 0; two-sided α = {result['design']['alpha']}.
Injected effect: {result['design']['effect']:g} starts/station/day. Practical threshold: {result['design']['practical_threshold']:g}.
Station-hours are not independent experimental units. No repeated peeking, stopping, subgroup search or metric selection.</p>
<p>Four nonoverlapping periods: two pre-treatment calibration periods, a baseline period, then post.
Theta is frozen using only the first two periods. Balanced assignment uses seed {result['design']['assignment_seed']}; simulation seed {result['design']['seed']}.
Data are synthetic Poisson period totals with shared lognormal station activity; no spillovers, trends or missingness.</p>
<h2>Station-level comparison</h2><p>Control n={raw['control_n']}, treatment n={raw['treatment_n']}.
Raw post SD: control {raw['control_sd']:.3f}, treatment {raw['treatment_sd']:.3f}.
Baseline treatment-minus-control: {result['balance']['effect']:.3f}; standardized difference {result['balance']['standardized_mean_difference']:.3f}.
Balance is diagnostic; its p-value is not proof of successful randomization and does not gate analysis.</p>
<div class="scroll"><table><tr><th>Estimator</th><th>Control mean</th><th>Treatment mean</th><th>Effect</th><th>95% CI</th><th>Two-sided p</th></tr>{rows}</table></div>
<p>CUPED arm means are on an adjusted scale, not newly observed counts. Raw relative lift: {raw['relative_lift_percent']:.2f}%.
Intervals and p-values use Welch's t-test across stations, with no equal-variance assumption.</p>
{''.join(svg)}
<h2>CUPED and precision</h2><p>Yadjusted = Y − θ(X − mean(X)); frozen θ = {result['calibration']['theta']:.5f}.
Measured reduction in the estimated variance of the treatment-effect estimator: {100*result['variance_reduction_fraction']:.2f}%.
This can be negative in other samples; it is not forced to be positive.</p>
{''.join(scatter)}<p>Teal: treatment; gray: control. Each point is one simulated station.</p>
<h2>Prospective planning approximation</h2><p>Using pre-treatment calibration SDs only, equal arms/variance and a two-sided noncentral-t model:
power for a {result['design']['practical_threshold']:g}-start effect with {raw['control_n']} stations/arm; MDE and sample size target 80% power.
No post-hoc “observed power” is used. Welch inference and this equal-variance planning approximation are distinct.</p>
<table><tr><th>Plan</th><th>Pre SD</th><th>Power</th><th>MDE</th><th>Required stations/arm</th></tr>{planning}</table>
<h2>Independent simulation checks</h2><table><tr><th>True effect</th><th>Replicates</th><th>Raw rejection</th><th>CUPED rejection [exact 95% CI]</th><th>CUPED CI coverage</th><th>Mean CUPED effect</th></tr>{checks}</table>
<p>Monte Carlo rates have sampling uncertainty; at 200 replicates, a 5% rejection rate has roughly 1.5 percentage-point standard error.
These checks assess this assumed data-generating process, not real-world calibration.</p>
<h2>Decision and limitations</h2><p>{escape(result['conclusion'])}. The primary practical criterion is a CUPED lower CI bound exceeding the predeclared threshold.
Do not deploy an intervention from this simulation. Real stations may interfere through redistribution and rider substitution;
use network/cluster or geographic designs, assess power at that level, and monitor service levels, safety and redistribution costs.
Recorded starts are not latent demand or lost rentals. Spillovers, seasonality, heterogeneous effects, missing data and temporal drift could invalidate this demonstration's assumptions.</p>
<h2>Reproduce</h2><p><code>PYTHONPATH=src .venv/bin/python -m citypulse_ai.experiments.redistribution --output outputs/experimentlab/new-run</code></p>
<p>Output is create-only: JSON station records/config/results, HTML, and SHA-256 checksums. Runtime: Python {escape(result['runtime']['python'])}, NumPy {escape(result['runtime']['numpy'])}, SciPy {escape(result['runtime']['scipy'])}.</p>
<p>Method references: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html">SciPy Welch test and confidence intervals</a>;
<a href="https://docs.datadoghq.com/experiments/statistics/cuped/">CUPED adjustment and fixed coefficients</a>.</p></html>'''
