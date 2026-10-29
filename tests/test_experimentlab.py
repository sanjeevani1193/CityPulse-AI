"""Station-level inference, pre-only CUPED and reproducibility checks."""
from dataclasses import replace
import json
import numpy as np
import pytest
from scipy import stats
from citypulse_ai.experiments.redistribution import (Design, simulate, assign, fit_cuped,
    compare, planning, power, analyze, simulation_check)
from citypulse_ai.experiments.report import render


def test_seeded_assignment_and_exact_station_units():
    a=simulate();b=simulate()
    assert a['station_ids']==b['station_ids']
    for key in ['treated','pre','outcome']:np.testing.assert_array_equal(a[key],b[key])
    assert a['treated'].sum()==50 and len(a['outcome'])==100
    assert set(a['station_ids'])=={f'SIM{i:04d}' for i in range(100)}
    assert not np.array_equal(a['treated'],assign(a['station_ids'],43))
    np.testing.assert_allclose(a['outcome']-a['untreated_outcome'],3*a['treated'])
    assert np.all(a['post_total']>=0)


@pytest.mark.parametrize('ids',[['a']*8, list('abc'), ['']+list('abcdefg')])
def test_invalid_station_assignments(ids):
    with pytest.raises(ValueError):assign(ids,1)


def test_cuped_exact_coefficient_and_no_post_outcome_fit():
    x=np.array([1.,2.,4.,8.]);y=2*x+5
    assert fit_cuped(x,y)['theta']==pytest.approx(2)
    null=simulate(Design(effect=0));effect=simulate(Design(effect=20))
    assert null['calibration']==effect['calibration']
    np.testing.assert_array_equal(null['pre'],effect['pre'])
    np.testing.assert_array_equal(null['treated'],effect['treated'])
    # Assignment cannot change the calibration nuisance parameter either.
    different=simulate(Design(assignment_seed=8))
    assert null['calibration']==different['calibration']


def test_welch_matches_independent_manual_equations():
    y=np.array([1.,3.,5.,8.,9.,14.]);z=np.array([False]*3+[True]*3)
    result=compare(y,z)
    a,b=y[z],y[~z];v1=a.var(ddof=1)/3;v0=b.var(ddof=1)/3
    se=np.sqrt(v1+v0);df=(v1+v0)**2/(v1*v1/2+v0*v0/2)
    effect=a.mean()-b.mean();margin=stats.t.ppf(.975,df)*se
    assert result['effect']==pytest.approx(effect)
    assert result['standard_error']==pytest.approx(se)
    assert result['ci_low']==pytest.approx(effect-margin)
    assert result['ci_high']==pytest.approx(effect+margin)
    assert result['p_value']==pytest.approx(2*stats.t.sf(abs(effect/se),df))


def test_power_null_mde_sample_size_and_monotonicity():
    assert power(50,0,4)==pytest.approx(.05)
    assert np.isfinite(power(50,10,4))
    assert power(50,10,4) > .999
    assert power(100,2,4)>power(50,2,4)
    assert power(50,3,4)>power(50,2,4)
    p=planning(50,4)
    assert power(50,p['mde'],4)==pytest.approx(.8)
    n=p['required_stations_per_arm']
    assert power(n,2,4)>=.8 and power(n-1,2,4)<.8


@pytest.mark.parametrize('kwargs',[{'stations':7},{'stations':9},{'days':0},
    {'effect':float('nan')},{'effect':-.5},{'alpha':1},{'effect':.001}])
def test_invalid_design(kwargs):
    with pytest.raises(ValueError):simulate(replace(Design(),**kwargs))


def test_inference_rejects_missing_misaligned_or_degenerate_inputs():
    for y,z in [([1,float('nan'),2,3],[True,True,False,False]),
                ([1,2,3,4],[1,1,0,0]),([1,2,3,4],[True,False]),
                ([1,1,1,1],[True,True,False,False])]:
        with pytest.raises(ValueError):compare(y,np.array(z))
    with pytest.raises(ValueError):fit_cuped([1,2],[2,3])
    with pytest.raises(ValueError):fit_cuped([1,1,1],[2,3,4])
    with pytest.raises(ValueError):fit_cuped([1,2,3],[2,3])
    with pytest.raises(ValueError):power(1,2,4)
    with pytest.raises(ValueError):planning(50,4,target_power=.01)


def test_known_null_and_effect_and_no_forced_precision_gain():
    null=analyze(Design(effect=0));effect=analyze(Design(effect=3))
    assert effect['cuped']['effect']-null['cuped']['effect']==pytest.approx(3)
    assert effect['unadjusted']['effect']-null['unadjusted']['effect']==pytest.approx(3)
    # Miscalibrated theta can increase uncertainty: the formula must expose that.
    d=simulate();raw=compare(d['outcome'],d['treated'])
    bad=compare(d['outcome']+10*d['pre'],d['treated'])
    assert 1-(bad['standard_error']/raw['standard_error'])**2<0


def test_repeated_simulations_are_reproducible_and_null_is_calibrated():
    a=simulation_check(100)
    assert a==simulation_check(100)
    assert 0<=a[0]['cuped_rejection_rate']<=.15
    assert .85<=a[0]['cuped_ci_coverage']<=1
    assert abs(a[0]['cuped_mean_estimate'])<.3
    assert abs(a[1]['cuped_mean_estimate']-3)<.3
    assert a[1]['cuped_rejection_ci'][0] < 1
    assert a[1]['cuped_rejection_ci'][1] <= 1


def test_report_is_standalone_and_explicitly_synthetic():
    result=analyze();result['simulation_checks']=simulation_check(20)
    text=render(result)
    assert 'Fully simulated' in text and '<svg' in text
    assert '<script' not in text and '<iframe' not in text
    assert 'Welch' in text and 'pre-treatment' in text
    json.dumps(result,allow_nan=False)


def test_cli_create_only_report_and_checksum_recovery(tmp_path,monkeypatch):
    import hashlib
    import sys
    from citypulse_ai.experiments.redistribution import main
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['experimentlab','--output',str(output),'--simulation-replicates','20'])
    main()
    manifest=json.loads((output/'checksums.json').read_text())
    assert set(manifest)=={'experiment.json','report.html'}
    for name,digest in manifest.items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest
    before=(output/'experiment.json').read_bytes()
    with pytest.raises(FileExistsError):main()
    assert (output/'experiment.json').read_bytes()==before
