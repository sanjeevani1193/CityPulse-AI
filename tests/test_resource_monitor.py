"""Synthetic OS counters only; no host pressure sampling or cloud access."""

import pytest

from citypulse_ai.etl import resource_monitor as monitor


@pytest.mark.parametrize('system,factor',[('Darwin',1),('Linux',1024)])
def test_peak_rss_units(system,factor):
    assert monitor.peak_rss_bytes(system=system,value=12345) == 12345*factor


@pytest.mark.parametrize('pressure,unsafe',[(1,False),(2,True),(4,True)])
def test_macos_snapshot_distinguishes_capacity_use_and_pressure(pressure,unsafe):
    def command(args):
        if 'vm.swapusage' in args:return 'total = 4096.00M  used = 1200.50M  free = 2895.50M'
        if 'kern.memorystatus_vm_pressure_level' in args:return str(pressure)
        if 'hw.memsize' in args:return str(8*1024**3)
        assert args[:3] == ['/bin/ps','-o','rss=']
        return '1234'
    s=monitor.snapshot(system='Darwin',read_command=command)
    assert s['rss_bytes']==1234*1024
    assert s['swap_allocated_bytes']==4*1024**3
    assert s['swap_used_bytes']==int(1200.5*1024**2)
    assert s['memory_pressure_unsafe'] is unsafe and s['memory_available_bytes'] is None


def linux(proc,available=8*1024**2,swap_total=1024,swap_free=1000,psi='full avg10=0.00 avg60=0.00 avg300=0.00 total=0'):
    proc.mkdir(exist_ok=True);(proc/'self').mkdir(exist_ok=True);(proc/'pressure').mkdir(exist_ok=True)
    (proc/'meminfo').write_text(f'MemTotal: 16777216 kB\nMemAvailable: {available} kB\nSwapTotal: {swap_total} kB\nSwapFree: {swap_free} kB\n')
    (proc/'self/status').write_text('Name: python\nVmRSS: 1234 kB\n')
    if psi is not None:(proc/'pressure/memory').write_text(psi+'\n')
    return monitor.snapshot(system='Linux',proc_root=proc,
        read_command=lambda *args:pytest.fail('Linux must not run macOS commands'))


def test_linux_units_and_healthy_swapless_host(tmp_path):
    s=linux(tmp_path,swap_total=0,swap_free=0)
    assert s['rss_bytes']==1234*1024 and s['memory_total_bytes']==16*1024**3
    assert s['memory_available_bytes']==8*1024**3
    assert s['swap_used_bytes']==s['swap_allocated_bytes']==0
    assert not s['memory_pressure_unsafe']


@pytest.mark.parametrize('kwargs',[{'available':100},{'psi':'full avg10=10.00 avg60=0 avg300=0 total=1'}])
def test_linux_pressure_stops_on_low_available_or_stalls(tmp_path,kwargs):
    assert linux(tmp_path,**kwargs)['memory_pressure_unsafe']


def test_linux_psi_unavailable_is_explicit_and_ram_guard_still_applies(tmp_path):
    s=linux(tmp_path,available=100,psi=None)
    assert s['memory_psi_full_avg10'] is None and 'unavailable' in s['pressure_method']
    assert s['memory_pressure_unsafe']


@pytest.mark.parametrize('kwargs',[{'swap_total':1,'swap_free':2},{'psi':'malformed'},
                                  {'psi':'full avg10=101 avg60=0 avg300=0 total=0'}])
def test_invalid_linux_counters_fail_closed(tmp_path,kwargs):
    with pytest.raises(ValueError):linux(tmp_path,**kwargs)


@pytest.mark.parametrize('text',['unavailable','total = 1M used = 2M free = 0M',
                                'total = 1P used = 0M free = 0M'])
def test_invalid_macos_swap_fails_closed(text):
    with pytest.raises(ValueError):monitor.parse_macos_swap(text)


def test_unsupported_platform_fails_closed():
    with pytest.raises(ValueError):monitor.snapshot(system='Windows')
