"""Small, read-only host/process snapshots. Host swap is not process attribution."""

import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess


def command(args):
    return subprocess.check_output(args, text=True, timeout=5).strip()


def peak_rss_bytes(*, system=None, value=None):
    system = system or platform.system()
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if value is None else value
    if system not in {'Darwin', 'Linux'}:
        raise ValueError(f'Unsupported resource-monitor platform: {system}')
    return int(value * (1024 if system == 'Linux' else 1))


def parse_macos_swap(text):
    values = {}
    for name in ('total', 'used', 'free'):
        match = re.search(rf'\b{name}\s*=\s*([0-9.]+)([KMGT]?)(?=\s|$)', text)
        if match is None:
            raise ValueError('Cannot parse macOS swap usage')
        values[name] = int(float(match[1]) * 1024**(' KMGT'.index(match[2]) if match[2] else 0))
    if values['used'] > values['total'] or values['free'] > values['total']:
        raise ValueError('Inconsistent macOS swap usage')
    return values


def snapshot(*, system=None, proc_root=Path('/proc'), read_command=command):
    system = system or platform.system()
    if system == 'Darwin':
        raw = read_command(['/usr/sbin/sysctl', 'vm.swapusage'])
        swap = parse_macos_swap(raw)
        pressure = int(read_command(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level']))
        if pressure not in {1, 2, 4}:
            raise ValueError('Unknown macOS memory pressure level')
        rss = int(read_command(['/bin/ps', '-o', 'rss=', '-p', str(os.getpid())])) * 1024
        total = int(read_command(['/usr/sbin/sysctl', '-n', 'hw.memsize']))
        if rss < 0 or total <= 0:
            raise ValueError('Invalid macOS memory counters')
        return {'platform': system, 'rss_bytes': rss, 'swap': raw,
                'swap_allocated_bytes': swap['total'], 'swap_used_bytes': swap['used'],
                'memory_total_bytes': total, 'memory_available_bytes': None,
                'pressure_method': 'macOS kernel pressure level (1 normal, 2 warning, 4 critical)',
                'memory_pressure_level': pressure, 'memory_pressure_unsafe': pressure != 1}
    if system != 'Linux':
        raise ValueError(f'Unsupported resource-monitor platform: {system}')
    memory = {k: int(v)*1024 for k, v in re.findall(r'^(\w+):\s+(\d+) kB$',
              (proc_root/'meminfo').read_text(), re.MULTILINE)}
    rss_match = re.search(r'^VmRSS:\s+(\d+) kB$',
                         (proc_root/'self/status').read_text(), re.MULTILINE)
    if rss_match is None:
        raise ValueError('Linux process RSS unavailable')
    total, available, allocated, free = (memory[k] for k in
        ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'))
    if total <= 0 or available > total or free > allocated:
        raise ValueError('Inconsistent Linux memory counters')
    psi = None
    pressure_path = proc_root/'pressure/memory'
    if pressure_path.exists():
        match = re.search(r'^full avg10=([0-9.]+)\b', pressure_path.read_text(), re.MULTILINE)
        if match is None:
            raise ValueError('Cannot parse Linux memory PSI')
        psi = float(match[1])
        if not math.isfinite(psi) or not 0 <= psi <= 100:
            raise ValueError('Invalid Linux memory PSI')
    # Policy: stop below 5% available RAM (at least 256 MiB), or when all
    # runnable work stalls for memory >=10% over the PSI ten-second average.
    unsafe = available < max(256*1024**2, total//20) or (psi is not None and psi >= 10)
    return {'platform': system, 'rss_bytes': int(rss_match[1])*1024,
            'swap_allocated_bytes': allocated, 'swap_used_bytes': allocated-free,
            'memory_total_bytes': total, 'memory_available_bytes': available,
            'memory_psi_full_avg10': psi, 'memory_pressure_unsafe': unsafe,
            'pressure_method': 'MemAvailable plus PSI' if psi is not None else 'MemAvailable (PSI unavailable)'}
