"""CI safety guards; no artifact access, network calls, or silently skipped checks."""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def forbid_local_artifacts(monkeypatch):
    original = Path.open
    def guarded_open(path, *args, **kwargs):
        resolved = path.resolve()
        for directory in (ROOT / 'outputs', ROOT / 'csv_files'):
            if resolved.is_relative_to(directory):
                pytest.fail(f'CI test accessed local data/artifacts: {path}')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)


def pytest_collectreport(report):
    if os.environ.get('CITYPULSE_CI_REQUIRE_NO_SKIPS') == '1' and report.skipped:
        pytest.exit(f'Required CI module skipped: {report.nodeid}', returncode=1)


def pytest_runtest_logreport(report):
    if os.environ.get('CITYPULSE_CI_REQUIRE_NO_SKIPS') == '1' and report.skipped:
        pytest.exit(f'Required CI test skipped: {report.nodeid}', returncode=1)
