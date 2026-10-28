"""Frozen public artifact identities; no framework or credential imports."""
from importlib.resources import files
import json


def catalog():
    return json.loads(files('citypulse_ai.api').joinpath('station_artifacts.json').read_text())
