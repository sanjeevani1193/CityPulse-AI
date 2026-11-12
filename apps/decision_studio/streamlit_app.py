"""Community Cloud/local entrypoint; existing six-model apps remain separate."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from citypulse_ai.decision_studio.app import main

main()
