"""Public-hosting settings and a source-only, network-denied release smoke test."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_hosting_does_not_force_loopback_or_disable_request_protections():
    settings = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
    assert "address" not in settings["server"]
    assert settings["server"].get("enableCORS", True)
    assert settings["server"].get("enableXsrfProtection", True)
    assert settings["browser"]["gatherUsageStats"] is False
    assert settings["client"]["showErrorDetails"] is False


def test_entrypoint_has_its_own_lightweight_dependency_file():
    folder = ROOT / "apps/decision_studio"
    assert (folder / "streamlit_app.py").is_file()
    lines = {line.split("==")[0] for line in (folder / "requirements.txt").read_text().splitlines()
             if line and not line.startswith("#")}
    assert lines == {"streamlit", "plotly", "pandas", "pyarrow", "duckdb", "pytz"}
    # Higher-priority dependency files must not override this app's requirements.
    assert not any((folder / name).exists() for name in ("uv.lock", "Pipfile", "environment.yml"))


def test_source_only_checkout_renders_all_pages_without_outbound_connections(tmp_path):
    checkout = tmp_path / "checkout"
    shutil.copytree(ROOT / "apps/decision_studio", checkout / "apps/decision_studio",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "src/citypulse_ai/decision_studio", checkout / "src/citypulse_ai/decision_studio",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("__init__.py", "sqlpilot/__init__.py", "sqlpilot/core.py"):
        path = checkout / "src/citypulse_ai" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "src/citypulse_ai" / name, path)
    (checkout / "docs").mkdir()
    shutil.copyfile(ROOT / "docs/experimentlab_report.html", checkout / "docs/experimentlab_report.html")
    shutil.copytree(ROOT / ".streamlit", checkout / ".streamlit")
    assert not (checkout / "outputs").exists()
    # Isolated interpreter: no editable install or original source PYTHONPATH.
    # Audit hook denies outbound sockets and tracks any subprocess attempts.
    script = r'''
import json, pathlib, sys
children = []
def audit(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        raise AssertionError("Unexpected outbound network attempt")
    if event == "subprocess.Popen":
        command = args[1]
        assert isinstance(command, list) and command[0] == sys.executable
        assert command[1].endswith("sql_demo_worker.py"), command
        children.append(command[1])
sys.addaudithook(audit)
from streamlit.testing.v1 import AppTest
entry = "apps/decision_studio/streamlit_app.py"
app = AppTest.from_file(entry, default_timeout=25).run()
assert not app.exception
for name in ("Overview", "Model Comparison", "Forecast Explorer", "SQLPilot", "ExperimentLab & Methodology"):
    app.sidebar.radio[0].set_value(name).run()
    assert not app.exception, name
    if name == "SQLPilot":
        app.button(key="sql_run").click().run()
        assert not app.exception
        assert any(table.value.to_dict("records") == [{"starts":1998}] for table in app.dataframe)
    if name == "ExperimentLab & Methodology":
        assert any(m.value == "+2.7531" for m in app.metric)
        assert len(app.get("download_button")) == 1
assert children
assert not any(name in sys.modules for name in ("torch", "tensorflow", "google.cloud", "httpx", "citypulse_ai.sqlpilot.app"))
print(json.dumps({"source_only_pages":5,"sql_starts":1998,"outbound_network_denied":True}))
'''
    env = {key: os.environ[key] for key in ("PATH", "TMPDIR") if key in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=checkout, env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr[-2500:]
    assert '"source_only_pages": 5' in result.stdout
