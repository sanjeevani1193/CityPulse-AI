"""Real loopback HTTP/WebSocket smoke test; always stop the owned app process."""
import asyncio
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]


async def render(port):
    theme_verified = False
    async with connect(f"ws://127.0.0.1:{port}/_stcore/stream",
                       origin=f"http://127.0.0.1:{port}", subprotocols=["streamlit"]) as ws:
        async def rerun(widget_id=None, workspace="Model Comparison"):
            nonlocal theme_verified
            request = BackMsg()
            request.rerun_script.query_string = ""
            if widget_id:
                widget = request.rerun_script.widget_states.widgets.add()
                widget.id = widget_id
                # Streamlit 1.66 radio widgets serialize formatted string values.
                widget.string_value = workspace
            await ws.send(request.SerializeToString())
            elements = []
            for _ in range(200):
                raw = await asyncio.wait_for(ws.recv(), timeout=25)
                message = ForwardMsg()
                message.ParseFromString(raw)
                if message.HasField("new_session"):
                    theme = message.new_session.custom_theme
                    assert theme.background_color == "#F3F7FA" and theme.text_color == "#21384B"
                    assert theme.primary_color == "#167E87"
                    assert theme.sidebar.background_color == "#102D40" and theme.sidebar.text_color == "#FFFFFF"
                    for scheme in (theme.light, theme.dark):
                        assert scheme.background_color == "#F3F7FA" and scheme.text_color == "#21384B"
                    theme_verified = True
                if message.HasField("delta") and message.delta.HasField("new_element"):
                    element = message.delta.new_element
                    if element.HasField("exception"):
                        # Only application exception type/message, never environment/log dumps.
                        raise AssertionError("Streamlit application exception: " + element.exception.type + ": " + element.exception.message)
                    elements.append(element)
                if message.HasField("script_finished"):
                    return elements
            raise AssertionError("App did not finish rendering")
        overview = await rerun()
        assert sum(e.HasField("metric") for e in overview) == 4
        navigation = next(e.radio.id for e in overview if e.HasField("radio") and e.radio.label == "Workspace")
        comparison = await rerun(navigation)
        assert any(e.HasField("heading") and e.heading.body == "Model Comparison" for e in comparison)
        assert sum(e.HasField("plotly_chart") for e in comparison) == 2
        explorer = await rerun(navigation, "Forecast Explorer")
        assert any(e.HasField("heading") and e.heading.body == "Forecast Explorer" for e in explorer)
        figures = [json.loads(e.plotly_chart.spec) for e in explorer if e.HasField("plotly_chart")]
        assert len(figures) == 2
        actual = next(trace for trace in figures[0]["data"] if trace["name"] == "Actual recorded starts")
        assert len(actual["x"]) == len(set(actual["x"])) == 24
        assert any(trace.get("fill") == "tonexty" for trace in figures[0]["data"])
        assert theme_verified
        return {"overview_rendered": True, "model_comparison_rendered": True,
                "forecast_explorer_rendered": True, "saved_forecast_points": 24,
                "tft_band_rendered": True, "plotly_charts_per_comparison_workspace": 2,
                "native_light_dark_theme_config_verified": True}


def main():
    started = time.monotonic()
    # Reserve a local-only free port; never disturb an existing application.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "SYSTEMROOT") if key in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="citypulse-studio-smoke-") as scratch:
        log = Path(scratch) / "streamlit.log"
        with log.open("w") as stream:
            process = subprocess.Popen([sys.executable, "-m", "streamlit", "run",
                "apps/decision_studio/streamlit_app.py", "--server.address=127.0.0.1",
                f"--server.port={port}", "--server.headless=true", "--server.fileWatcherType=none",
                "--browser.gatherUsageStats=false"], cwd=ROOT, env=env, stdout=stream, stderr=stream)
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Owned Streamlit process exited before startup")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                        assert response.status == 200
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Streamlit startup deadline exceeded")
                    time.sleep(0.2)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
                assert response.status == 200
            checks = asyncio.run(render(port))
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
        print(json.dumps({"http_health": 200, "http_root": 200, **checks,
                          "owned_process_stopped": process.poll() is not None,
                          "server_peak_rss_bytes": rss_bytes, "seconds": time.monotonic() - started}, indent=2))


if __name__ == "__main__":
    main()
