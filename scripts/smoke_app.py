"""Asset-aware deployment smoke check for the disaster-tweets app."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from disaster_tweets_nlp import config, model  # noqa: E402


def check_model() -> None:
    pipeline = model.load_pipeline()
    row = pd.DataFrame(
        [{config.TEXT_COL: "Wildfire evacuation ordered", config.KEYWORD_COL: "wildfire", config.LOCATION_COL: "California"}]
    )
    if len(pipeline.predict(row)) != 1:
        raise RuntimeError("The packaged model did not return one prediction.")


def check_streamlit() -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless=true", "--server.port=8501"],
        cwd=PROJECT_ROOT,
    )
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming healthy.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(1)
        raise RuntimeError("Streamlit health endpoint did not become ready within 90 seconds.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    check_model()
    check_streamlit()
    print("disaster-tweets-nlp deployment smoke check passed")
