#!/usr/bin/env python3
"""Launch the Streamlit dashboard: python scripts/dashboard.py"""
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from src.utils.config import load_config


def main() -> None:
    cfg = load_config("configs/mvp.yaml")
    app = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"
    port = str(cfg.dashboard.get_path("port", 8501))
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.port", port, "--server.address", "0.0.0.0"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
