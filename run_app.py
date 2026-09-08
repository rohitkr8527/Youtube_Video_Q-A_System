from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def main() -> None:
    load_dotenv(ROOT / ".env")
    host = os.getenv("API_HOST", "127.0.0.1")
    api_port = os.getenv("API_PORT", "8000")
    ui_port = os.getenv("STREAMLIT_PORT", "8501")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", host, "--port", api_port],
        cwd=ROOT,
        env=env,
    )

    time.sleep(1.2)

    ui = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend/app.py",
            "--server.port",
            ui_port,
            "--server.headless",
            "true",
        ],
        cwd=ROOT,
        env=env,
    )

    processes = [api, ui]

    def shutdown(*_: object) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    raise SystemExit(code)
            time.sleep(0.5)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
