"""
run_asisten.py - Peluncur Utama Asisten-AI Autonomous Intelligence
Menjalankan Dashboard Streamlit (dan opsi FastAPI server) secara otomatis.
"""
import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
STREAMLIT_EXE = BASE_DIR / ".venv" / "Scripts" / "streamlit.exe"


def main():
    parser = argparse.ArgumentParser(description="Peluncur Asisten-AI Autonomous Intelligence")
    parser.add_argument("--mode", choices=["dashboard", "api", "both"], default="dashboard", help="Mode aplikasi")
    parser.add_argument("--port", type=int, default=8501, help="Port untuk dashboard Streamlit (default: 8501)")
    parser.add_argument("--api-port", type=int, default=8000, help="Port untuk FastAPI backend (default: 8000)")
    args = parser.parse_args()

    python_exec = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    streamlit_exec = str(STREAMLIT_EXE) if STREAMLIT_EXE.exists() else "streamlit"

    if args.mode == "dashboard":
        print(f"[*] Menjalankan Asisten-AI Streamlit Dashboard pada port {args.port}...")
        cmd = [
            streamlit_exec,
            "run",
            str(BASE_DIR / "asisten_ai" / "dashboard.py"),
            "--server.port",
            str(args.port),
            "--server.headless",
            "false",
        ]
        subprocess.run(cmd)

    elif args.mode == "api":
        print(f"[*] Menjalankan Asisten-AI FastAPI Backend pada port {args.api_port}...")
        cmd = [
            python_exec,
            "-m",
            "uvicorn",
            "asisten_ai.app_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.api_port),
            "--reload",
        ]
        subprocess.run(cmd)

    elif args.mode == "both":
        print("[*] Menjalankan FastAPI Backend di latar belakang...")
        api_proc = subprocess.Popen([
            python_exec,
            "-m",
            "uvicorn",
            "asisten_ai.app_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.api_port),
        ])
        try:
            print(f"[*] Menjalankan Dashboard Streamlit pada port {args.port}...")
            cmd = [
                streamlit_exec,
                "run",
                str(BASE_DIR / "asisten_ai" / "dashboard.py"),
                "--server.port",
                str(args.port),
            ]
            subprocess.run(cmd)
        finally:
            api_proc.terminate()


if __name__ == "__main__":
    main()
