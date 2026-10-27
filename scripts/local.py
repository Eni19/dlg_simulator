"""Install and run the complete local laboratory using only stdlib tooling."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def available_port(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("A porta deve estar entre 1 e 65535.")
    return port


def main():
    parser = argparse.ArgumentParser(description="Instale e execute o laboratorio local completo.")
    parser.add_argument("--setup", action="store_true", help="Instala as dependencias e encerra.")
    parser.add_argument("--check", action="store_true", help="Verifica as dependencias sem iniciar servidores.")
    parser.add_argument("--api-port", type=available_port, default=8000)
    parser.add_argument("--port", type=available_port, default=5173)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.exit(1, "Use Python 3.12: py -3.12 scripts/local.py\n")
    node = shutil.which("node")
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not node or not npm:
        parser.exit(1, "Instale Node.js 24 com npm e abra novamente o terminal.\n")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if int(version.lstrip("v").split(".")[0]) != 24:
        parser.exit(1, f"Use Node.js 24 (encontrado {version}).\n")
    if args.setup:
        if not PYTHON.exists():
            venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
        subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", "requirements-full.txt"], cwd=ROOT, check=True)
        subprocess.run([npm, "ci"], cwd=ROOT / "frontend", check=True)
        print("Instalacao concluida. Execute novamente sem --setup.", flush=True)
        return
    vite = ROOT / "frontend/node_modules/vite/bin/vite.js"
    if not PYTHON.exists() or not vite.exists():
        parser.exit(1, "Execute primeiro: py -3.12 scripts/local.py --setup\n")
    probe = subprocess.run([str(PYTHON), "-c", "import sys, torch, torchvision, fastapi, uvicorn, flwr, flwr_datasets; assert sys.version_info[:2] == (3, 12); print('Python:', sys.version.split()[0]); print('Flower:', flwr.__version__); print('PyTorch:', torch.__version__); print('CUDA disponivel:', torch.cuda.is_available())"], cwd=ROOT)
    if probe.returncode:
        parser.exit(1, "Dependencias incompletas. Execute novamente com --setup.\n")
    print(f"Node.js: {version}", flush=True)
    if args.check:
        return
    if os.environ.get("VERCEL") == "1":
        parser.exit(1, "O servidor completo e apenas local. VERCEL=1 preserva o showcase.\n")
    if args.port == args.api_port:
        parser.exit(1, "O frontend e a API precisam de portas diferentes.\n")
    for port in (args.api_port, args.port):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                parser.exit(1, f"Porta {port} ocupada. Use --api-port e --port para escolher outras.\n")
    env = dict(os.environ, DLG_APP_MODE="full", VITE_LAB_MODE="full", DLG_DEV_API_URL=f"http://127.0.0.1:{args.api_port}")
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    children = []
    try:
        children.append(subprocess.Popen([str(PYTHON), "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(args.api_port)], cwd=ROOT, env=env, **options))
        children.append(subprocess.Popen([node, str(vite), "--host", "127.0.0.1", "--port", str(args.port), "--strictPort"], cwd=ROOT / "frontend", env=env, **options))
        print(f"\nLaboratorio completo: http://127.0.0.1:{args.port}\nCtrl+C encerra os dois servidores.\n", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(0.25)
        raise RuntimeError("Um dos servidores encerrou. Confira a mensagem acima.")
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        sys.exit(1)
