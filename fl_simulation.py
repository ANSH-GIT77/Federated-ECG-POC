"""
fl_simulation.py — Federated Learning Simulation Coordinator
============================================================
Launches:
1. TensorBoard (Background)
2. Flower Server (New Window)
3. 3 Patient Clients (3 New Windows)
"""

import subprocess
import sys
import time
import os
import shutil

NUM_CLIENTS = 3
PYTHON = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TB_LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs")

def main():
    print()
    print("=" * 60)
    print("  Federated Learning POC — EEG Multi-Window Simulation")
    print("=" * 60)
    print()

    # ── 0. Clean old TensorBoard logs ─────────────────────────────
    if os.path.exists(TB_LOG_DIR):
        try:
            shutil.rmtree(TB_LOG_DIR)
        except Exception as e:
            print(f"  ⚠️ Warning: Could not clean tb_logs: {e}")
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    # ── 0b. Launch TensorBoard in the background ──────────────────
    print("  📊 [TensorBoard] Starting on http://localhost:6006 ...")
    subprocess.Popen(
        [PYTHON, "-m", "tensorboard.main",
         "--logdir", TB_LOG_DIR,
         "--port", "6006",
         "--bind_all"],
        cwd=SCRIPT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("     Keep this tab open or visit http://localhost:6006 in your browser.\n")

    # ── 1. Start the simulation server in a new window ────────────
    print("  🚀 Starting Flower server in a new terminal...")
    server_script = os.path.join(SCRIPT_DIR, "_sim_server.py")
    subprocess.Popen(
        [PYTHON, server_script],
        cwd=SCRIPT_DIR,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    # Give the server time to bind the port
    time.sleep(5)

    # ── 2. Start all clients in their own windows ──────────────────
    client_script = os.path.join(SCRIPT_DIR, "_sim_client.py")
    
    for cid in range(NUM_CLIENTS):
        print(f"  📡 Launching Patient Node {cid} in a new terminal...")
        subprocess.Popen(
            [PYTHON, client_script, str(cid)],
            cwd=SCRIPT_DIR,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        time.sleep(1)

    print("\n  ✅ All windows launched!")
    print("  Monitor training plots at: http://localhost:6006")
    print()

if __name__ == "__main__":
    main()
