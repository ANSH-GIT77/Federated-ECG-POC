"""
fl_server.py — Federated Learning Server with Device Discovery & OTP
=====================================================================
Adapted for EEG Seizure Detection.
"""

import argparse
import json
import random
import socket
import sys
import threading
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

import flwr as fl
import torch

from model import EEGNet1D
from dataset import get_eeg_data, get_dataloader
from utils import get_parameters, set_parameters, evaluate, print_comparison_table

# ── Config ────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ROUNDS = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "patient_records")
META_FILE = os.path.join(SCRIPT_DIR, "epilipsysample.txt")

# ── Global state ─────────────────────────────────────────────────────
registered_devices: dict = {}
otp_code: str = ""
lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────

def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ── HTTP Discovery Service ───────────────────────────────────────────

class DiscoveryHandler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        if self.path != "/register":
            self._reply(404, {"error": "not found"})
            return

        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        incoming_otp  = body.get("otp", "")
        device_name   = body.get("device_name", "Unknown")
        device_id     = body.get("device_id", "")

        if incoming_otp != otp_code:
            self._reply(401, {"error": "Invalid OTP"})
            return

        with lock:
            registered_devices[device_id] = {
                "name": device_name,
                "ip": self.client_address[0],
                "time": time.strftime("%H:%M:%S"),
                "selected": True,
            }

        local_ip = get_local_ip()
        flower_port = self.server.flower_port
        print(f"  ✅  Device registered: {device_name} ({self.client_address[0]})")
        self._reply(200, {
            "status": "registered",
            "flower_server": f"{local_ip}:{flower_port}",
        })

    def do_GET(self):
        if self.path == "/ping": self._reply(200, {"status": "ok"})
        else: self._reply(404, {"error": "not found"})

    def _reply(self, code: int, payload: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

def start_discovery_server(discovery_port: int, flower_port: int):
    server = HTTPServer(("0.0.0.0", discovery_port), DiscoveryHandler)
    server.flower_port = flower_port
    server.serve_forever()

# ── Interactive Device Menu ──────────────────────────────────────────

def device_selection_menu() -> int:
    while True:
        with lock: devs = dict(registered_devices)
        print("\n" + "-" * 55 + "\n📱  Connected Devices\n" + "-" * 55)
        if not devs: print("    (none yet)")
        else:
            for idx, (did, info) in enumerate(devs.items(), 1):
                sel = "✅" if info["selected"] else "  "
                print(f"  {sel} {idx}. {info['name']:<20} {info['ip']:<16} joined {info['time']}")
        print("-" * 55 + "\n  Commands: [Enter] Start, [t <n>] Toggle, [r] Refresh, [q] Quit\n" + "-" * 55)
        choice = input("  ▸ ").strip().lower()
        if choice == "" and devs:
            with lock: selected = sum(1 for d in registered_devices.values() if d["selected"])
            if selected > 0: return selected
        elif choice == "q": sys.exit(0)
        elif choice.startswith("t"):
            try:
                num = int(choice.split()[1])
                with lock:
                    keys = list(registered_devices.keys())
                    if 1 <= num <= len(keys): registered_devices[keys[num - 1]]["selected"] = not registered_devices[keys[num - 1]]["selected"]
            except Exception: pass

# ── Server-side Evaluation ───────────────────────────────────────────

def get_evaluate_fn(testloader):
    def evaluate_fn(server_round, parameters, config):
        model = EEGNet1D().to(DEVICE)
        set_parameters(model, parameters)
        loss, accuracy = evaluate(model, testloader, device=DEVICE)
        print(f"\n{'='*55}\n🌐  [Round {server_round}/{NUM_ROUNDS}] Global Accuracy: {accuracy:.4f}\n{'='*55}")
        return loss, {"accuracy": accuracy}
    return evaluate_fn

# ── Main ─────────────────────────────────────────────────────────────

def main():
    global otp_code
    parser = argparse.ArgumentParser(description="FL Server — Real EEG Device Mode")
    parser.add_argument("--discovery-port", type=int, default=5000)
    parser.add_argument("--flower-port", type=int, default=8080)
    args = parser.parse_args()

    otp_code = f"{random.randint(1000, 9999)}"
    local_ip = get_local_ip()

    print(f"\n" + "=" * 60 + "\n🏥  Federated Learning Server — EEG Mode\n" + "=" * 60)
    print(f"  📡  Server IP   : {local_ip}\n  🔐  OTP Code    : {otp_code}\n" + "=" * 60)

    threading.Thread(target=start_discovery_server, args=(args.discovery_port, args.flower_port), daemon=True).start()
    min_clients = device_selection_menu()

    # Evaluation on all data
    testset = get_eeg_data(DATA_ROOT, META_FILE)
    testloader = get_dataloader(testset, batch_size=32, shuffle=False)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=min_clients, min_evaluate_clients=min_clients, min_available_clients=min_clients,
        evaluate_fn=get_evaluate_fn(testloader),
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(EEGNet1D())),
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.flower_port}",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    print_comparison_table()

if __name__ == "__main__":
    main()
