"""
fl_client.py — Federated Learning Client for Real Devices
==========================================================
Run this on every participating device.
Adapted for EEG Seizure Detection.
"""

import argparse
import json
import socket
import uuid
import urllib.request
import os

import flwr as fl
import torch

from model import EEGNet1D
from dataset import get_eeg_data, get_dataloader
from utils import get_parameters, set_parameters, train, evaluate

# ── Config ────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISCOVERY_PORT = 5000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "patient_records")
META_FILE = os.path.join(SCRIPT_DIR, "epilipsysample.txt")

# ── Helpers ───────────────────────────────────────────────────────────

def get_device_name() -> str:
    return socket.gethostname()

def register_with_server(server_ip: str, otp: str, discovery_port: int):
    device_id   = str(uuid.uuid4())[:8]
    device_name = get_device_name()

    url = f"http://{server_ip}:{discovery_port}/register"
    payload = json.dumps({
        "device_id":   device_id,
        "device_name": device_name,
        "otp":         otp,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if "error" in data:
                return None, data["error"]
            return data, None
    except Exception as e:
        return None, str(e)

# ── Flower Client ────────────────────────────────────────────────────

class RealDeviceClient(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader):
        self.model = EEGNet1D().to(DEVICE)
        self.trainloader = trainloader
        self.testloader  = testloader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print("  🏋️  Training locally …")
        train(self.model, self.trainloader, epochs=1, device=DEVICE)
        print("  ✅  Local training complete.")
        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = evaluate(self.model, self.testloader, device=DEVICE)
        print(f"  📈  Local eval — accuracy: {accuracy:.4f}  loss: {loss:.4f}")
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FL Client — Wearable EEG Device")
    parser.add_argument("--server", type=str, required=True,
                        help="Server IP address")
    parser.add_argument("--patient_id", type=str, default="PN00",
                        help="Patient ID (e.g. PN00, PN01, PN05)")
    parser.add_argument("--discovery-port", type=int, default=5000,
                        help="Server discovery port (default 5000)")
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("📱  Federated Learning Client — EEG Wearable")
    print("=" * 55)
    print(f"  🏷️   Patient ID   : {args.patient_id}")
    print(f"  📡  Server       : {args.server}")
    print()

    # ── OTP Authentication ────────────────────────────────────────
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        otp = input(f"  🔐  Enter OTP (attempt {attempt}/{max_attempts}): ").strip()
        print("  ⏳  Registering with server …")
        result, error = register_with_server(args.server, otp, args.discovery_port)
        if error:
            print(f"  ❌  Registration failed: {error}")
            if attempt == max_attempts: return
        else:
            break

    flower_server = result["flower_server"]
    print(f"  ✅  Registered!  Flower server → {flower_server}\n")

    # ── Load Dataset ──────────────────────────────────────────────
    print(f"  📦  Loading EEG data for {args.patient_id} …")
    dataset = get_eeg_data(DATA_ROOT, META_FILE, patient_id=args.patient_id)
    trainloader = get_dataloader(dataset, batch_size=32)
    # Use same data for local evaluation for simplicity in this POC
    testloader = get_dataloader(dataset, batch_size=32, shuffle=False)
    print(f"  📂  {len(dataset)} segments loaded.")
    print()

    # ── Connect to Flower Server ─────────────────────────────────
    print("  🌸  Connecting to Flower server …\n")
    fl.client.start_client(
        server_address=flower_server,
        client=RealDeviceClient(trainloader, testloader).to_client(),
    )

if __name__ == "__main__":
    main()
