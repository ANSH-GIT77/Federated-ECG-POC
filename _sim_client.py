"""
_sim_client.py — Internal: Flower client for the simulation mode
================================================================
Launched by fl_simulation.py — adapted for EEG Seizure Detection on specific patients.
"""

import sys
import os
import flwr as fl
import torch
from torch.utils.tensorboard import SummaryWriter

from model import EEGNet1D
from dataset import get_eeg_data, get_dataloader
from utils import get_parameters, set_parameters, train, evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SERVER_ADDRESS = "127.0.0.1:8099"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "patient_records")
META_FILE = os.path.join(SCRIPT_DIR, "epilipsysample.txt")
LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs")

# Map index to actual patient IDs from the dataset
PATIENTS = ["PN00", "PN01", "PN05"]

class SimulatedClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = cid
        self.patient_id = PATIENTS[cid] if cid < len(PATIENTS) else PATIENTS[0]
        
        self.model = EEGNet1D().to(DEVICE)
        
        # Load local data for this specific patient
        self.trainset = get_eeg_data(DATA_ROOT, META_FILE, patient_id=self.patient_id)
        self.trainloader = get_dataloader(self.trainset, batch_size=32)
        
        # In this EEG POC, evaluate on the same local data or a separate test set if available.
        # For simplicity, we use the local data for eval.
        self.testloader = get_dataloader(self.trainset, batch_size=32, shuffle=False)
        self.current_round = 0

        # TensorBoard writer for this device
        tag = f"patient_{self.patient_id}"
        self.writer = SummaryWriter(
            log_dir=os.path.join(LOG_DIR, tag)
        )
        print(f"    [{tag}] TensorBoard logs → tb_logs/{tag}/", flush=True)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"    [Patient {self.patient_id}] Round {self.current_round + 1}: "
              f"training on {len(self.trainset)} samples …", flush=True)

        train(
            self.model, self.trainloader, epochs=1, device=DEVICE,
            tb_writer=self.writer,
            tb_round=self.current_round,
            tb_tag=f"patient_{self.patient_id}",
        )

        # Also log local eval metrics after training
        loss, acc = evaluate(self.model, self.testloader, device=DEVICE)
        self.writer.add_scalar(f"patient_{self.patient_id}/eval_accuracy", acc, self.current_round)
        self.writer.add_scalar(f"patient_{self.patient_id}/eval_loss", loss, self.current_round)
        self.writer.flush()

        self.current_round += 1
        return get_parameters(self.model), len(self.trainset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.trainloader, device=DEVICE)
        return float(loss), len(self.trainset), {"accuracy": float(acc)}

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.close()

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    client = SimulatedClient(idx)
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )
