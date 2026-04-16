"""
_sim_server.py — Internal: Flower server for the simulation mode
================================================================
Launched by fl_simulation.py — adapted for EEG Seizure Detection.
"""

import os
import flwr as fl
import torch
from torch.utils.tensorboard import SummaryWriter

from model import EEGNet1D
from dataset import get_eeg_data, get_dataloader
from utils import get_parameters, set_parameters, evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 3 # PN00, PN01, PN05
NUM_ROUNDS = 3
SERVER_ADDRESS = "127.0.0.1:8099"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "patient_records")
META_FILE = os.path.join(SCRIPT_DIR, "epilipsysample.txt")
LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs")

tb_writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, "global_server"))

def get_evaluate_fn():
    # Use a small subset for global evaluation to save memory/time
    testset = get_eeg_data(DATA_ROOT, META_FILE, max_files=1)
    testloader = get_dataloader(testset, batch_size=32, shuffle=False)

    def evaluate_fn(server_round, parameters, config):
        model = EEGNet1D().to(DEVICE)
        set_parameters(model, parameters)
        loss, accuracy = evaluate(model, testloader, device=DEVICE)

        # Log to TensorBoard
        tb_writer.add_scalar("global/accuracy", accuracy, server_round)
        tb_writer.add_scalar("global/loss", loss, server_round)
        tb_writer.flush()

        print(
            f"\n{'='*55}\n"
            f"  [Round {server_round}/{NUM_ROUNDS}] "
            f"Global Accuracy: {accuracy:.4f}  |  Loss: {loss:.4f}\n"
            f"{'='*55}",
            flush=True,
        )
        return loss, {"accuracy": accuracy}
    return evaluate_fn

def weighted_average(metrics):
    accs = [n * m["accuracy"] for n, m in metrics]
    total = [n for n, _ in metrics]
    if sum(total) == 0:
        avg = 0
    else:
        avg = sum(accs) / sum(total)

    tb_writer.add_scalar("aggregated_clients/accuracy", avg, weighted_average._round)
    tb_writer.flush()
    weighted_average._round += 1

    return {"accuracy": avg}

weighted_average._round = 1

if __name__ == "__main__":
    print(f"  [Server] TensorBoard logs -> tb_logs/global_server/", flush=True)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(),
        initial_parameters=fl.common.ndarrays_to_parameters(
            get_parameters(EEGNet1D())
        ),
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    tb_writer.close()
