"""
utils.py — Shared Utilities for Federated Learning POC
======================================================
Model parameter helpers, training / evaluation loops, and a
presentation-ready comparison table.
"""

from collections import OrderedDict

import torch
from model import EEGNet1D

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


# ── Parameter Serialisation ──────────────────────────────────────────

def get_parameters(model: torch.nn.Module) -> list:
    """Return model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: list) -> None:
    """Load a list of NumPy arrays into the model's state_dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# ── Training & Evaluation ────────────────────────────────────────────

def train(model, trainloader, epochs: int = 1, device: str = "cpu",
          tb_writer=None, tb_round: int = 0, tb_tag: str = "client"):
    """
    Run local training for the given number of epochs.

    If `tb_writer` (a SummaryWriter) is provided, logs:
      • per-batch training loss
      • per-epoch training accuracy
    """
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    global_step_offset = tb_round * epochs * len(trainloader)

    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # ── TensorBoard: batch-level loss ──
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if tb_writer is not None:
                step = global_step_offset + epoch * len(trainloader) + batch_idx
                tb_writer.add_scalar(f"{tb_tag}/batch_loss", loss.item(), step)

        # ── TensorBoard: epoch-level accuracy ──
        epoch_acc = correct / total if total else 0
        epoch_loss = running_loss / len(trainloader)
        if tb_writer is not None:
            tb_writer.add_scalar(f"{tb_tag}/epoch_accuracy", epoch_acc, tb_round * epochs + epoch)
            tb_writer.add_scalar(f"{tb_tag}/epoch_loss", epoch_loss, tb_round * epochs + epoch)
            tb_writer.flush()


def evaluate(model, testloader, device: str = "cpu"):
    """
    Evaluate model accuracy and loss on the given DataLoader.

    Returns
    -------
    avg_loss : float
    accuracy : float   (0-1)
    """
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / len(testloader)
    return avg_loss, accuracy


# ── Presentation Comparison Table ────────────────────────────────────

def print_comparison_table():
    """Print a mock Centralized-vs-Federated comparison for the demo."""
    W = 70
    print()
    print("=" * W)
    print("📊  PERFORMANCE COMPARISON: Centralized vs Federated Learning")
    print("=" * W)
    print(f"{'Metric':<28} {'Centralized':>18} {'Federated (FL)':>18}")
    print("-" * W)

    rows = [
        ("Accuracy",           "97.2 %",         "95.8 %"),
        ("Training Time",      "45 min",          "12 min / device"),
        ("Data Privacy",       "❌ None",          "✅ Preserved"),
        ("Communication",      "All raw data",    "Model params only"),
        ("Single-Point Failure","❌ Yes",          "✅ No"),
        ("Scalability",        "Limited",         "✅ High"),
        ("Data Freshness",     "Batch upload",    "Real-time on-device"),
    ]
    for name, cent, fed in rows:
        print(f"  {name:<26} {cent:>18} {fed:>18}")

    print("=" * W)
    print("💡  Federated Learning keeps data on-device while achieving")
    print("    competitive accuracy — ideal for wearable / ECG pipelines.")
    print()
