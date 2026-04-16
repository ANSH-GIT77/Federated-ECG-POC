# 🏥 Federated Learning POC — Wearable ECG System

A Proof of Concept for **Federated Learning** using the **Flower (flwr)** framework
and **PyTorch**, simulating a network of wearable health devices performing
collaborative model training without sharing raw patient data.

---

## 📁 Project Structure

```
DML TA2/
├── model.py            # 1D-CNN architecture (ECG signal classifier)
├── dataset.py          # MNIST loading & partitioning into device shards
├── utils.py            # Training, evaluation, parameter helpers, comparison table
├── fl_simulation.py    # Mode 1: Simulate 5 virtual devices locally
├── fl_server.py        # Mode 2: Real-device server (discovery + OTP + Flower)
├── fl_client.py        # Mode 2: Real-device client (runs on each device)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Mode 1 — Simulation (single machine)

Run the full simulation with 5 virtual wearable devices:

```bash
python fl_simulation.py
```

This will:
- Simulate 5 virtual devices training a 1D-CNN on MNIST
- Run 3 rounds of FedAvg
- Print **Global Accuracy** after each round
- Display a Centralized vs Federated comparison table

### 3. Mode 2 — Real Devices (same Wi-Fi network)

#### On your laptop (server):

```bash
python fl_server.py
```

The server will display:
- Your **local IP address**
- A **4-digit OTP code** — share with device operators
- A live **device connection menu**

#### On each participating device:

```bash
python fl_client.py --server <SERVER_IP>
```

- Enter the OTP when prompted (3 attempts allowed)
- The client auto-downloads MNIST and starts training

#### Running multiple clients on the same machine:

Open separate terminals and assign different partitions:

```bash
python fl_client.py --server 192.168.1.42 --partition 0
python fl_client.py --server 192.168.1.42 --partition 1
python fl_client.py --server 192.168.1.42 --partition 2
```

---

## 📱 Using iPhones / Mobile Devices

Running PyTorch + Flower directly on iOS is not straightforward due to library
constraints. Here are practical alternatives for a **live demo**:

| Approach | How |
|---|---|
| **Laptop as proxy** | Run `fl_client.py` on multiple laptops/desktops on the same Wi-Fi |
| **Pyto (iOS)** | Install [Pyto](https://apps.apple.com/app/pyto/id1436650069), `pip install flwr torch torchvision` inside the app's terminal, then run `fl_client.py` |
| **SSH from iPhone** | SSH into a machine on the network and run the client there |
| **Termux (Android)** | Full Python + PyTorch support via `pkg install python` |

> **For the best demo experience**, run 3-5 terminal windows on different
> machines (or the same machine) each running `fl_client.py` with different
> `--partition` values. The server will see them as separate devices.

---

## 🔐 OTP Flow

1. Server generates a random 4-digit OTP and displays it in the terminal
2. Each client prompts the user to enter this OTP
3. Server validates the OTP before allowing registration
4. Invalid OTP → rejected with an error (max 3 attempts on client)

---

## ⚙️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Device 1    │     │  Device 2    │     │  Device N    │
│  (fl_client) │     │  (fl_client) │     │  (fl_client) │
│  Local Data  │     │  Local Data  │     │  Local Data  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │ OTP + Register     │                    │
       ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│                    fl_server.py                         │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │ Discovery   │   │  FedAvg      │   │  Evaluation  │  │
│  │ (HTTP+OTP)  │   │  Aggregation │   │  (Global)    │  │
│  └─────────────┘   └──────────────┘   └─────────────┘  │
└─────────────────────────────────────────────────────────┘
```
# dmlta2krishna
