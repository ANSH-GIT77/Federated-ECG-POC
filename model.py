import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet1D(nn.Module):
    """
    1D-CNN designed for multi-channel EEG signal classification.
    Adapted from the ECGNet but handles multiple channels and binary classification.

    Input shape : (batch, in_channels, length)
    Output shape: (batch, 2)
    """

    def __init__(self, in_channels: int = 20, num_classes: int = 2, input_length: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate flattened size
        # L_out = (L_in + 2*p - k)/s + 1
        # After 2 iterations of Pool(2): 1024 -> 512 -> 256
        self.flatten_size = 64 * (input_length // 4)
        
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        x = self.pool(F.relu(self.conv1(x)))   # → (batch, 32, length/2)
        x = self.pool(F.relu(self.conv2(x)))   # → (batch, 64, length/4)
        x = x.view(x.size(0), -1)              # → Flatten
        x = self.dropout(F.relu(self.fc1(x)))   # → (batch, 128)
        x = self.fc2(x)                         # → (batch, num_classes)
        return x

def get_model(in_channels=20, num_classes=2, input_length=1024):
    return EEGNet1D(in_channels=in_channels, num_classes=num_classes, input_length=input_length)
