"""Classic DQN CNN: one linear Q value per discrete action, no softmax."""
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, actions: int = 7):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, actions),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # Replay stores uint8 on CPU; convert only the batch being used.
        return self.layers(pixels.float() / 255.0)
