import torch
from torch import nn

# Create the DQN Model
class Blackjack_DQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.qnet = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, x):
        return self.qnet(x)