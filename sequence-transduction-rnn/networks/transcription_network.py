import torch
import torch.nn as nn
from torch import Tensor

class TranscriptionNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            n_layers: int,
            dropout: float,
            is_bidirectional: bool
            ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=is_bidirectional
        )

    def forward(self, x: Tensor) -> Tensor:
        out, *_ = self.lstm(x)
        return out