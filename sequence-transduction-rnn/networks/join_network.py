import torch
import torch.nn as nn
from torch import Tensor

class JoinNet(nn.Module):
    MODES = {
        'multiplicative': lambda f, g: f * g,
        'mul': lambda f, g: f * g,
        'additive': lambda f, g: f + g,
        'add': lambda f, g: f + g
    }

    def __init__(
            self,
            input_size: int,
            vocab_size: int,
            mode: str
            ) -> None:
        super().__init__()
        self.join_mood = self.MODES[mode]
        self.fc = nn.Linear(
            in_features=input_size,
            out_features=vocab_size
        )

    def forward(self, f: Tensor, g: Tensor) -> Tensor:
        out = self.join_mood(f, g)
        out = self.fc(out)
        return torch.softmax(out, dim=-1)