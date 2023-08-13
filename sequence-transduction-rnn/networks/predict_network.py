from typing import Tuple
from torch import Tensor
import torch
import torch.nn as nn


class PredictNet(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            emb_dim: int,
            pad_idx: int,
            hidden_size: int,
            n_layers: int,
            dropout: float
            ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(
            vocab_size, emb_dim, padding_idx=pad_idx
            )
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(
            self,
            x: Tensor,
            hn: Tensor,
            cn: Tensor
            ) -> Tuple[Tensor, Tensor, Tensor]:
        self.validate_dims(hn, cn)
        out = self.emb(x)
        out, (hn, cn) = self.lstm(out, (hn, cn))
        return out, hn, cn

    def get_zeros_hidden_state(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        return (
            torch.zeros((self.n_layers, batch_size, self.hidden_size)),
            torch.zeros((self.n_layers, batch_size, self.hidden_size))
        )

    def validate_dims(
            self,
            hn: Tensor,
            cn: Tensor
            ) -> None:
        assert hn.shape[0] == self.n_layers, \
            'The hidden state should match the number of layers'
        assert hn.shape[2] == self.hidden_size, \
            'The hidden state should match the hiden size'
        assert cn.shape[0] == self.n_layers, \
            'The cell state should match the number of layers'
        assert cn.shape[2] == self.hidden_size, \
            'The cell state should match the hiden size'