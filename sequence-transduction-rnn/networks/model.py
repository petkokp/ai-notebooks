import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from predict_network import PredictNet
from transcription_network import TranscriptionNet
from join_network import JoinNet

class Model(nn.Module):
    def __init__(
            self,
            prednet_params: dict,
            transnet_params: dict,
            joinnet_params: dict,
            phi_idx: int,
            pad_idx: int,
            sos_idx: int,
            device='cuda'
            ) -> None:
        super().__init__()
        self.prednet = PredictNet(**prednet_params).to(device)
        self.transnet = TranscriptionNet(**transnet_params).to(device)
        self.joinnet = JoinNet(**joinnet_params).to(device)
        self.prednet_hidden_size = prednet_params['hidden_size']
        self.device = device
        self.phi_idx = phi_idx
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx

    def forward(
            self,
            x: Tensor,
            max_length: int
            ) -> Tensor:
        batch_size, T, *_ = x.shape
        counter = self.get_counter_start(batch_size, T)
        counter_ceil = self.get_counter_ceil(counter, T)
        term_state = torch.zeros(batch_size)
        trans_result = self.feed_into_transnet(x)
        trans_result = trans_result.reshape(batch_size * T, -1)
        h, c = self.prednet.get_zeros_hidden_state(batch_size)
        h = h.to(self.device)
        c = c.to(self.device)
        gu = self.get_sos_seed(batch_size)
        t = 0
        while True:
            t += 1
            preds, h, c = self.predict_next(gu, h, c, counter, trans_result)
            if t == 1:
                result = preds
            else:
                result = torch.concat([result, preds], dim=1)
            gu = self.keep_last_char(gu, torch.argmax(preds, dim=-1))
            counter, update_mask, term_state = self.update_states(
                gu, counter, counter_ceil, term_state, t
                )
            if (update_mask.sum().item() == batch_size) or (max_length == t):
                break
        return result, term_state

    def keep_last_char(self, gu: Tensor, preds: Tensor) -> Tensor:
        is_phi = preds == self.phi_idx
        return (is_phi * gu) + (~is_phi * preds)

    def update_states(
            self,
            gu: Tensor,
            counter: Tensor,
            counter_ceil: Tensor,
            term_state: Tensor,
            t: int
            ) -> Tuple[Tensor, Tensor, Tensor]:
        counter = counter + ((gu.cpu() == self.phi_idx).squeeze())
        counter, update_mask = self.clip_counter(counter, counter_ceil)
        term_state = self.update_termination_state(
            term_state, update_mask, t
            )
        return counter, update_mask, term_state

    def predict_next(
            self,
            gu: Tensor,
            h: Tensor,
            c: Tensor,
            counter: Tensor,
            trans_result: Tensor
            ) -> Tuple[Tensor, Tensor, Tensor]:
        out, h, c = self.prednet(gu, h, c)
        fy = trans_result[counter, :].unsqueeze(dim=1)
        preds = self.joinnet(fy, out)
        return preds, h, c

    def get_counter_ceil(
            self, counter: Tensor, T: int
            ) -> Tensor:
        return counter + T - 1

    def get_sos_seed(
            self, batch_size: int
            ) -> Tensor:
        return torch.LongTensor([[self.sos_idx]] * batch_size).to(self.device)

    def feed_into_transnet(self, x: Tensor) -> Tensor:
        return self.transnet(x)

    def feed_into_prednet(
            self, yu: Tensor, h: Tensor, c: Tensor
            ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.transnet(yu, h, c)

    def get_counter_start(
            self, batch_size: int, max_size: int
            ) -> Tensor:
        return torch.arange(0, batch_size * max_size, max_size)

    def clip_counter(
            self, counter: Tensor, ceil_vector: Tensor
            ) -> Tuple[Tensor, Tensor]:
        update_mask = counter >= ceil_vector
        upper_bounded = update_mask * ceil_vector
        kept_counts = (counter < ceil_vector) * counter
        return upper_bounded + kept_counts, update_mask

    def update_termination_state(
            self,
            term_state: Tensor,
            update_mask: Tensor,
            last_index: int
            ) -> Tensor:
        is_unended = term_state == 0
        to_update = is_unended & update_mask
        return term_state + to_update * last_index