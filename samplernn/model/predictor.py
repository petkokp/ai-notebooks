import torch
from model.utilities import linear_dequantize
from model.runner import Runner


class Predictor(Runner, torch.nn.Module):

    def __init__(self, model):
        super().__init__(model)

    def forward(self, input_sequences, reset):
        if reset:
            self.reset_hidden_states()

        (batch_size, _) = input_sequences.size()

        upper_tier_conditioning = None
        for rnn in reversed(self.model.frame_level_rnns):
            from_index = self.model.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1
            prev_samples = 2 * linear_dequantize(
                input_sequences[:, from_index: to_index],
                self.model.q_levels
            )
            prev_samples = prev_samples.contiguous().view(
                batch_size, -1, rnn.n_frame_samples
            )

            upper_tier_conditioning = self.run_rnn(
                rnn, prev_samples, upper_tier_conditioning
            )

        bottom_frame_size = self.model.frame_level_rnns[0].frame_size
        mlp_input_sequences = input_sequences[:,
                                              self.model.lookback - bottom_frame_size:]

        return self.model.sample_level_mlp(
            mlp_input_sequences, upper_tier_conditioning
        )
