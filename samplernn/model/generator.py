import torch
from utilities import q_zero, linear_dequantize
from runner import Runner


class Generator(Runner):

    def __init__(self, model, cuda=False):
        super().__init__(model)
        self.cuda = cuda

    def __call__(self, n_seqs, seq_len):
        torch.backends.cudnn.enabled = False

        self.reset_hidden_states()

        bottom_frame_size = self.model.frame_level_rnns[0].n_frame_samples
        sequences = torch.LongTensor(n_seqs, self.model.lookback + seq_len) \
                         .fill_(q_zero(self.model.q_levels))
        frame_level_outputs = [None for _ in self.model.frame_level_rnns]

        for i in range(self.model.lookback, self.model.lookback + seq_len):
            for (tier_index, rnn) in \
                    reversed(list(enumerate(self.model.frame_level_rnns))):
                if i % rnn.n_frame_samples != 0:
                    continue

                prev_samples = torch.autograd.Variable(
                    2 * linear_dequantize(
                        sequences[:, i - rnn.n_frame_samples: i],
                        self.model.q_levels
                    ).unsqueeze(1),
                    volatile=True
                )
                if self.cuda:
                    prev_samples = prev_samples.cuda()

                if tier_index == len(self.model.frame_level_rnns) - 1:
                    upper_tier_conditioning = None
                else:
                    frame_index = (i // rnn.n_frame_samples) % \
                        self.model.frame_level_rnns[tier_index + 1].frame_size
                    upper_tier_conditioning = \
                        frame_level_outputs[tier_index + 1][:, frame_index, :] \
                        .unsqueeze(1)

                frame_level_outputs[tier_index] = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning
                )

            prev_samples = torch.autograd.Variable(
                sequences[:, i - bottom_frame_size: i],
                volatile=True
            )
            if self.cuda:
                prev_samples = prev_samples.cuda()
            upper_tier_conditioning = \
                frame_level_outputs[0][:, i % bottom_frame_size, :] \
                .unsqueeze(1)
            sample_dist = self.model.sample_level_mlp(
                prev_samples, upper_tier_conditioning
            ).squeeze(1).exp_().data
            sequences[:, i] = sample_dist.multinomial(1).squeeze(1)

        torch.backends.cudnn.enabled = True

        return sequences[:, self.model.lookback:]
