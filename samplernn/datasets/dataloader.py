from torch.utils.data import (
    DataLoader as DataLoaderBase
)


class DataLoader(DataLoaderBase):

    def __init__(self, dataset, batch_size, seq_len, overlap_len,
                 *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len

    def __iter__(self):
        for batch in super().__iter__():
            (_, n_samples) = batch.size()

            reset = True

            for seq_begin in range(self.overlap_len, n_samples, self.seq_len):
                from_index = seq_begin - self.overlap_len
                to_index = seq_begin + self.seq_len
                sequences = batch[:, from_index: to_index]
                input_sequences = sequences[:, : -1]
                target_sequences = sequences[:, self.overlap_len:].contiguous()

                yield (input_sequences, reset, target_sequences)

                reset = False

    def __len__(self):
        raise NotImplementedError()
