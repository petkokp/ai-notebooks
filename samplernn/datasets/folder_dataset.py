from model.utilities import q_zero, linear_quantize

import torch
from torch.utils.data import (
    Dataset
)

from librosa.core import load
from natsort import natsorted

from os import listdir
from os.path import join


class FolderDataset(Dataset):

    def __init__(self, path, overlap_len, q_levels, ratio_min=0, ratio_max=1):
        super().__init__()
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        file_names = natsorted(
            [join(path, file_name) for file_name in listdir(path)]
        )
        self.file_names = file_names[
            int(ratio_min * len(file_names)): int(ratio_max * len(file_names))
        ]

    def __getitem__(self, index):
        (seq, _) = load(self.file_names[index], sr=None, mono=True)
        return torch.cat([
            torch.LongTensor(self.overlap_len)
                 .fill_(q_zero(self.q_levels)),
            linear_quantize(
                torch.from_numpy(seq), self.q_levels
            )
        ])

    def __len__(self):
        return len(self.file_names)
