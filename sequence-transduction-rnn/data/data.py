import math
import pandas as pd
import torch
import torchaudio
from torch import Tensor
from pathlib import Path
from typing import List, Union, Tuple
from tokenizer import BaseTokenizer
from torchaudio.transforms import (
    Resample,
    MelSpectrogram
)


def load_audio(file_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    x, sr = torchaudio.load(file_path, normalize=True)
    return x, sr


class AudioPipeline():
    sample_rate = 32000

    def __init__(self) -> None:
        super().__init__()

    def run(
            self,
            audio_path: Union[str, Path],
            *args,
            **kwargs
    ) -> Tensor:
        x, sr = load_audio(audio_path)
        x = self._get_resampler(sr)(x)
        x = self._get_mel_spec_transformer()(x)
        x = x.permute(0, 2, 1)
        return x

    def _get_resampler(self, sr: int):
        return Resample(sr, self.sample_rate)

    def _get_mel_spec_transformer(self):
        n_mel_channels = 80
        return MelSpectrogram(
            self.sample_rate,
            n_mels=n_mel_channels
        )


class TextPipeline():
    def __init__(self) -> None:
        super().__init__()

    def run(
            self,
            text: str
    ) -> str:
        text = text.lower()
        return text.strip()


class BaseData:
    def __init__(
            self,
            text_pipeline: TextPipeline,
            audio_pipeline: AudioPipeline,
            tokenizer: BaseTokenizer,
            max_len: int
    ) -> None:
        self.text_pipeline = text_pipeline
        self.audio_pipeline = audio_pipeline
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _get_padded_aud(
            self,
            aud_path: Union[str, Path],
            max_duration: int,
    ) -> Tensor:
        hop_length = 50
        max_len = 1 + math.ceil(
            max_duration * self.sampling_rate / hop_length
        )
        aud = self.audio_pipeline.run(aud_path)
        assert aud.shape[0] == 1, f'expected audio of 1 channels got \
            {aud_path} with {aud.shape[0]} channels'
        return self.pad_mels(aud, max_len)

    def _get_padded_tokens(self, text: str) -> Tensor:
        text = self.text_pipeline.run(text)
        tokens = self.tokenizer.tokens2ids(text)
        tokens.append(self.tokenizer.special_tokens.eos_id)
        tokens = self.pad_tokens(tokens)
        return torch.LongTensor(tokens)

    def pad_mels(self, mels: Tensor, max_len: int) -> Tensor:
        n = max_len - mels.shape[1]
        zeros = torch.zeros(size=(1, n, mels.shape[-1]))
        return torch.cat([zeros, mels], dim=1)

    def pad_tokens(self, tokens: list) -> Tensor:
        length = self.max_len - len(tokens)
        return tokens + [self.tokenizer.special_tokens.pad_id] * length


class DataLoader(BaseData):
    def __init__(
            self,
            file_path: Union[str, Path],
            text_pipeline: TextPipeline,
            audio_pipeline: AudioPipeline,
            tokenizer: BaseTokenizer,
            batch_size: int,
            max_len: int,
            file_text: str,
            file_duration: float
    ) -> None:
        super().__init__(text_pipeline, audio_pipeline,
                         tokenizer, max_len, file_text, file_duration)
        self.batch_size = batch_size
        self.file_path = file_path
        self.file_text = file_text
        self.file_duration = file_duration
        self.df = pd.read_csv(file_path)
        self.num_examples = len(self.df)
        self.idx = 0

    def __len__(self):
        length = self.num_examples // self.batch_size
        mod = self.num_examples % self.batch_size
        return length + 1 if mod > 0 else length

    def get_max_duration(self, start_idx: int, end_idx: int) -> float:
        return self.df[
            self.file_duration
        ].iloc[start_idx: end_idx].max()

    def get_audios(self, start_idx: int, end_idx: int) -> Tensor:
        max_duration = self.get_max_duration(start_idx, end_idx)
        result = list(map(
            self._get_padded_aud,
            self.df[self.file_path].iloc[start_idx: end_idx],
            [max_duration] * (end_idx - start_idx)
        ))
        result = torch.stack(result, dim=1)
        return torch.squeeze(result)

    def get_texts(self, start_idx: int, end_idx: int) -> Tensor:
        args = self.df[self.file_text].iloc[start_idx: end_idx]
        lengths = list(map(lambda x: len(x) + 1, args.values))
        result = list(map(self._get_padded_tokens, args))
        result = torch.stack(result, dim=0)
        return result, torch.IntTensor(lengths)

    def __iter__(self):
        self.idx = 0
        while True:
            start = self.idx * self.batch_size
            end = (self.idx + 1) * self.batch_size
            end = min(end, self.num_examples)
            if start > self.num_examples or start == end:
                break
            self.idx += 1
            yield (
                self.get_audios(start, end),
                *self.get_texts(start, end)
            )
