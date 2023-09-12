from __future__ import unicode_literals, print_function, division
from io import open

import torch
import torch.utils.data
import torch.nn as nn

SOS_token = 1
EOS_token = 2


def sentence_from_tensor(lang, tensor):
    indexes = tensor.squeeze()
    indexes = indexes.tolist()
    return [lang.index2word[index] for index in indexes]


def indices_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indices = indices_from_sentence(lang, sentence)
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long).view(1, -1)


def tensors_from_pair(pair, input_lang, output_lang, max_input_length):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])

    with torch.no_grad():
        pad_input = nn.ConstantPad1d(
            (0, max_input_length - input_tensor.shape[1]), 0)
        pad_target = nn.ConstantPad1d(
            (0, max_input_length - target_tensor.shape[1]), 0)

        input_tensor_padded = pad_input(input_tensor)
        target_tensor_padded = pad_target(target_tensor)

    from torch.nn.utils.rnn import pad_sequence
    pair_tensor = pad_sequence(
        [input_tensor_padded, target_tensor_padded], batch_first=False, padding_value=0)

    return pair_tensor


def reformat_tensor(tensor):
    tensor = tensor.transpose(0, 2, 1)
    tensor = tensor.squeeze()
    return tensor[tensor != -1].view(-1, 1)


def reformat_tensor_mask(tensor):
    tensor = tensor.squeeze(dim=1)
    tensor = tensor.transpose(1, 0)
    mask = tensor != 0
    return tensor, mask
