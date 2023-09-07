import torch.nn as nn
import math

def sequence_nll_loss_bits(input, target, *args, **kwargs):
    (_, _, n_classes) = input.size()
    return nn.functional.nll_loss(
        input.view(-1, n_classes), target.view(-1), *args, **kwargs
    ) * math.log(math.e, 2)
