import os
import numpy as np
import random
import torch


def set_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def categorical_kl(p_nk: torch.Tensor, q_nk: torch.Tensor):
    ratio_nk = p_nk / (q_nk + 1e-6)
    ratio_nk[p_nk == 0] = 1
    ratio_nk[(q_nk == 0) & (p_nk != 0)] = np.inf
    return (p_nk * torch.log(ratio_nk)).sum(dim=1)


def get_flat_params_from(model: torch.nn.Module):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params_to(params: torch.nn.Module.parameters, model: torch.nn.Module):
    pointer = 0
    for p in model.parameters():
        p.data.copy_(params[pointer:pointer + p.data.numel()].view_as(p.data))
        pointer += p.data.numel()
