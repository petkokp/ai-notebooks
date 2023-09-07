import torch.nn as nn
import math

def lecun_uniform(tensor):
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    nn.init.uniform(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
