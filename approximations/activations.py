import torch

def custom_activation(x):
    return torch.nn.functional.gelu(x)