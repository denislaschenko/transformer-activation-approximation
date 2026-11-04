import torch.nn as nn

def custom_softmax(x,dim):
    return nn.functional.softmax(x,dim=dim)