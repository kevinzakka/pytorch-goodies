import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """An abstract class representing a model architecture.

    Any model definition should subclass `BaseModel`.
    """
    def __init__(self):
        super().__init__()

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters())

    def forward(self, x):
        raise NotImplementedError
