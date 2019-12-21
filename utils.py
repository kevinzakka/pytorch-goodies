import torch


def nanmean(x):
  """Computes the arithmetic mean ignoring any NaNs.
  """
  return torch.mean(x[x == x])


def num_params(model):
  """Computes the number of parameters in a model.
  """
  return sum(param.numel() for param in model.parameters())