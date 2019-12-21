"""Some commonly-used layer blocks.
"""

import torch

from torch import nn


def conv2d(
  in_channels,
  out_channels,
  kernel_size=3,
  stride=1,
  dilation=1,
  bias=True,
):
  """`same` convolution, i.e. output shape equals input shape.
  Args:
    in_planes (int): The number of input feature maps.
    out_planes (int): The number of output feature maps.
    kernel_size (int): The filter size.
    dilation (int): The filter dilation factor.
    stride (int): The filter stride.
  """
  # compute new filter size after dilation
  # and necessary padding for `same` output size
  dilated_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
  same_padding = (dilated_kernel_size - 1) // 2

  return nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=same_padding,
    dilation=dilation,
    bias=bias,
  )


class Flatten(nn.Module):
  """Flattens convolutional feature maps for fc layers.
  """
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)


class Identity(nn.Module):
  """An identity layer.
  """
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x


class CausalConv1D(nn.Conv1d):
  """A causal 1D convolution.
  """
  def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    bias=True
  ):
    self.__padding = (kernel_size - 1) * dilation

    super().__init__(
      in_channels,
      out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=self.__padding,
      dilation=dilation,
      bias=bias,
    )

  def forward(self, x):
    res = super().forward(x)
    if self.__padding != 0:
      return res[:, :, :-self.__padding]
    return res


class ResidualBlock(nn.Module):
  """A simple residual block.
  """
  def __init__(self, channels):
    super().__init__()

    self.conv1 = conv2d(channels, channels, bias=False)
    self.conv2 = conv2d(channels, channels, bias=False)
    self.bn1 = nn.BatchNorm2d(channels)
    self.bn2 = nn.BatchNorm2d(channels)
    self.act = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.act(x)
    out = self.act(self.bn1(self.conv1(out)))
    out = self.bn2(self.conv2(out))
    return out + x


class UpsamplingBlock(nn.Module):
  """An upsampling block.
  """
  def __init__(
    self,
    in_channels,
    out_channels,
    mode,
    dropout=None,
    norm=nn.BatchNorm2d,
  )
    """Constructor.

    Args:
      in_channels (int): The number of input channels.
      out_channels (int): The number of output channels.
      mode (str): What type of upsampling to use. Can be one of:
        `bilinear`: Bilinear interpolation followed by a convolution.
        `nearest`: Nearest neighbor interpolation followed by a convolution.
        `transpose`: Transpose convolution.
      dropout (float): The dropout probability. Set to `None` to disable dropout.
      norm (obj): The type of norm to use. Set to `None` to disable normalization.
    """
    super().__init__()



  def forward(self, x):
    pass