import torch
import pytest

from models import MnistConvNet


def test_num_params():
    net = MnistConvNet()
    print("# of params: {:,}".format(net.num_params))

