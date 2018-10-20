import numpy as np
from .resnet import GatedResnet
from .conv2d import DownShiftedConv2D, DownShiftedDeconv2D


def test_call_with_conv2d():
    b = 4
    n = 16
    c = 8
    model = GatedResnet(c, DownShiftedConv2D)
    x = np.zeros((b, c, n, n), dtype='f')
    a = np.zeros((b, c, n, n), dtype='f')
    h = model(x, a)
    assert h.shape == (b, c, n, n)


def test_call_with_deconv2d():
    b = 4
    n = 16
    c = 8
    model = GatedResnet(c, DownShiftedDeconv2D)
    x = np.zeros((b, c, n, n), dtype='f')
    a = np.zeros((b, c, n, n), dtype='f')
    h = model(x, a)
    assert h.shape == (b, c, n, n)
