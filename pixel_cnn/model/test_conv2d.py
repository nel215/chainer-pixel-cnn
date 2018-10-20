import numpy as np
from .conv2d import (
    DownShiftedConv2D, DownRightShiftedConv2D,
    DownShiftedDeconv2D, DownRightShiftedDeconv2D,
)


def test_down_call():
    b = 4
    n = 16
    n_out = 8
    model = DownShiftedConv2D(n_out, (2, 3))
    x = np.zeros((b, 3, n, n), dtype='f')
    h = model(x)
    assert h.shape == (b, n_out, n, n)


def test_down_right_call():
    b = 4
    n = 16
    n_out = 8
    model = DownRightShiftedConv2D(n_out, (2, 1))
    x = np.zeros((b, 3, n, n), dtype='f')
    h = model(x)
    assert h.shape == (b, n_out, n, n)


def test_down_deconv_call():
    b = 4
    n = 16
    n_out = 8
    model = DownShiftedDeconv2D(n_out, (2, 3))
    x = np.zeros((b, 16, n, n), dtype='f')
    h = model(x)
    assert h.shape == (b, n_out, n, n)


def test_down_right_deconv_call():
    b = 4
    n = 16
    n_out = 8
    model = DownRightShiftedDeconv2D(n_out, (2, 2))
    x = np.zeros((b, 16, n, n), dtype='f')
    h = model(x)
    assert h.shape == (b, n_out, n, n)
