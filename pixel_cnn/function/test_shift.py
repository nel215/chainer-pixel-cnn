import numpy as np
from .shift import down_shift, right_shift


def test_down_shift():
    b, c, n = 4, 8, 16
    x = np.ones((b, c, n, n))
    h = down_shift(x)
    assert np.all(h[:, :, 0, :].array == 0)
    assert np.all(h[:, :, 1:, :].array == 1)
    assert h.shape == x.shape


def test_right_shift():
    b, c, n = 4, 8, 16
    x = np.ones((b, c, n, n))
    h = right_shift(x)
    assert np.all(h[:, :, :, 0].array == 0)
    assert np.all(h[:, :, :, 1:].array == 1)
    assert h.shape == x.shape
