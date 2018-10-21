import numpy as np
from .loss import mixture_of_discretized_logistics_nll


def test_mixture_of_discretized_logistics_nll():
    b = 4
    c = 3
    h = 10*10
    n = 16
    x = np.zeros((b, c, n, n), dtype='f')
    y = np.zeros((b, h, n, n), dtype='f')
    loss = mixture_of_discretized_logistics_nll(x, y)
    assert loss.shape == ()
