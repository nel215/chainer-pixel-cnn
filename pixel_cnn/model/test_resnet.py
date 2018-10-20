import numpy as np
from .resnet import GatedResnet


def test_call():
    b = 4
    n = 16
    c = 8
    model = GatedResnet(c)
    x = np.zeros((b, c, n, n), dtype='f')
    a = np.zeros((b, c, n, n), dtype='f')
    h = model(x, a)
    assert h.shape == (b, c, n, n)
