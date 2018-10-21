import numpy as np
from .pixel_cnn import PixelCNN


def test_call():
    b = 1
    n = 16
    c = 3
    model = PixelCNN()
    x = np.zeros((b, c, n, n), dtype='f')
    loss = model(x)
    assert loss.shape == ()
