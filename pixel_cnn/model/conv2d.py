from chainer import Chain
from chainer import links as L
from chainer import functions as F


class DownRightShiftedConv2D(Chain):

    def __init__(self, n_out, ksize):
        super(DownRightShiftedConv2D, self).__init__()
        self.pad = [
            (0, 0), (0, 0), (ksize[0]-1, 0), (ksize[1]-1, 0),
        ]
        with self.init_scope():
            self.conv2d = L.Convolution2D(None, n_out, ksize=ksize)

    def __call__(self, x):
        h = F.pad(x, self.pad, 'constant')
        h = self.conv2d(h)
        return h


class DownShiftedConv2D(Chain):

    def __init__(self, n_out, ksize):
        super(DownShiftedConv2D, self).__init__()
        self.pad = [
            (0, 0), (0, 0),
            (ksize[0]-1, 0), ((ksize[1]-1)//2, (ksize[1]-1)//2),
        ]
        with self.init_scope():
            self.conv2d = L.Convolution2D(None, n_out, ksize=ksize)

    def __call__(self, x):
        h = F.pad(x, self.pad, 'constant')
        h = self.conv2d(h)
        return h
