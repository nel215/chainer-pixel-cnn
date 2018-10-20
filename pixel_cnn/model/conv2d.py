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


class DownShiftedDeconv2D(Chain):

    def __init__(self, n_out, ksize):
        super(DownShiftedDeconv2D, self).__init__()
        self.ksize = ksize
        with self.init_scope():
            self.deconv2d = L.Deconvolution2D(None, n_out, ksize=ksize)

    def __call__(self, x):
        h = self.deconv2d(x)
        s1 = slice(0, -self.ksize[0]+1)
        s2 = slice((self.ksize[1]-1)//2, -(self.ksize[1]-1)//2)
        return h[:, :, s1, s2]


class DownRightShiftedDeconv2D(Chain):

    def __init__(self, n_out, ksize):
        super(DownRightShiftedDeconv2D, self).__init__()
        self.ksize = ksize
        with self.init_scope():
            self.deconv2d = L.Deconvolution2D(None, n_out, ksize=ksize)

    def __call__(self, x):
        h = self.deconv2d(x)
        s1 = slice(0, -self.ksize[0]+1)
        s2 = slice(0, -self.ksize[1]+1)
        return h[:, :, s1, s2]
