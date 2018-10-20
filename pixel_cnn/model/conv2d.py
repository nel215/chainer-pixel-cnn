from chainer import Chain
from chainer import links as L
from chainer import functions as F


class Deconvolution2D(L.Deconvolution2D):

    def __call__(self, x, outsize):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return F.deconvolution_2d(
            x, self.W, self.b, self.stride, self.pad,
            groups=self.groups, outsize=outsize)


class DownRightShiftedConv2D(Chain):

    def __init__(self, n_out, ksize=None, stride=1):
        ksize = (2, 2) if ksize is None else ksize
        super(DownRightShiftedConv2D, self).__init__()
        self.pad = [
            (0, 0), (0, 0), (ksize[0]-1, 0), (ksize[1]-1, 0),
        ]
        with self.init_scope():
            self.conv2d = L.Convolution2D(
                None, n_out, ksize=ksize, stride=stride)

    def __call__(self, x):
        h = F.pad(x, self.pad, 'constant')
        h = self.conv2d(h)
        return h


class DownShiftedConv2D(Chain):

    def __init__(self, n_out, ksize=None, stride=1):
        super(DownShiftedConv2D, self).__init__()
        ksize = (2, 3) if ksize is None else ksize
        self.pad = [
            (0, 0), (0, 0),
            (ksize[0]-1, 0), ((ksize[1]-1)//2, (ksize[1]-1)//2),
        ]
        with self.init_scope():
            self.conv2d = L.Convolution2D(None, n_out, ksize=ksize, stride=stride)

    def __call__(self, x):
        h = F.pad(x, self.pad, 'constant')
        h = self.conv2d(h)
        return h


class DownShiftedDeconv2D(Chain):

    def __init__(self, n_out, ksize=None, stride=1):
        super(DownShiftedDeconv2D, self).__init__()
        ksize = (2, 3) if ksize is None else ksize
        self.pad = [
            (0, 0), (0, 0),
            (ksize[0]-1, 0), ((ksize[1]-1)//2, (ksize[1]-1)//2),
        ]
        self.stride = stride
        self.ksize = ksize
        with self.init_scope():
            self.deconv2d = Deconvolution2D(
                None, n_out, ksize=ksize, stride=stride)

    def __call__(self, x):
        h, w = x.shape[2:]
        outsize = (
            h*self.stride+self.ksize[0]-1, w*self.stride+self.ksize[1]-1)
        z = self.deconv2d(x, outsize)
        s1 = slice(0, -self.ksize[0]+1)
        s2 = slice((self.ksize[1]-1)//2, -(self.ksize[1]-1)//2)
        return z[:, :, s1, s2]


class DownRightShiftedDeconv2D(Chain):

    def __init__(self, n_out, ksize=None, stride=1):
        super(DownRightShiftedDeconv2D, self).__init__()
        ksize = (2, 2) if ksize is None else ksize
        self.ksize = ksize
        self.stride = stride
        with self.init_scope():
            self.deconv2d = Deconvolution2D(
                None, n_out, ksize=ksize, stride=stride)

    def __call__(self, x):
        h, w = x.shape[2:]
        outsize = (
            h*self.stride+self.ksize[0]-1, w*self.stride+self.ksize[1]-1)
        z = self.deconv2d(x, outsize)
        s1 = slice(0, -self.ksize[0]+1)
        s2 = slice(0, -self.ksize[1]+1)
        return z[:, :, s1, s2]
