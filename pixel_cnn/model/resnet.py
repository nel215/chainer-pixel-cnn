from chainer import Chain
from chainer import links as L
from chainer import functions as F


def concat_elu(x):
    return F.elu(F.concat([x, -x], 1))


class GatedResnet(Chain):

    def __init__(self, n_out):
        super(GatedResnet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_out, ksize=3, pad=1)
            self.conv2 = L.Convolution2D(None, n_out, ksize=1)
            self.conv3 = L.Convolution2D(None, 2*n_out, ksize=3, pad=1)

    def __call__(self, x, a=None):
        h = self.conv1(concat_elu(x))
        if a is not None:
            h += self.conv2(concat_elu(a))

        h = F.dropout(concat_elu(h))
        h = self.conv3(h)

        # TODO: conditional generation

        a, b = F.split_axis(h, 2, 1)
        h = a * F.sigmoid(b)
        return x + h
