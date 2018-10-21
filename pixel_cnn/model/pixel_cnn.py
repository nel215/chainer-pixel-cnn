from chainer import links as L
from chainer import functions as F
from chainer import Chain, ChainList
from .conv2d import (
    DownShiftedConv2D, DownRightShiftedConv2D,
    DownShiftedDeconv2D, DownRightShiftedDeconv2D,
)
from .resnet import GatedResnet
from pixel_cnn.function.shift import down_shift, right_shift
from pixel_cnn.function.loss import mixture_of_discretized_logistics_nll


class GatedResnetList(ChainList):
    def __init__(self, n, n_out, Conv2D):
        super(GatedResnetList, self).__init__()
        with self.init_scope():
            for i in range(n):
                self.add_link(GatedResnet(n_out, Conv2D))


class Initializer(Chain):

    def __init__(self, n_out):
        super(Initializer, self).__init__()
        with self.init_scope():
            self.ds_conv1 = DownShiftedConv2D(n_out)
            self.ds_conv2 = DownShiftedConv2D(n_out, (1, 3))
            self.drs_conv1 = DownRightShiftedConv2D(n_out, (2, 1))

    def __call__(self, x):
        u = down_shift(self.ds_conv1(x))
        ul = down_shift(self.ds_conv2(x)) + right_shift(self.drs_conv1(x))
        return u, ul


class PixelCNN(Chain):

    def __init__(self, n_mix=10):
        super(PixelCNN, self).__init__()
        n_out = 128
        n = 2
        with self.init_scope():
            self.initializer = Initializer(n_out)
            self.u_res_list1 = GatedResnetList(n, n_out, DownShiftedConv2D)
            self.ul_res_list1 = GatedResnetList(
                n, n_out, DownRightShiftedConv2D)

            self.ds_conv1 = DownShiftedConv2D(n_out, stride=2)
            self.drs_conv1 = DownRightShiftedConv2D(n_out, stride=2)

            self.u_res_list2 = GatedResnetList(n, n_out, DownShiftedConv2D)
            self.ul_res_list2 = GatedResnetList(
                n, n_out, DownRightShiftedConv2D)

            # down pass
            self.u_res_list3 = GatedResnetList(n, n_out, DownShiftedConv2D)
            self.ul_res_list3 = GatedResnetList(
                n, n_out, DownRightShiftedConv2D)

            self.ds_deconv1 = DownShiftedDeconv2D(n_out, stride=2)
            self.drs_deconv1 = DownRightShiftedDeconv2D(n_out, stride=2)

            self.u_res_list4 = GatedResnetList(n+1, n_out, DownShiftedConv2D)
            self.ul_res_list4 = GatedResnetList(
                n+1, n_out, DownRightShiftedConv2D)

            self.last_conv = L.Convolution2D(None, 10*n_mix, ksize=1)

    def __call__(self, x):
        """
        Args:
            x: (b, c, h, w)
        """
        # up pass
        u, ul = self.initializer(x)
        u_list, ul_list = [u], [ul]

        for u_res, ul_res in zip(self.u_res_list1, self.ul_res_list1):
            u_list.append(u_res(u_list[-1]))
            ul_list.append(ul_res(ul_list[-1], u_list[-1]))

        u_list.append(self.ds_conv1(u_list[-1]))
        ul_list.append(self.drs_conv1(ul_list[-1]))

        for u_res, ul_res in zip(self.u_res_list2, self.ul_res_list2):
            u_list.append(u_res(u_list[-1]))
            ul_list.append(ul_res(ul_list[-1], u_list[-1]))

        # down pass
        u = u_list.pop()
        ul = ul_list.pop()

        for u_res, ul_res in zip(self.u_res_list3, self.ul_res_list3):
            u = u_res(u, u_list.pop())
            ul = ul_res(ul, F.concat([u, ul_list.pop()], 1))

        u = self.ds_deconv1(u)
        ul = self.drs_deconv1(ul)

        for u_res, ul_res in zip(self.u_res_list4, self.ul_res_list4):
            u = u_res(u, u_list.pop())
            ul = ul_res(ul, F.concat([u, ul_list.pop()], 1))

        assert len(u_list) == 0
        assert len(ul_list) == 0

        y = self.last_conv(F.elu(ul))

        loss = mixture_of_discretized_logistics_nll(x, y)
        return loss
