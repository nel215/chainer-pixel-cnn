from chainer import functions as F


def down_shift(x):
    x = F.pad(x, [(0, 0), (0, 0), (1, 0), (0, 0)], 'constant')
    return x[:, :, :-1, :]


def right_shift(x):
    x = F.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)], 'constant')
    return x[:, :, :, :-1]
