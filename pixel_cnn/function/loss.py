from chainer import functions as F
from chainer.backends.cuda import get_array_module


def mixture_of_discretized_logistics_nll(x, y):
    """
    Args:
        x: (b, c, n, n)
        y: (b, 10*n_mix, n, n)
    """
    xp = get_array_module(x)
    n_mix = y.shape[1] // 10
    logit_prob = y[:, :n_mix, :, :]
    y = F.reshape(y[:, n_mix:, :, :], x.shape + (n_mix*3, ))
    mean = y[:, :, :, :, 0:n_mix]
    log_scale = y[:, :, :, :, n_mix:2*n_mix]
    log_scale = F.maximum(log_scale, -7 * xp.ones(log_scale.shape, dtype='f'))
    coeff = F.tanh(y[:, :, :, :, 2*n_mix:3*n_mix])

    x = xp.repeat(xp.expand_dims(x, 4), n_mix, 4)
    m1 = F.expand_dims(mean[:, 0, :, :, :], 1)
    m2 = F.expand_dims(
        mean[:, 1, :, :, :] + coeff[:, 0, :, :, :] * x[:, 0, :, :, :], 1)
    m3 = F.expand_dims((
        mean[:, 2, :, :, :] +
        coeff[:, 1, :, :, :] * x[:, 0, :, :, :] +
        coeff[:, 2, :, :, :] * x[:, 1, :, :, :]
    ), 1)
    mean = F.concat([m1, m2, m3])
    centered_x = x - mean
    inv_std = F.exp(-log_scale)
    max_in = inv_std * (centered_x + 1./255.)
    cdf_max = F.sigmoid(max_in)
    min_in = inv_std * (centered_x - 1./255.)
    cdf_min = F.sigmoid(min_in)
    log_cdf_max = max_in - F.softplus(max_in)  # 0
    log_one_minus_cdf_min = -F.softplus(min_in)  # 255
    cdf_delta = cdf_max - cdf_min  # 0 ~ 255
    mid_in = inv_std * centered_x
    log_pdf_mid = mid_in - log_scale - 2. * F.softplus(mid_in)  # mid

    log_prob = F.where(
        x < -0.999,
        log_cdf_max,
        F.where(
            x > 0.999,
            log_one_minus_cdf_min,
            F.where(
                cdf_delta.array > 1e-5,
                F.log(F.maximum(
                    cdf_delta, xp.ones(cdf_delta.shape, dtype='f') * 1e-12)
                ),
                log_pdf_mid - xp.log(127.5)
            )
        )
    )

    log_prob = F.transpose(F.sum(log_prob, 1), (0, 3, 1, 2))
    log_prob = log_prob + log_prob_from_logit(logit_prob)

    loss = F.logsumexp(log_prob, 1)
    loss = F.sum(loss, axis=(1, 2))
    return -F.mean(loss)


def log_prob_from_logit(x):
    c = x.shape[1]
    m = F.max(x, 1, keepdims=True)
    b = m + F.log(F.sum(F.exp(x-F.repeat(m, c, 1)), 1, keepdims=True))
    return x - F.repeat(b, c, 1)
