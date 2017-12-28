from layers import *
import numpy as np

"""
CONVOLUTIONAL NEURAL NETWORK LAYERS
"""


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Wrapper layer around Convolution, ReLU, and MaxPool
    :param x: (num_samples, num_channels, height, width) np array of samples
    :param w: (num_filters, num_channels, f_height, f_width) np array of filters
    :param b: (num_filters,) np array of biases
    :param conv_param: refer to conv_forward pass
    :param pool_param: refer to pool_forward pass
    :return: output of wrapper layer forward pass, cache of layers
    """
    cout, conv_cache = conv_forward(x, w, b, conv_param)
    rout, relu_cache = relu_forward(cout)
    out, pool_cache = max_pool_forward(rout, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Computes gradients of wrapper layer around Convolution, ReLU, and MaxPool
    :param dout: gradients on output of conv_relu_pool_forward
    :param cache: cache from conv_relu_pool_forward
    """
    conv_cache, relu_cache, pool_cache = cache
    dp = max_pool_backward(dout, pool_cache)
    dr = relu_backward(dp, relu_cache)
    dx, dw, db = conv_backward(dr, conv_cache)
    return dx, dw, db


def conv_forward(x, w, b, conv_param):
    """
    Niave forward pass for a convolutional layer
    :param x: (num_samples, num_channels, height, width) np array of samples
    :param w: (num_filters, num_channels, f_height, f_width) np array of filters
    :param b: (num_filters) np array of biases
    :param conv_param: dict with conv config
        stride: int pixels between each field
        pad: number of pixels to pad input with
    :return: (num_samples, num_filters, new_height, new_width) output data, cache tuple of intermediates
    """
    # Unpack variables, calculate new sizes, pad samples
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Hnew = 1 + (H + 2 * pad - HH) // stride
    Wnew = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, Hnew, Wnew))
    x = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # Convolve each filter on every image
    for f_i, (fil, bias) in enumerate(zip(w, b)):
        for s_i, sample in enumerate(x):
            forward_convolve_sample(fil, sample, bias, out, stride, f_i, s_i)

    cache = (x, w, b, conv_param)
    return out, cache


def forward_convolve_sample(fil, sample, bias, out, stride, fil_i, sample_i):
    """
    Apply filter to one sample
    :param fil: (num_channels, f_width, f_height) np array filter
    :param sample: (num_channels, height, width) np array sample
    :param bias: float bias for filter
    :param out: (num_samples, num_filters, new_height, new_width) output data
    :param stride: int pixels between each field
    :param fil_i: filter number
    :param sample_i: sample_number
    """
    C, HH, WW = fil.shape
    C, H, W = sample.shape
    for map_i, i in enumerate(range(0, H-HH+1, stride)):
        for map_j, j in enumerate(range(0, W-WW+1, stride)):
            prod = sample[:, i:i+HH, j:j+WW] * fil
            prodsum = np.sum(prod)
            out[sample_i, fil_i, map_i, map_j] = prodsum + bias


def conv_backward(dout, cache):
    """
    Naive backward pass for convolutional layer
    :param dout: gradients on output of convolutional forward pass
    :param cache: tuple of intermediate variables of forward pass
    :return: grads dict dx, dw, db
    """
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)
    for f_i, fil in enumerate(w):
        for s_i, sample in enumerate(x):
            backward_convolve_sample(fil, sample, dout, dx, dw, db, stride, f_i, s_i)

    dx = dx[:, :, pad:-pad, pad:-pad]
    return dx, dw, db


def backward_convolve_sample(fil, sample, dout, dx, dw, db, stride, fil_i, sample_i):
    """
    Compute gradients for sample from one filter
    :param fil: (num_channels, f_width, f_height) np array filter
    :param sample: (num_channels, height, width) np array sample
    :param dout: (num_samples, num_filters, new_height, new_width) gradients on output data
    :param dx: (num_samples, num_channels, height, width) np array of gradients on x
    :param dw: (num_filters, num_channels, f_height, f_width) np array of gradients on w
    :param db: (num_filters,) np array of biases
    :param stride: int pixels between each field
    :param fil_i: filter number
    :param sample_i: sample_number
    """
    C, HH, WW = fil.shape
    C, H, W = sample.shape
    for map_i, i in enumerate(range(0, H-HH+1, stride)):
        for map_j, j in enumerate(range(0, W-WW+1, stride)):
            douti = dout[sample_i, fil_i, map_i, map_j]
            dx[sample_i, :, i:i+HH, j:j+WW] += fil * douti
            dw[fil_i] += sample[:, i:i+HH, j:j+WW] * douti
            db[fil_i] += douti


def max_pool_forward(x, pool_param):
    """
    Naive max pooling forward pass
    :param x: (num_samples, num_channels, height, width) np array of samples
    :param pool_param: dict with config
        pool_height: int pooling region height
        pool_width: int pooling region width
        stride: int pixels between pooling regions
    :return: (num_samples, num_channels, new_height, new_width) np array of pooled samples, cache of intermediates
    """
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W = x.shape
    Hnew, Wnew = ((H - pool_height) // stride) + 1, ((W - pool_width) // stride) + 1
    out = np.zeros((N, C, Hnew, Wnew))
    switches = np.zeros_like(out)
    for sample_i, sample in enumerate(x):
        forward_pool_sample(sample, sample_i, out, switches, pool_height, pool_width, stride)
    cache = (x, pool_param, switches)

    return out, cache


def forward_pool_sample(sample, sample_i, pool, switches, pool_height, pool_width, stride):
    """
    Apply pooling to one sample
    :param sample: (num_channels, height, width) np array sample
    :param sample_i: sample number
    :param pool: np array of pooled samples
    :param switches: cache of max indices for each pooling region
    :param pool_height: int pooling region height
    :param pool_width: int pooling region width
    :param stride: int pixels between pooling regions
    """
    C, H, W = sample.shape
    for pi, i in enumerate(range(0, H-pool_height+1, stride)):
        for pj, j in enumerate(range(0, W-pool_width+1, stride)):
            for k in range(C):
                switch = np.argmax(sample[k, i:i+pool_height, j:j+pool_width])
                h, w = flat_to_matrix_index(switch, pool_height, pool_width)
                pool[sample_i, k, pi, pj] = sample[k, i+h, j+w]
                switches[sample_i, k, pi, pj] = switch


def flat_to_matrix_index(i, pool_height, pool_width):
    """
    Converts flattened region index to 2d region indices
    :param i: int flattened index
    :param pool_height: int pooling region height
    :param pool_width: int pooling region width
    :return: h, w 2d indices
    """
    h = i // pool_height
    w = i % pool_width
    return int(h), int(w)


def max_pool_backward(dout, cache):
    """
    Naive max pooling backward pass
    :param dout: Gradients on max pooling forward pass output
    :param cache: cache from forward pass
    :return: gradients on x
    """
    x, pool_param, switches = cache
    dx = np.zeros_like(x)
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    for sample_i, sample in enumerate(x):
        backward_pool_sample(sample, sample_i, dout, dx, switches, pool_height, pool_width, stride)

    return dx


def backward_pool_sample(sample, sample_i, dpool, dx, switches, pool_height, pool_width, stride):
    """
    Compute gradients pooling one sample
    :param sample: (num_channels, height, width) np array sample
    :param sample_i: int sample number
    :param dpool: gradients on output from forward pool
    :param dx: gradients on x
    :param switches: max index for each pooling region for samples in x
    :param pool_height: pooling region height
    :param pool_width: pooling region width
    :param stride: int pixel distance between regions
    """
    C, H, W = sample.shape
    for pi, i in enumerate(range(0, H-pool_height+1, stride)):
        for pj, j in enumerate(range(0, W-pool_width+1, stride)):
            for k in range(C):
                switch = switches[sample_i, k, pi, pj]
                h, w = flat_to_matrix_index(switch, pool_height, pool_width)
                dx[sample_i, k, i+h, j+w] += dpool[sample_i, k, pi, pj]


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Apply spatial batch normalization to data along N * H * W axis
    :param x: (N, C, H, W) np array of data
    :param gamma: (C) np array to scale batch normalized data
    :param beta: (C) np array to shift batch normalized data
    :param bn_param: dict with config
        mode: 'test' or 'train'
        eps: numeric stability constant
        momentum: Scaling constant for old information for running mean
        running_mean: (D) np array of running mean of features
        running_var: (D) np array of running var of features
    :return: (N, C, H, W) np array of normalized data and cache of intermediates
    """
    N, C, H, W = x.shape
    x = x.swapaxes(0, 1).reshape(C, -1).T
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.T.reshape((C, N, H, W)).swapaxes(0, 1)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Apply backward pass of spatial batch normalization to compute gradients
    :param dout: gradients on spatial_batchnorm_forward output
    :param cache: cache from spatial_batchnorm_forward
    :return: dx, dgamma, dbeta of the same shape as their corresponding data
    """
    N, C, H, W = dout.shape
    dout = dout.swapaxes(0, 1).reshape((C, -1)).T
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.T.reshape((C, N, H, W)).swapaxes(0, 1)

    return dx, dgamma, dbeta


