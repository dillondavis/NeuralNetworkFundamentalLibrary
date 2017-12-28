import numpy as np

"""
REGULAR NEURAL NETWORK LAYERS
"""


def full_relu_forward(x, w, b):
    """
    Compute forward pass of a fully connected relu layer
    :param x: (N, f1....fn) np array of data
    :param w: (num_features, output_size) np array of weights
    :param b: (output_size) np array of biases
    :return: hidden_states of layer and cache
    """
    hidden, full_cache = full_forward(x, w, b)
    hidden, relu_cache = relu_forward(hidden)
    return hidden, (full_cache, relu_cache)


def full_relu_backward(dout, cache):
    """
    Compute backward pass/find gradients of a fully connected relu layer
    :param dout: gradient of output
    :param cache: cache of full and relu layers
    :return: gradients of x, w, b
    """
    full_cache, relu_cache = cache
    dhidden = relu_backward(dout, relu_cache)
    dx, dw, db = full_backward(dhidden, full_cache)
    return dx, dw, db


def full_forward(x, w, b):
    """
    Compute forward pass of fully connected layer
    :param x: (N, f1....fn) np array of data
    :param w: (num_features, output_size) np array of weights
    :param b: (output_size) np array of biases
    :return: output of forward pass and cache of layer
    """
    x_flat = np.reshape(x, (x.shape[0], -1))
    out = x_flat @ w + b
    cache = (x, w, b)

    return out, cache


def full_backward(dout, cache):
    """
    Compute backward pass/find gradients of a fully connected layer
    :param dout: gradient of output
    :param cache: cache of full and relu layers
    :return: gradients of x, w, b
    """
    x, w, b = cache
    dx = np.reshape(dout @ w.T, x.shape)
    dw = np.reshape(x, (x.shape[0], -1)).T @ dout
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Compute forward pass of fully connected layer
    :param x: (N, f1....fn) np array of data
    :param w: (num_features, output_size) np array of weights
    :param b: (output_size) np array of biases
    :return: output of forward pass and cache of layer
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Compute backward pass/find gradients of a fully connected layer
    :param dout: gradient of output
    :param cache: cache of full and relu layers
    :return: gradient of x
    """
    x = cache
    dx = dout
    dx[x <= 0] = 0
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Batch normalization forward pass
    Keep exponentially decaying running mean to normalize test time
    :param x: (num_samples, num_features) np array of samples
    :param gamma: (num_features) np array of scaling parameters
    :param beta: (num_features) np array of shifting parameters
    :param bn_param: dict with config parameters
        :mode = 'train' or 'test'
        :eps = epsilon numerical stability constant
        :momentum: scaling paramater for decaying mean
        :running_mean: (num_features) np array of running feature means
        :running_var: (num_features) np array of running feature variances
    :return:
        out: (num_samples, num_features) normalized data
        cache: cache of variables for backward pass
    """

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'test':
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        return out, None

    # Stepwise normalize data and update running variables
    N, D = x.shape

    mu = np.mean(x, axis=0)

    xdiff = x - mu

    xdiffsq = np.square(xdiff)

    var = np.sum(xdiffsq / N, axis=0)

    svar = np.sqrt(var + eps)

    ivar = 1 / svar

    xnorm = xdiff * ivar

    out = gamma * xnorm + beta
    cache = (gamma, xnorm, xdiff, ivar, svar, var, xdiffsq, mu, eps)
    running_mean = momentum * running_mean + (1 - momentum) * mu
    running_var = momentum * running_var + (1 - momentum) * var

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Compute gradients for batchnorm layer
    :param dout: Gradients on batchnorm forward output
    :param cache: Intermediate variables from forward pass
    :return:
        dx: (num_samples, num_features) np array of gradients for x
        dgamma: (num_samples,) np array of gradients for gamma
        dbeta: (num_samples,) np array of gradients for beta
    """
    gamma, xnorm, xdiff, ivar, svar, var, xdiffsq, mu, eps = cache
    N, D = dout.shape

    # Find gradient for each step in forward pass
    dgamma = np.sum(dout * xnorm, axis=0)
    dbeta = np.sum(dout, axis=0)
    dxnorm = dout * gamma

    dxdiff1 = dxnorm * ivar
    divar = np.sum(dxnorm * xdiff, axis=0)

    dsvar = divar * (-svar**(-2))

    dvar = dsvar * (0.5 * var**(-0.5))

    dxdiffsq = (np.ones_like(xdiffsq) / N) * dvar

    dxdiff2 = dxdiffsq * (2 * xdiff)
    dxdiff = dxdiff1 + dxdiff2

    dx1 = dxdiff
    dmu = -np.sum(dxdiff, axis=0)

    dx2 = (np.ones_like(xdiff) / N) * dmu
    dx = dx1 + dx2

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Forward pass for dropout layer
    Randomly drops neuron connetions
    :param x: (num_samples, num_features) np array of samples
    :param dropout_param: dict with dropout config
        p: float probability to drop a neuron
        mode: 'train' or 'test'
        seed: random number generator seed
    :return: (num_samples, num_features) of dropout data and cache of dropout intermediates
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    if mode == 'train':
        mask = (np.random.rand(*x.shape) > p) / p
        out = mask * x
    elif mode == 'test':
        out = x
        mask = None

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Backward pass for inverted dropout
    :param dout: gradients on dropout forward output
    :param cache: tuple of intermediate variables of forward pass
    :return: (num_samples, num_features) of gradients on x
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = mask * dout
    elif mode == 'test':
        dx = dout
    return dx


"""
LOSS FUNCTIONS
"""


def softmax_loss(scores, y):
    """
    Find softmax cross entropy loss from predicted scores and real
    :param scores: (num_samples, num_classes) np array of scores per sample
    :param y: (num_samples) np array of classes for samples
    :return: float softmax loss
    """
    num_samples = scores.shape[0]
    fstable = (scores.T - np.amax(scores, axis=1)).T
    fexp = np.exp(fstable)
    fprob = fexp / np.sum(fexp, axis=1, keepdims=True)
    floss = np.log(fprob[np.arange(num_samples), y] + 1e-14)
    loss = -(np.sum(floss) / num_samples)

    dScores = fprob.copy()
    dScores[np.arange(num_samples), y] -= 1
    dScores /= num_samples

    return loss, dScores


