from layers import *
import numpy as np


"""
RECURRENT NEURAL NETWORK LAYERS
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Apply tanh forward pass of RNN for a single timestep
    :param x: (N, D) np array of data
    :param prev_h: (N, H) np array of hidden states from previous timesteps
    :param Wx: (D, H) np array of x weights
    :param Wh: (H, H) np array of prev_h weights
    :param b: (H) np array of biases
    :return: (N, H) np array of hidden states and cache of intermediate states
    """
    curr_h = x @ Wx
    curr_prev_h = prev_h @ Wh
    h_sum = curr_h + curr_prev_h + b
    next_h = 2 * sigmoid(2 * h_sum) - 1
    cache = (x, prev_h, Wx, Wh, curr_h, curr_prev_h, h_sum, next_h)
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Apply rnn backward pass for a single timestep to compute gradients
    :param dnext_h: (N, H) np array of gradients on output of rnn step forward pass
    :param cache: cache of intermediate states from rnn
    :return: np arrays of gradients on x, prev_h, Wx, Wh, and b of the same shape
    """
    x, prev_h, Wx, Wh, curr_h, curr_prev_h, h_sum, next_h = cache
    dhsum = dnext_h * 4 * (sigmoid(2 * h_sum) ** 2) * np.exp(-2*h_sum)
    dcurrh = dhsum
    dcurrprevh = dhsum
    db = np.sum(dhsum, axis=0)
    dprev_h = dcurrprevh @ Wh.T
    dWh = prev_h.T @ dcurrprevh
    dx = dcurrh @ Wx.T
    dWx = (dcurrh.T @ x).T
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Apply tanh forward pass of RNN on complete sequences of data
    :param x: (N, D) np array of data
    :param h0: (N, H) np array of initial hidden states
    :param Wx: (D, H) np array of x weights
    :param Wh: (H, H) np array of prev_h weights
    :param b: (H) np array of biases
    :return: (N, T, H) np array of hidden states and cache of intermediate states
    """
    (N, T, D), H = x.shape, b.shape[0]
    h = np.zeros((N, T, H))
    cache = []
    prev_h = h0

    for t_i in range(T):
        prev_h, cache_i = rnn_step_forward(x[:, t_i], prev_h, Wx, Wh, b)
        h[:, t_i] = prev_h
        cache.append(cache_i)
    return h, cache


def rnn_backward(dh, cache):
    """
    Apply rnn backward pass for whole sequences of data to compute gradients
    :param dh: (N, T, H) np array of gradients for each timestep
    :param cache: cache of intermediate states from rnn
    :return: np arrays of gradients on x, h0, Wx, Wh, and b of the same shape
    """
    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)
    dprev_hi = np.zeros_like(dh0)

    # Compute gradients of timesteps in reverse order
    for t_i in range(T-1, -1, -1):
        dht = dh[:, t_i] + dprev_hi
        dxi, dprev_hi, dWxi, dWhi, dbi = rnn_step_backward(dht, cache[t_i])
        dx[:, t_i] = dxi
        dh0 = dprev_hi
        dWx += dWxi
        dWh += dWhi
        db += dbi
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Apply word embedding forward pass replacing each vocab index of each timestep of a sequence
    with a vector representing the word
    :param x: (N, T) np array of vocab indices for each timestep of each sequence
    :param W: (V, D) np array of vectors representing each word
    :return: (N, T, D) np array of word vectors for each timestep, cache of intermediates
    """
    out = np.reshape(W[x.flatten()], (x.shape[0], x.shape[1], -1))
    cache = (x, W)

    return out, cache


def word_embedding_backward(dout, cache):
    """
    Apply word embedding backward pass to compute gradients on word embeddings
    :param dout: (N, T, D) np array of gradients on output of word embedding forward
    :param cache: cache of intermediates from word embedding forward pass
    :return: (N, T, D) np array of gradients on embeddings
    """
    N, T, D = dout.shape
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x.flatten(), dout.reshape(-1, D))

    return dW


def sequential_full_forward(x, w, b):
    """
    Compute forward pass of fully connected layer
    :param x: (N, T, D) np array of data
    :param w: (D, H) np array of weights
    :param b: (H) np array of biases
    :return: (N, T, H) output of forward pass and cache of layer
    """
    N, T, D = x.shape
    H = b.shape[0]
    x_flat = x.reshape((N*T, D))
    out = (x_flat @ w + b).reshape((N, T, H))
    cache = (x, w, b)

    return out, cache


def sequential_full_backward(dout, cache):
    """
    Compute backward pass/find gradients of a fully connected layer
    :param dout: gradient of output
    :param cache: cache of full and relu layers
    :return: gradients of x, w, b
    """
    x, w, b = cache
    N, T, D = x.shape
    H = b.shape[0]

    dx = np.reshape(dout.reshape((-1, H)) @ w.T, x.shape)
    dw = x.reshape(-1, D).T @ dout.reshape((-1, H))
    db = np.sum(dout, axis=(0, 1))

    return dx, dw, db


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Apply tanh forward pass of LSTM for a single timestep
    :param x: (N, D) np array of data
    :param prev_h: (N, H) np array of hidden states from previous timesteps
    :param prev_c: (N, H) np array of hidden states from previous timesteps
    :param Wx: (D, 4H) np array of x weights
    :param Wh: (H, 4H) np array of prev_h weights
    :param b: (4H) np array of biases
    :return: (N, H) np arrays of hidden states/cell states and cache of intermediate states
    """
    _, h = prev_h.shape
    a = x @ Wx + prev_h @ Wh + b
    ai, af, ao, ag = a[:, :h], a[:, h:2*h], a[:, 2*h:3*h], a[:, 3*h:4*h]
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = 2 * sigmoid(2*ag) - 1

    next_c = f * prev_c + i * g
    next_h = o * (2 * sigmoid(2*next_c) - 1)
    cache = (x, Wx, prev_h, Wh, prev_c, a, ai, af, ao, ag, i, f, o, g, next_c, next_h)

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Apply rnn backward pass for a single timestep to compute gradients
    :param dnext_h: (N, H) np array of gradients on hidden output of lstm step forward pass
    :param dnext_c: (N, H) np array of gradients on cell output of lstm step forward pass
    :param cache: cache of intermediate states from lstm
    :return: np arrays of gradients on x, prev_h, prev_c, Wx, Wh, and b of the same shape
    """
    x, Wx, prev_h, Wh, prev_c, a, ai, af, ao, ag, i, f, o, g, next_c, next_h = cache
    _, h = prev_h.shape
    dnext_c += (dnext_h * o * 4 * sigmoid(2 * next_c) ** 2 * np.exp(-2*next_c))
    do = dnext_h * next_h / o
    df = dnext_c * prev_c
    dprev_c = dnext_c * f
    di = dnext_c * g
    dg = dnext_c * i
    dag = dg * 4 * (sigmoid(2 * ag) ** 2) * np.exp(-2*ag)
    dao = do * (sigmoid(ao) ** 2) * np.exp(-ao)
    daf = df * (sigmoid(af) ** 2) * np.exp(-af)
    dai = di * (sigmoid(ai) ** 2) * np.exp(-ai)

    da = np.zeros_like(a)
    da[:, :h] = dai
    da[:, h:2*h] = daf
    da[:, 2*h:3*h] = dao
    da[:, 3*h:4*h] = dag

    dWx = x.T @ da
    dWh = prev_h.T @ da
    db = np.sum(da, axis=0)
    dx = da @ Wx.T
    dprev_h = da @ Wh.T

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Apply tanh forward pass of LSTM on complete sequences of data
    :param x: (N, T, D) np array of data
    :param h0: (N, H) np array of initial hidden states
    :param Wx: (D, 4H) np array of x weights
    :param Wh: (H, 4H) np array of prev_h weights
    :param b: (4H) np array of biases
    :return: (N, T, H) np array of hidden states and cache of intermediate states
    """
    (N, T, D), H = x.shape, b.shape[0]//4
    h = np.zeros((N, T, H))
    cache = []
    prev_h = h0
    prev_c = np.zeros((N, H))
    for t_i in range(T):
        prev_h, prev_c, cache_i = lstm_step_forward(x[:, t_i], prev_h, prev_c, Wx, Wh, b)
        h[:, t_i] = prev_h
        cache.append(cache_i)

    return h, cache


def lstm_backward(dh, cache):
    """
    Apply LSTM backward pass for whole sequences of data to compute gradients
    :param dh: (N, T, H) np array of gradients for each timestep
    :param cache: cache of intermediate states from LSTM
    :return: np arrays of gradients on x, h0, Wx, Wh, and b of the same shape
    """

    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros(4*H)
    dprev_hi = np.zeros_like(dh0)
    dprev_ci = np.zeros_like(dh0)
    for t_i in range(T-1, -1, -1):
        dht = dh[:, t_i] + dprev_hi
        dxi, dprev_hi, dprev_ci, dWxi, dWhi, dbi = lstm_step_backward(dht, dprev_ci, cache[t_i])
        dx[:, t_i] = dxi
        dh0 = dprev_hi
        dWx += dWxi
        dWh += dWhi
        db += dbi

    return dx, dh0, dWx, dWh, db


def sigmoid(x):
    """
    Numerically stable logistic sigmoid function implementation
    :param x: (N, T, H) np array of sequential data
    :return: x with sigmoid applied to it of same shape
    """
    pos_x = x >= 0
    neg_x = x < 0
    bottom = np.zeros_like(x)
    bottom[pos_x] = np.exp(-x[pos_x])
    bottom[neg_x] = np.exp(x[neg_x])
    top = bottom.copy()
    top[pos_x] = 1
    sigm = top / (1 + bottom)

    return sigm


def sequential_softmax_loss(x, y, mask):
    """
    Softmax Loss for sequential data in RNNs.
    :param x: (N, T, V) np array of scores for vocab for each timestep
    :param y: (N, T) np array of correct captions
    :param mask: mask for padded data
    :return: float loss and gradient for x
    """
    N, T, V = x.shape

    scores = x.reshape(N * T, V)
    y = y.reshape(N * T)
    mask = mask.reshape(N * T)

    # Compute Loss
    num_samples = scores.shape[0]
    fstable = (scores.T - np.amax(scores, axis=1)).T
    fexp = np.exp(fstable)
    fprob = fexp / np.sum(fexp, axis=1, keepdims=True)
    floss = np.log(fprob[np.arange(num_samples), y] + 1e-14)
    loss = -(np.sum(mask * floss) / num_samples)

    # Compute gradients on scores
    dScores = fprob.copy()
    dScores[np.arange(num_samples), y] -= 1
    dScores /= num_samples
    dScores *= mask[:, None]
    dScores = dScores.reshape(N, T, V)

    return loss, dScores


