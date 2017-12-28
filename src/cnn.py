import numpy as np
from layers import *
from cnn_layers import *


class ThreeLayerConvNet(object):
    """
    Three Layer Convolutional Network
    (Conv, ReLU, 2x2 MaxPool) -> (FullyConnected, ReLU) -> Full -> Softmax
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, std=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        :param input_dim: (num_channels, height, width) dimension of input images
        :param num_filters: int num_filters for conv layer
        :param filter_size: int filter size for width/height of conv layer filters
        :param hidden_dim: int size of hidden state of FullyConnected/ReLU layer
        :param num_classes: int num classes for data
        :param std: float scaling factor for random initilization
        :param reg: float strength of L2 regularization
        :param dtype: dtype for weights
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.init_conv_pool_config(filter_size, input_dim)

        C, H, W = input_dim
        self.params['W1'] = std * np.random.randn(num_filters, C, filter_size, filter_size).astype(dtype)
        self.params['b1'] = np.zeros(num_filters).astype(dtype)
        self.params['W2'] = std * np.random.randn(self.Hpool*self.Wpool*num_filters, hidden_dim).astype(dtype)
        self.params['b2'] = np.zeros(hidden_dim).astype(dtype)
        self.params['W3'] = std * np.random.randn(hidden_dim, num_classes).astype(dtype)
        self.params['b3'] = np.zeros(num_classes).astype(dtype)

    def init_conv_pool_config(self, filter_size, input_dim):
        """
        Initializes convolutional and pool config member variables
        :param filter_size: int size of filter length/width
        :param input_dim: shape of input images
        """
        C, H, W = input_dim
        self.pad = (filter_size - 1) // 2
        self.conv_stride = 1
        self.pool_height = 2
        self.pool_width = 2
        self.pool_stride = 2
        self.Hconv = 1 + (H + 2 * self.pad - filter_size) // self.conv_stride
        self.Wconv = 1 + (W + 2 * self.pad - filter_size) // self.conv_stride
        self.Hpool = ((self.Hconv - self.pool_height) // self.pool_stride) + 1
        self.Wpool = ((self.Wconv - self.pool_width) // self.pool_stride) + 1

    def loss(self, X, y=None):
        """
        Find loss and gradients of ConvNet with respect to mini batch of data
        :param X: (num_samples, num_channels, height, width) np array of samples
        :param y:  (num_samples) np array of classes for samples
        :return: float loss and dict gradients
        """
        scores, caches = self.forward_pass(X)

        if y is None:
            return scores

        loss, grads = self.backward_pass(scores, y, caches)

        return loss, grads

    def forward_pass(self, X):
        """
        Apply forward pass of network to batch of data to get classwise scores and cache
        :param X: (num_samples, num_channels, height, width) np array of samples
        :return: (num_samples, num_classes) np array of class wise scores for samples and cache
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        conv_param = {'stride': self.conv_stride, 'pad': self.pad}
        pool_param = {'pool_height': self.pool_height, 'pool_width': self.pool_width,
                      'stride': self.pool_stride}
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out2, cache2 = full_relu_forward(out1, W2, b2)
        scores, cache3 = full_forward(out2, W3, b3)

        return scores, (cache1, cache2, cache3)

    def backward_pass(self, scores, y, caches):
        """
        Apply backward pass of network from batch of data to get loss and gradients
        :param scores: classwise scores for each sample
        :param y: real classes for each sample
        :param caches: caches from forward pass
        :return: float loss and dict of gradients
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        cache1, cache2, cache3 = caches
        grads = {}
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.square(np.sum(W3)))

        dx, grads['W3'], grads['b3'] = full_backward(dx, cache3)
        grads['W3'] += self.reg * W3

        dx, grads['W2'], grads['b2'] = full_relu_backward(dx, cache2)
        grads['W2'] += self.reg * W2

        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, cache1)
        grads['W1'] += self.reg * W1

        return loss, grads

    def predict(self, X):
        """
        Predict the classes of given samples with the network
        :param X: (num_samples, num_channels, height, width) np array
        :return: (num_samples) np array of classes for data
        """
        scores, _ = self.forward_pass(X)
        return np.argmax(scores, axis=1)
