from layers import *


class FullyConnectedNet(object):
    """
    A general purpose class for a fully-connected neural network of arbitrary
    size built with ReLU activation and Softmax loss.
    """
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10, reg=0.0,
                 std=1e-2, dtype=np.float32):
        """
        :param hidden_dims: list of integers of hidden layer sizes
        :param input_dim: int number of input features
        :param num_classes: int number of classes
        :param reg: float regularization
        :param std: scale for random initialization
        :param dtype: desired type of weight matrices
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.w_keys = ['W{}'.format(layer+1) for layer in range(self.num_layers)]
        self.b_keys = ['b{}'.format(layer+1) for layer in range(self.num_layers)]
        self.num_classes = num_classes
        self._reset_parameters(input_dim, hidden_dims, std, dtype)

    def _reset_parameters(self, input_dim, hidden_dims, std, dtype):
        """
        Resets all weights and biases of the net
        :param input_dim: int number of input features
        :param hidden_dims: int list of hidden_layer sizes
        :param std: scale for random initialization
        :param dtype: desired type of weight matrices
        :return:
        """
        previous_dim = input_dim
        for w_key, b_key, hidden_dim in zip(self.w_keys[:-1], self.b_keys[:-1], hidden_dims):
            self.params[w_key] = std * np.random.randn(previous_dim, hidden_dim).astype(dtype)
            self.params[b_key] = np.zeros(hidden_dim, dtype=dtype)
            previous_dim = hidden_dim
        self.params[self.w_keys[-1]] = std * np.random.randn(previous_dim, self.num_classes).astype(dtype)
        self.params[self.b_keys[-1]] = np.zeros(self.num_classes, dtype=dtype)

    def loss(self, X, y=None):
        """
        Compute loss of network on data
        :param X: (num_samples, num_features) np array of samples
        :param y: (num_samples) np array of classes for samples
        :return: float loss and dict of gradients
        """
        X = X.astype(self.dtype)

        # Apply forward pass and save caches
        scores, caches = self.forward_pass(X)

        # Return scores if testing
        if y is None:
            return scores

        loss, grads = self.backward_pass(caches, scores, y)
        return loss, grads

    def forward_pass(self, X):
        """
        Apply network's forward pass on data to get classwise scores per sample
        :param X: (num_samples, num_features) np array of samples
        :return: (num_samples, num_classes) np array of class scores and layer caches
        """
        caches = []
        states = X
        for w_key, b_key in zip(self.w_keys[:-1], self.b_keys[:-1]):
            w, b = self.params[w_key], self.params[b_key]
            states, cache = full_relu_forward(states, w, b)
            caches.append(cache)

        w, b = self.params[self.w_keys[-1]], self.params[self.b_keys[-1]]
        scores, cache = full_forward(states, w, b)
        caches.append(cache)
        return scores, caches

    def backward_pass(self, caches, scores, y):
        """
        Apply backward pass on data by computing gradients
        :param caches: list of tuple caches from forward pass through layers
        :param scores: (num_samples, num_classes) np array of class scores
        :param y: (num_samples) np array of classes for samples
        :return: float loss for samples and dict of gradients for layers
        """
        grads = {}
        loss, dscores = softmax_loss(scores, y)
        dhidden, grads[self.w_keys[-1]], grads[self.b_keys[-1]] = full_backward(dscores, caches[-1])
        for w_key, b_key, cache in zip(self.w_keys[:-1][::-1], self.b_keys[:-1][::-1], caches[:-1][::-1]):
            dhidden, grads[w_key], grads[b_key] = full_relu_backward(dhidden, cache)

        # Apply regularization if necessary
        if self.reg:
            loss += 0.5 * self.reg * np.sum([np.sum(np.square(self.params[w_key])) for w_key in self.w_keys])
            for w_key in self.w_keys:
                grads[w_key] += self.reg * self.params[w_key]
        return loss, grads

    def predict(self, X):
        """
        Predict the classes of given samples with the network
        :param X: (num_samples, num_features) np array
        :return: (num_samples) np array of classes for data
        """
        scores, _ = self.forward_pass(X)
        return np.argmax(scores, axis=1)
