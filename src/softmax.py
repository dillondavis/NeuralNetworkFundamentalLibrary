import numpy as np


class Softmax(object):
    """
    Softmax classifier with cross entropy loss
    """
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200):
        """
        Train Softmax using SGD
        :param X: (num_samples, num_dimensions) numpy array of training data
        :param y: (num_samples) numpy array of labels for training_data
        :param learning_rate: float step size for optimization
        :param reg: float lambda regularization strength
        :param num_iters: int number of steps for SGD
        :param batch_size: int num_samples/batch
        :return: Float list of losses
        """

        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        self.W = 0.001 * np.random.randn(dim, num_classes) if not self.W else self.W

        # Run SGD to optimize softmax
        losses = []
        for iter in range(num_iters):
            # Choose a random minibatch
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # Find the loss and gradient and update parameters
            loss, grad = self.loss(X_batch, y_batch, reg)
            losses.append(loss)
            self.W -= grad * learning_rate

            if iter % 100 == 0:
                print('iteration {} / {}: loss {}'.format(iter, num_iters, loss))

        return losses

    def predict(self, X):
        """
        Predict classes for samples in X
        :param X: (num_samples, num_features) numpy array of data
        :return: (num_samples) numpy array of classes
        """
        y_pred = np.argmax(X @ self.W, axis=1)

        return y_pred

    def loss(self, X, y, reg):
        """
        Find cross entropy loss of classifier on training data
        :param X: (num_samples, num_features) numpy array of data
        :param y: (num_samples) numpy array of classes for data
        :param reg: float regularization strength
        :return:
        """
        num_train, img_size = X.shape

        # Find scores, stabilize, and apply cross entropy
        f = X @ self.W
        fstable = (f.T - np.amax(f, axis=1)).T
        fexp = np.exp(fstable)
        fprob = fexp / np.sum(fexp, axis=1, keepdims=True)

        # Find loss from cross entropy
        floss = np.log(fprob[np.arange(num_train), y] + 1e-16)
        loss = -(np.sum(floss) / len(y)) + 0.5 * reg * np.sum(np.square(self.W))

        # Find gradient of parameters from cross entropy
        dW = fprob.copy()
        dW[np.arange(num_train), y] -= 1
        dW /= num_train
        dW = X.T @ dW + reg * self.W

        return loss, dW
