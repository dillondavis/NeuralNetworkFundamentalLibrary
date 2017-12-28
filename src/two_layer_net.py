import numpy as np


class TwoLayerNet(object):
    """
    Two layer fully connected neural network
    input -> fully connected -> ReLU -> fully connected -> softmax
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize weights randomly and biases to zero
        :param input_size: Num features of input data
        :param hidden_size: Size of hidden layer
        :param output_size: Number of classes
        :param std: stddev for random initialization
        """
        self.params = {
            'W1': std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }

    def loss(self, X, y=None, reg=0.0):
        """
        Find the loss and gradients for a two layer net
        :param X: (num_samples, num_features) numpy array of data
        :param y: (num_samples) array of classes
        :param reg: float regularization strength
        :return: (num_samples, num_classes) Softmax scores if no y
                 float loss and dictionary of gradients if y is given
        """
        # Unpack weights and apply forward pass
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        hidden_states, scores = self.forward_pass(X)
        if y is None:
            return scores

        # Find the loss
        fstable = (scores.T - np.amax(scores, axis=1)).T
        fexp = np.exp(fstable)
        fprob = fexp / np.sum(fexp, axis=1, keepdims=True)
        floss = np.log(fprob[np.arange(N), y] + 1e-14)
        loss = -(np.sum(floss) / N) + 0.5 * reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        # Backward pass and find gradients
        grads = {}
        dscores = fprob.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N
        grads['W2'] = hidden_states.T @ dscores + reg*W2
        grads['b2'] = np.sum(dscores, axis=0)
        dhidden = W2 @ dscores.T
        dhidden[hidden_states.T <= 0] = 0

        grads['W1'] = X.T @ dhidden.T + reg*W1
        grads['b1'] = np.sum(dhidden, axis=1)

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200):
        """
        Train two layer net with Momentum SGD
        :param X: (num_train, num_features) numpy array of data
        :param y: (num_train) array of classes
        :param X_val: (num_val, num_features) numpy array of validation data
        :param y_val: (num_val) numpy array of classes for validation data
        :param learning_rate: float step size for SGD
        :param learning_rate_decay: float decay rate for step size
        :param reg: float regularization strength
        :param num_iters: number of steps for SGD
        :param batch_size: int data items per step
        :return: Dictionary of loss, train acc, and validation acc over time
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use Momentum SGD to optimize the parameters in self.model
        losses = []
        train_accs = []
        val_accs = []
        mu = 0.9
        param_names = ['W1', 'W2', 'b1', 'b2']
        velocity = {param_name: 0 for param_name in param_names}

        for iter in range(num_iters):
            # Choose a random minibatch
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # Find loss and gradients and update params
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            losses.append(loss)

            for param_name in param_names:
                velocity[param_name] = mu * velocity[param_name] - learning_rate * grads[param_name]
                self.params[param_name] += velocity[param_name]

            if iter % 100 == 0:
                print('iteration {} / {}: loss {}'.format(iter, num_iters, loss))

            # Check accuracies and decay learning rate every epoch
            if iter % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_accs.append(train_acc)
                val_accs.append(val_acc)

                learning_rate *= learning_rate_decay

        return {
            'loss': losses,
            'train_acc': train_accs,
            'val_acc': val_accs,
        }

    def predict(self, X):
        """
        Predict the class of each sample in X
        :param X: (num_samples, num_features) numpy array of data
        :return: (num_samples) numpy array of classes for the data
        """

        hidden_states, scores = self.forward_pass(X)
        y_pred = np.argmax(scores, axis=1)

        return y_pred

    def forward_pass(self, X):
        """
        Compute the forward pass of the two layer net and output hidden states and scores
        :param X: (num_samples, num_features) numpy array of data
        :return: (hidden_size, num_classes) numpy array of hidden states
                 (num_samples, num_classes) numpy array of classwise scores for each sample
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        hidden_states = TwoLayerNet.activation((X @ W1) + b1)
        scores = hidden_states @ W2 + b2
        return hidden_states, scores

    def activation(X):
        """
        Apply reLU activation function to X
        """
        return np.maximum(0, X)

