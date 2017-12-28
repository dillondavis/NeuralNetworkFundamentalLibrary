import optimizer_updates
from data_utils import random_mini_batch


class Optimizer(object):
    """
    General purpose optimizer compatible with SGD, SGD-Momentum, RMSProp, and Adam for a model
    Requires input model to follow the same API as FullyConnectedNet
    i.e. model.params dict, model.loss(x, y) computing loss and gradients
    """

    def __init__(self, model, data, optim='sgd', optim_config={}, lr_decay=1.0,
                 batch_size=100, num_epochs=10, num_train=10000, num_val=0):
        """
        :param model: model to optimize with API of FullyConnectedNet
        :param data: dictionary containing X_train, y_train, X_val, y_val
        :param optim: string optimizer name i.e. sgd, sgd_momentum, rms, adam
        :param optim_config: dictionary of config for optimizer
        :param lr_decay: float decay rate of learning
        :param batch_size: int minibatch size for training
        :param num_epochs: int num epochs for training
        :param num_train: int num training samples to use
        :param num_val: int num validation samples to use
        """
        self.model = model
        self.X_train, self.y_train = data['X_train'], data['y_train']
        self.X_val, self.y_val = random_mini_batch(data['X_val'], data['y_val'], num_val)

        self.optim = optim
        self.optim_config = optim_config
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_train = num_train
        self.num_val = num_val

        self.update_rule = getattr(optimizer_updates, self.optim)
        self._set_params_configs()

    def _set_params_configs(self):
        """
        Initializes history/best/config model variables for training
        """

        self.losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.best_params = {}
        self.optim_configs = {param_name: self.optim_config.copy() for param_name in self.model.params}

    def _step(self):
        """
        Make one gradient update based on a minibatch of the training data
        """
        X_batch, y_batch = random_mini_batch(self.X_train, self.y_train, self.batch_size)

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.losses.append(loss)
        self.update_params(grads)

    def update_params(self, grads):
        """
        Update params of model based on gradients
        :param grads: dictionary of gradients for each param
        """
        for param, weights in self.model.params.items():
            dw = grads[param]
            config = self.optim_configs[param]
            weights, config = self.update_rule(weights, dw, config)
            self.model.params[param] = weights
            self.optim_configs[param] = config

    def test(self, X, y, batch_size=None):
        """
        Predict classes of data and compute accuracy
        :param X: (num_samples, num_features) np array of samples
        :param y: (num_samples) np array of classes for samples
        :param batch_size: optional int subset size of data to test
        :return: float acc on given data
        """
        X, y = random_mini_batch(X, y, batch_size) if batch_size else (X, y)

        y_pred = self.model.predict(X)
        acc = (y_pred == y).mean()

        return acc

    def train(self):
        """
        Train the model with specified optimization
        """
        num_train = self.X_train.shape[0]
        steps_per_epoch = max(num_train // self.batch_size, 1)

        for epoch in range(self.num_epochs):
            self.evaluate(epoch)
            for step in range(steps_per_epoch):
                self._step()
                print('Epoch {} Step {} / {} loss: {}'.format(epoch, step + 1, steps_per_epoch, self.losses[-1]))
            self.decay_learning()
        self.evaluate(self.num_epochs)

        self.model.params = self.best_params

    def decay_learning(self):
        """
        Decay learning rate in configs for each parameter
        """
        for param in self.optim_configs:
            self.optim_configs[param]['learning_rate'] *= self.lr_decay

    def evaluate(self, epoch=None):
        """
        Evaluate the current state of the model on train and test data
        :param epoch: optional int epoch of training
        :return: float accuracy on train data and float accuracy on validation data
        """
        train_acc = self.test(self.X_train, self.y_train, batch_size=self.num_train)
        self.train_accs.append(train_acc)
        val_acc = self.test(self.X_val, self.y_val, batch_size=self.num_val)
        self.val_accs.append(val_acc)

        if epoch:
            print('Epoch {} / {} train acc: {}; val_acc: {}'.format(epoch, self.num_epochs, train_acc, val_acc))

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_params = self.model.params.copy()
        return train_acc, val_acc
