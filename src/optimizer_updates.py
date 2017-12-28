import numpy as np


def sgd(x, dx, config=None):
    """
    Run vanilla sgd update
    :param x: np matrix of weights
    :param dx: gradient of weights
    :param config: dict containing learning_rate
    :return: updated weights and config
    """
    config = _get_default_config(config, x, 1e-2)
    next_x = x - config['learning_rate'] * dx

    return next_x, config


def sgd_momentum(x, dx, config=None):
    """
    Run momentum sgd update
    :param x: np matrix of weights
    :param dx: gradient of weights
    :param config: dict containing learning_rate, momentum, velocity
    :return: updated weights and config
    """
    config = _get_default_config(config, x, 1e-2)
    v = config['velocity']

    v = config['momentum'] * v - config['learning_rate'] * dx
    next_x = x + v
    config['velocity'] = v

    return next_x, config


def rmsprop(x, dx, config=None):
    """
    Run root mean square propagation update
    :param w: np matrix of weights
    :param dw: gradient of weights
    :param config: dict containing learning_rate, decay_rate, epsilon, cache
    :return: updated weights and config
    """
    config = _get_default_config(config, x, 1e-2)

    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dx**2
    next_x = x + (-config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon']))

    return next_x, config


def adam(x, dx, config=None):
    """
    Run adam optimizer update
    :param w: np matrix of weights
    :param dw: gradient of weights
    :param config: dict containing learning_rate, beta1, beta2, epsilon, m, v, t
    :return: updated weights and config
    """
    # Unpack variables and set up config
    config = _get_default_config(config, x, 1e-3)
    m = config['m']
    v = config['v']
    t = config['t'] + 1
    beta1 = config['beta1']
    beta2 = config['beta2']
    lr = config['learning_rate']
    eps = config['epsilon']

    # Update momentum and velocity based off gradient
    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * dx**2

    # Adjust momemtum and gradient based off timesetp
    mb = m / (1 - beta1**t)
    vb = v / (1 - beta2**t)

    next_x = x - lr * mb / (np.sqrt(vb) + eps)

    config['m'] = m
    config['v'] = v
    config['t'] = t

    return next_x, config


def _get_default_config(config, x, lr):
    """
    Sets default config params for optimizer updates
    :param config: dict of params given to optimizer
    :param x: np array of data given to optimizer
    :param lr: default lr for optimizer type
    :return: updated config
    """
    config = {} if config is None else config
    config.setdefault('learning_rate', lr)
    config.setdefault('momentum', 0.9)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)
    config.setdefault('velocity', np.zeros_like(x))

    return config
