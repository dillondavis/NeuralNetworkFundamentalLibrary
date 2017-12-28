import pickle
import glob
import numpy as np


def _load_batch(filename):
    """
    Load one cifar batch from file
    :param filename: string filename of cifar batch
    :return: data from cifar batch
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    X, y = data['data'].reshape(10000, 32, 32, 3).astype('float'), data['labels']

    return X, np.array(y)


def _load_cifar(root_dir):
    """
    Load cifar batch data from a root directory
    :param root_dir: string path of directory containing cifar data
    :return: train and test datasets
    """
    data_files = glob.glob(root_dir+'/data*')
    data = [_load_batch(filename) for filename in data_files]
    X_train = np.concatenate([batch[0] for batch in data])
    y_train = np.concatenate([batch[1] for batch in data])
    X_test, y_test = _load_batch(glob.glob(root_dir + '/test*')[0])
    return X_train, y_train, X_test, y_test


def get_cifar_data(num_train=49000, num_validation=1000, num_test=1000):
    """
    Builds a normalized dataset of cifar data containing train, test, and validation
    :param num_train: number of training samples
    :param num_validation: number of validation samples
    :param num_test: number of test samples
    :return:
    """
    # Get data and build train, test, validation sets
    cifar_dir = '/Users/Dillon/UIUC/cs242/cs242FinalProject/data/cifar'
    X_train, y_train, X_test, y_test = _load_cifar(cifar_dir)
    X_val, y_val = X_train[num_train: num_train+num_validation], y_train[num_train:num_train+num_validation]
    X_train, y_train = X_train[:num_train], y_train[:num_train]
    X_test, y_test = X_test[:num_test], y_test[:num_test]

    # Normalize data
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data into two dimensions and return data as dictionary
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


def get_conv_cifar_data(num_train=49000, num_validation=1000, num_test=1000):
    """
    Builds a normalized dataset of cifar data containing train, test, and validation for a ConvNet (4 dimensions)
    :param num_train: number of training samples
    :param num_validation: number of validation samples
    :param num_test: number of test samples
    :return:
    """
    # Get data and build train, test, validation sets
    cifar_dir = '/Users/Dillon/UIUC/cs242/cs242FinalProject/data/cifar'
    X_train, y_train, X_test, y_test = _load_cifar(cifar_dir)
    X_val, y_val = X_train[num_train: num_train+num_validation], y_train[num_train:num_train+num_validation]
    X_train, y_train = X_train[:num_train], y_train[:num_train]
    X_test, y_test = X_test[:num_test], y_test[:num_test]

    # Normalize data
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data into two dimensions and return data as dictionary
    X_train = X_train.transpose(0, 3, 1, 2)
    X_val = X_val.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


def random_mini_batch(X, y, batch_size):
    """
    Create a random subset of given size of the given data
    :param X: (num_samples, num_features) np array of data
    :param y: (num_samples) np array of classes
    :param batch_size: size of desired batch
    :return:
    """
    batch_size = X.shape[0] if not batch_size else batch_size
    batch = np.random.choice(X.shape[0], batch_size)
    return X[batch], y[batch]