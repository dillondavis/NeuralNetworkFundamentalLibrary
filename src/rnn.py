import numpy as np
from layers import *
from rnn_layers import *
from cnn_layers import *


class ImageCaptionRNN(object):
    """
    Generates captions for images from a vocab given image features.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, dtype=np.float32):
        """
        :param word_to_idx: dict mapping a word to a unique int
        :param input_dim: int D dimension of input image features
        :param wordvec_dim: int W dimension of word vectors
        :param hidden_dim: int H dimension for RNN hidden state
        :param dtype: np datatype for weights
        """
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize image features weights
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim).astype(dtype)
        self.params['W_proj'] /= np.sqrt(input_dim).astype(dtype)
        self.params['b_proj'] = np.zeros(hidden_dim).astype(dtype)

        # Initialize word embedding weights
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim).astype(dtype)
        self.params['W_embed'] /= 100

        # Initialize RNN weights
        dim_mul = 1
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim).astype(dtype)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim).astype(dtype)
        self.params['b'] = np.zeros(dim_mul * hidden_dim).astype(dtype)

        # Initialize hidden to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim).astype(dtype)
        self.params['b_vocab'] = np.zeros(vocab_size).astype(dtype)


    def loss(self, features, captions):
        """
        Apply forward pass on input image features and computes softmax loss given scores
        :param features: (N, D) np array of image features
        :param captions: (N, T) np array of integers mapping to correct captions
        :return: float cross entropy loss and dict of grads for parameters
        """
        captions_out = captions[:, 1:]
        mask = (captions_out != self._null)

        scores, cache = self.forward_pass(features, captions)
        loss, dscores = sequential_softmax_loss(scores, captions_out, mask)
        grads = self.backward_pass(dscores, cache)
        return loss, grads

    def forward_pass(self, features, captions):
        """
        Apply forward pass to data to compute vocaba scores
        :param features: (N, D) np array of image features to caption
        :param captions: (N, T) np array of integers mapping to correct captions
        :return: scores per timestep per vocab word and cache
        """
        captions_in = captions[:, :-1]
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        initial_hidden, initial_cache = full_forward(features, W_proj, b_proj)
        start_captions, word_cache = word_embedding_forward(captions_in, W_embed)
        hh, rnn_cache = rnn_forward(start_captions, initial_hidden, Wx, Wh, b)
        scores, score_cache = sequential_full_forward(hh, W_vocab, b_vocab)
        return scores, (initial_cache, word_cache, rnn_cache, score_cache)

    def backward_pass(self, dscores, cache):
        """
        Compute gradients for each layer given upstream gradient and cache
        :param dscores: (N, T, V) np array of vocab scores for each timestep of each image
        :param cache: cache of forward pass
        :return: dict grads of gradients for each layer
        """
        grads = {}
        initial_cache, word_cache, rnn_cache, score_cache = cache
        dhh, grads['W_vocab'], grads['b_vocab'] = sequential_full_backward(dscores, score_cache)
        dcaptions, dinitialhidden, grads['Wx'], grads['Wh'], grads['b']= rnn_backward(dhh, rnn_cache)
        grads['W_embed'] = word_embedding_backward(dcaptions, word_cache)
        dfeatures, grads['W_proj'], grads['b_proj'] = full_backward(dinitialhidden, initial_cache)

        return grads


