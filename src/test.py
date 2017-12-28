import unittest
from layers import *
from cnn_layers import *
from rnn_layers import *
from softmax import Softmax
from optimizer import Optimizer
from rnn import ImageCaptionRNN
from cnn import ThreeLayerConvNet
from two_layer_net import TwoLayerNet
from data_utils import get_cifar_data, get_conv_cifar_data
from fully_connected_net import FullyConnectedNet


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.data = get_cifar_data()
        self.X_train = self.data['X_train']
        self.y_train = self.data['y_train']
        self.X_val = self.data['X_val']
        self.y_val = self.data['y_val']
        self.X_test = self.data['X_test']
        self.y_test = self.data['y_test']
        self.model = Softmax()

    def testModel(self):
        np.random.seed(100)
        num_epochs = 15
        num_train_examples = self.X_train.shape[0]
        batch_size = 200
        num_iters = int(np.round(num_epochs * num_train_examples / float(batch_size)))
        lr = 5.6e-06
        reg = 1.5e+02

        losses = self.model.train(self.X_train, self.y_train, learning_rate=lr, reg=reg, num_iters=num_iters)
        self.assertLess(losses[-1], 2.0)
        val_acc = np.mean(self.model.predict(self.X_val) == self.y_val)
        self.assertGreater(val_acc, 0.35)
        test_acc = np.mean(self.model.predict(self.X_test) == self.y_test)
        self.assertGreater(test_acc, 0.35)


class TestTwoLayerNet(unittest.TestCase):
    def setUp(self):
        self.data = get_cifar_data()
        self.X_train = self.data['X_train']
        self.y_train = self.data['y_train']
        self.X_val = self.data['X_val']
        self.y_val = self.data['y_val']
        self.X_test = self.data['X_test']
        self.y_test = self.data['y_test']
        self.model = TwoLayerNet(32*32*3, 50, 10)

    def testModel(self):
        np.random.seed(100)
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val,
                         num_iters=5000, batch_size=500,
                         learning_rate=1e-3, learning_rate_decay=0.95,
                         reg=0.5)

        val_acc = (self.model.predict(self.X_val) == self.y_val).mean()
        test_acc = (self.model.predict(self.X_test) == self.y_test).mean()
        self.assertGreater(val_acc, 0.48)
        self.assertGreater(test_acc, 0.48)


class TestLayers(unittest.TestCase):
    def testFullForward(self):
        num_inputs = 2
        input_shape = (4, 5, 6)
        output_dim = 3

        input_size = num_inputs * np.prod(input_shape)
        weight_size = output_dim * np.prod(input_shape)

        x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
        w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
        b = np.linspace(-0.3, 0.1, num=output_dim)
        out, _ = full_forward(x, w, b)
        correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                                [3.25553199, 3.5141327, 3.77273342]])
        self.assertAlmostEquals(rel_error(out, correct_out), 1e-9)

    def testFullBackward(self):
        np.random.seed(231)
        x = np.random.randn(2, 1, 3)
        w = np.random.randn(3, 2)
        b = np.random.randn(2)
        dout = np.random.randn(2, 2)
        dx_correct = np.array([[[0.77763186, -2.09243622, -0.30026834]],
                                [[1.67825251, -0.12366044, 0.31302074]]])
        dw_correct = np.array([[0.66055649, -0.41124988],
                               [1.8682554, 0.66652153],
                               [-2.25925226, -1.62638125]])
        db_correct = [1.18121661,  1.4141844]
        _, cache = full_forward(x, w, b)
        dx, dw, db = full_backward(dout, cache)
        self.assertLess(rel_error(dx, dx_correct), 1e-5)
        self.assertLess(rel_error(dw, dw_correct), 1e-5)
        self.assertLess(rel_error(db, db_correct), 1e-5)

    def testReluForward(self):
        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

        out, _ = relu_forward(x)
        correct_out = np.array([[0., 0., 0., 0., ],
                                [0., 0., 0.04545455, 0.13636364, ],
                                [0.22727273, 0.31818182, 0.40909091, 0.5, ]])
        self.assertAlmostEquals(rel_error(out, correct_out), 1e-9)

    def testReluBackward(self):
        np.random.seed(231)
        x = np.random.randn(3, 4)
        dout = np.random.randn(*x.shape)

        dx_correct = [[-0.83732373, 0.95218767, 0, 0],
                  [ 0, 0, 0, 0.99109251],
                  [ 0, 0, 0, 0.90101716]]
        _, cache = relu_forward(x)
        dx = relu_backward(dout, cache)
        self.assertLess(rel_error(dx_correct, dx), 1e-5)

    def testBatchNormForward(self):
        np.random.seed(231)
        N, D1, D2, D3 = 200, 50, 60, 3
        X = np.random.randn(N, D1)
        W1 = np.random.randn(D1, D2)
        W2 = np.random.randn(D2, D3)
        a = np.maximum(0, X.dot(W1)).dot(W2)

        gamma = np.asarray([1.0, 2.0, 3.0])
        beta = np.asarray([11.0, 12.0, 13.0])
        a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
        self.assertLess(rel_error(a_norm.mean(axis=0), beta), 1e-8)
        self.assertLess(rel_error(a_norm.std(axis=0), gamma), 1e-8)

    def testBatchNormBackward(self):
        np.random.seed(231)
        N, D = 2, 2
        x = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)
        dx_correct = np.array([[2.42034003e-09, 1.78371813e-08], [-2.42034003e-09, -1.78371813e-08]])
        dgamma_correct = np.array([-1.08325373, -0.67126421])
        dbeta_correct = np.array([-1.76780469, -0.08144906])
        _, cache = batchnorm_forward(x, gamma, beta, {'mode':'train'})
        dx, dgamma, dbeta = batchnorm_backward(dout, cache)
        self.assertLess(rel_error(dx_correct, dx), 4e-1)
        self.assertLess(rel_error(dgamma_correct, dgamma), 1e-8)
        self.assertLess(rel_error(dbeta_correct, dbeta_correct), 1e-8)

    def testDropoutForward(self):
        prob = 0.3
        np.random.seed(231)
        x = np.random.randn(500, 500) + 10
        out_train, _ = dropout_forward(x, {'mode': 'train', 'p': prob})
        out_test, _ = dropout_forward(x, {'mode': 'test', 'p': prob})
        self.assertLess(abs(prob-(out_train == 0).mean()), 1e-3)
        self.assertLess(abs(0-(out_test == 0).mean()), 1e-3)

    def testDropoutBackward(self):
        np.random.seed(231)
        x = np.random.randn(2, 2) + 10
        dout = np.random.randn(*x.shape)
        dropout_param = {'mode': 'train', 'p': 0.5, 'seed': 123}
        out, cache = dropout_forward(x, dropout_param)
        dx = dropout_backward(dout, cache)
        dx_num = [[-0.14945063, 0], [0, 3.72345805]]
        self.assertLess(rel_error(dx, dx_num), 1e-8)

    def testConvForward(self):
        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)

        conv_param = {'stride': 2, 'pad': 1}
        out, _ = conv_forward(x, w, b, conv_param)
        correct_out = np.array([[[[-0.08759809, -0.10987781],
                                  [-0.18387192, -0.2109216]],
                                 [[0.21027089, 0.21661097],
                                  [0.22847626, 0.23004637]],
                                 [[0.50813986, 0.54309974],
                                  [0.64082444, 0.67101435]]],
                                [[[-0.98053589, -1.03143541],
                                  [-1.19128892, -1.24695841]],
                                 [[0.69108355, 0.66880383],
                                  [0.59480972, 0.56776003]],
                                 [[2.36270298, 2.36904306],
                                  [2.38090835, 2.38247847]]]])
        self.assertLess(rel_error(out, correct_out), 1e-7)

    def testConvBackward(self):
        np.random.seed(231)
        x = np.random.randn(2, 2, 3, 3)
        w = np.random.randn(2, 2, 2, 2)
        b = np.random.randn(3, )
        dout = np.random.randn(2, 2, 6, 6)
        conv_param = {'stride': 1, 'pad': 2}
        dx_correct = np.array([
            [
                [
                    [-0.99586749, 2.17409167, -0.45352415], [-1.41899788, -0.64510689, 2.07242211],
                    [-2.53884708, -0.82629111, 0.78561257]
                ],
                [
                    [2.88860475, -0.42936398, 2.62903049], [1.40539801, 0.75107387, 0.16949314],
                    [1.20343246, 3.00942677, -3.29852985]
                ]
            ],
            [
                [
                    [0.12246593, -5.26511368, -0.97937742], [5.39620988, -3.1473564, 2.77271701],
                    [1.91213639, -1.78738795, 5.53291789]
                ],
                [
                    [-0.20174922, 5.17612768, 2.07536866], [5.99681263, 1.39950946, 1.13419772],
                    [2.84893118, 0.82971516, 2.88537637]
                ]
            ]
        ])
        dw_correct = np.array([[[[-4.6241439, 1.8094091], [2.73723036, 1.93393013]],
               [[-2.96136066, -1.71379998], [-4.22428831,1.91391067]]],
              [[[-2.3100721, -0.26151352], [-0.14307245, -1.20981533]],
              [[2.07924848, -2.76650744], [0.03612067, -0.63962934]]]])
        db_correct = np.array([-26.3931733, -23.88900438, 0])

        out, cache = conv_forward(x, w, b, conv_param)
        dx, dw, db = conv_backward(dout, cache)
        self.assertLess(rel_error(dx_correct, dx), 1e-7)
        self.assertLess(rel_error(dw_correct, dw), 1e-7)
        self.assertLess(rel_error(db_correct, db), 1e-10)

    def testMaxPoolForward(self):
        x_shape = (2, 3, 4, 4)
        x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
        pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

        out, _ = max_pool_forward(x, pool_param)

        correct_out = np.array([[[[-0.26315789, -0.24842105],
                                  [-0.20421053, -0.18947368]],
                                 [[-0.14526316, -0.13052632],
                                  [-0.08631579, -0.07157895]],
                                 [[-0.02736842, -0.01263158],
                                  [0.03157895, 0.04631579]]],
                                [[[0.09052632, 0.10526316],
                                  [0.14947368, 0.16421053]],
                                 [[0.20842105, 0.22315789],
                                  [0.26736842, 0.28210526]],
                                 [[0.32631579, 0.34105263],
                                  [0.38526316, 0.4]]]])

        self.assertLess(rel_error(out, correct_out), 1e-7)

    def testMaxPoolBackward(self):
        np.random.seed(231)
        x = np.random.randn(2, 2, 3, 3)
        dout = np.random.randn(2, 2, 2, 2)
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 1}
        out, cache = max_pool_forward(x, pool_param)
        dx_correct = [[[[0, -0.72618213, 0], [0, 0, 0], [0, -1.99893926, 0]],
               [[0, 0, 0],[0, -1.9069208, 0.55304798],[0, 0, 0]]],
              [[[0, 1.07324902, 0], [1.27292534, 0, 1.58102968], [0, 0, 0]],
               [[0, 0, 0], [0, -2.94423441, 0], [0, 0, 0]]]]
        dx = max_pool_backward(dout, cache)
        self.assertLess(rel_error(dx_correct, dx), 1e-8)

    def testSpatialBatchNormForward(self):
        np.random.seed(231)
        N, C, H, W = 2, 3, 4, 5
        x = 4 * np.random.randn(N, C, H, W) + 10

        gamma, beta = np.ones(C), np.zeros(C)
        bn_param = {'mode': 'train'}
        out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
        self.assertLess(rel_error(0, out.mean(axis=(0, 2, 3))), 1e-7)
        self.assertLess(rel_error(1, out.std(axis=(0, 2, 3))), 1e-6)

    def testRNNStepForward(self):
        N, D, H = 3, 10, 4
        x = np.linspace(-0.4, 0.7, num=N * D).reshape(N, D)
        prev_h = np.linspace(-0.2, 0.5, num=N * H).reshape(N, H)
        Wx = np.linspace(-0.1, 0.9, num=D * H).reshape(D, H)
        Wh = np.linspace(-0.3, 0.7, num=H * H).reshape(H, H)
        b = np.linspace(-0.2, 0.4, num=H)

        next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
        expected_next_h = np.asarray([
            [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
            [0.66854692, 0.79562378, 0.87755553, 0.92795967],
            [0.97934501, 0.99144213, 0.99646691, 0.99854353]])
        self.assertLess(rel_error(next_h, expected_next_h), 1e-8)

    def testRNNStepBackward(self):
        np.random.seed(231)
        N, D, H = 2, 1, 2
        x = np.random.randn(N, D)
        h = np.random.randn(N, H)
        Wx = np.random.randn(D, H)
        Wh = np.random.randn(H, H)
        b = np.random.randn(H)
        out, cache = rnn_step_forward(x, h, Wx, Wh, b)
        dnext_h = np.random.randn(*out.shape)

        dx_correct = [[0.02999498], [0.02260007]]
        dprev_h_correct = [[-0.17540737, -0.0323028], [0.13645769, 0.034438]]
        dWx_correct = [[-0.08672269, 0.01671559]]
        dWh_correct = [[-0.20050225, -0.0458132], [-0.00737577, -0.0214218]]
        db_correct = [0.01945126, 0.02981573]
        dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)

        self.assertLess(rel_error(dx, dx_correct), 1e-6)
        self.assertLess(rel_error(dprev_h, dprev_h_correct), 1e-6)
        self.assertLess(rel_error(dWx, dWx_correct), 1e-6)
        self.assertLess(rel_error(dWh, dWh_correct), 1e-6)
        self.assertLess(rel_error(db, db_correct), 1e-6)

    def testWordEmbeddingForward(self):
        N, T, V, D = 2, 4, 5, 3
        x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
        W = np.linspace(0, 1, num=V * D).reshape(V, D)
        out, _ = word_embedding_forward(x, W)
        expected_out = np.asarray([
            [[0., 0.07142857, 0.14285714],
             [0.64285714, 0.71428571, 0.78571429],
             [0.21428571, 0.28571429, 0.35714286],
             [0.42857143, 0.5, 0.57142857]],
            [[0.42857143, 0.5, 0.57142857],
             [0.21428571, 0.28571429, 0.35714286],
             [0., 0.07142857, 0.14285714],
             [0.64285714, 0.71428571, 0.78571429]]])

        self.assertLess(rel_error(expected_out, out), 1e-7)

    def testWordEmbeddingBackward(self):
        np.random.seed(231)
        N, T, V, D = 2, 1, 2, 1
        x = np.random.randint(V, size=(N, T))
        W = np.random.randn(V, D)
        dW_correct = [[-0.52653556], [0.93210596]]
        out, cache = word_embedding_forward(x, W)
        dout = np.random.randn(*out.shape)
        dW = word_embedding_backward(dout, cache)
        self.assertLess(rel_error(dW, dW_correct), 1e-7)

    def testSequentialFull(self):
        np.random.seed(231)

        # Gradient check for temporal affine layer
        N, T, D, M = 2, 1, 1, 2
        x = np.random.randn(N, T, D)
        w = np.random.randn(D, M)
        b = np.random.randn(M)

        out, cache = sequential_full_forward(x, w, b)
        dout = np.random.randn(*out.shape)
        dx, dw, db = sequential_full_backward(dout, cache)
        dx_correct = [[[-1.05212045]], [[2.81263098]]]
        dw_correct = [[-2.05421443,  0.25228933]]
        db_correct = [-1.5753272, 1.48537232]

        self.assertLess(rel_error(dx_correct, dx), 1e-7)
        self.assertLess(rel_error(dw_correct, dw), 1e-7)
        self.assertLess(rel_error(db_correct, db), 1e-7)

    def testSoftmax(self):
        np.random.seed(231)
        num_classes, num_inputs = 10, 50
        x = 0.001 * np.random.randn(num_inputs, num_classes)
        y = np.random.randint(num_classes, size=num_inputs)
        loss, _ = softmax_loss(x, y)
        self.assertLess(loss - 2.3, 1e-2)

    def testSequentialSoftmax(self):
        np.random.seed(231)
        N, T, V = 2, 1, 2
        x = np.random.randn(N, T, V)
        y = np.random.randint(V, size=(N, T))
        mask = (np.random.rand(N, T) > 0.5)
        loss, dx = sequential_softmax_loss(x, y, mask)
        dx_correct = [[[0.13652956, -0.13652956]], [[-0, 0]]]
        self.assertLess(rel_error(dx, dx_correct), 1e-7)


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.data = get_cifar_data()
        self.data['X_train'] = self.data['X_train'][:4000]
        self.data['y_train'] = self.data['y_train'][:4000]
        np.random.seed(3)

    def testOptimizers(self):
        optimizer_updates = ['sgd', 'sgd_momentum', 'rmsprop', 'adam']
        lrs = [1e-2, 1e-2, 1e-4, 1e-3]
        losses = []
        for optim, lr in zip(optimizer_updates, lrs):
            model = FullyConnectedNet([100, 100, 100, 100, 100], std=5e-2)
            optimizer = Optimizer(model, self.data, num_epochs=5,
                                  batch_size=100, optim=optim,
                                  optim_config={'learning_rate': lr})
            optimizer.train()
            losses.append(min(optimizer.losses))
        sgd_loss, sgdm_loss, rms_loss, adam_loss = losses
        self.assertLess(sgd_loss, 1.5)
        self.assertLess(sgdm_loss, 1.3)
        self.assertLess(rms_loss, 1.25)
        self.assertLess(adam_loss, 1.1)


class TestFullyConnectedNet(unittest.TestCase):
    def setUp(self):
        self.data = get_cifar_data()
        np.random.seed(2)
        self.model = FullyConnectedNet([100, 150, 175], reg=1e-5, std=1e-3)

    def testModel(self):
        optimizer = Optimizer(self.model, self.data, num_epochs=20, batch_size=500, optim='adam',
                              optim_config={'learning_rate': 1e-4}, num_val=1000)
        optimizer.train()
        _, val_acc = optimizer.evaluate()
        self.assertGreater(val_acc, 0.52)


class TestThreeLayerConvNet(unittest.TestCase):
    def setUp(self):
        self.data = get_conv_cifar_data()
        self.model = ThreeLayerConvNet(std=0.001, hidden_dim=500, filter_size=3)

    def testModel(self):
        optimizer = Optimizer(self.model, self.data,
                        num_epochs=8, batch_size=100,
                        optim='adam',
                        optim_config={'learning_rate': 1e-1})
        optimizer.train()
        _, val_acc = optimizer.evaluate()
        self.assertGreater(val_acc, 0.65)


class TestRNN(unittest.TestCase):
    def testModel(self):
        N, D, W, H = 10, 20, 30, 40
        word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
        V = len(word_to_idx)
        T = 13

        model = ImageCaptionRNN(word_to_idx,
                              input_dim=D,
                              wordvec_dim=W,
                              hidden_dim=H,
                              dtype=np.float64)

        for k, v in model.params.items():
            model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)

        features = np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D)
        captions = (np.arange(N * T) % V).reshape(N, T)

        loss, grads = model.loss(features, captions)
        expected_loss = 0.819362992502

        self.assertLess(abs(loss - expected_loss), 1e-10)


def rel_error(x, y):
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
