# Much of this code is taken verbatim from tutorials on the Theano website.

from __future__ import print_function
from __future__ import division

import os
import sys
import timeit

import numpy as np
import pickle as pkl
rng = np.random.RandomState(23455)

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from TheanoExtras import LogisticRegression, HiddenLayer
from theano.tensor.nnet import softplus

import matplotlib.pyplot as plt

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = softplus(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

class LeNet():
    def __init__(self, image_size = None, get_training_batch = None, nkerns=[20, 50],
                 filter_diam = 12, maxpool_size = 4, lambduh = 0.01,
                 batch_size = 20, path = None, mode = "full"):
        """
        :type  learning_rate: float
        :param learning_rate: initial learning rate (learning rate decays over time )

        :type n_batches: int
        :param n_batches: maximal number of batches to run the optimizer

        :type get_training_batch: function
        :param get_training_batch: function that returns tuples of the form (images, labels).
        Its sole input is batch_size. See the documentation of
        AstroImageMunger.get_batch for more details.
        Note: rather than a hasNext() type of interface, get_batch is assumed to have infinite capacity,
        i.e. automatically cycle back through the data once it reaches the end.

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
        if not path is None:
            with open(path, "rb") as f:
                loaded_dict = pkl.load(f)

            image_size = loaded_dict["init_image_size"]
            get_training_batch = None
            nkerns = loaded_dict["nkerns"]
            filter_diam = loaded_dict["filter_diam"]
            maxpool_size = loaded_dict["maxpool_size"]
            lambduh = loaded_dict["lambduh"]
            batch_size = loaded_dict["batch_size"]
            mode = "unknown"

        else:
            assert not any((image_size is None, get_training_batch is None))

        self.init_image_size = list(image_size)
        self.filter_diam = filter_diam
        self.batch_size = batch_size
        self.filter_diam = filter_diam
        self.maxpool_size = maxpool_size
        self.nkerns = nkerns
        self.rng = np.random.RandomState(23455)
        self.batch_size = batch_size
        self.get_training_batch = get_training_batch
        self.lambduh = lambduh
        self.mode = mode
        self.learning_rate = 0.1

        print('... building the model')
        #=========  Set up Theano basics  =========

        #Symbolic input to Theano
        self.x = T.dtensor4("x")
        self.y = T.ivector("y")



        #set up two convolutional pooling layers
        self.layer0, image_size = self.do_conv_pool(self.x,             image_size,
                                                    nkern = self.nkerns[0], nkern_prev = image_size[2])
        self.layer1, image_size = self.do_conv_pool(self.layer0.output, image_size,
                                                    nkern = self.nkerns[1], nkern_prev = nkerns[0])

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        self.layer2_input = self.layer1.output.flatten(2)
        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            rng,
            input=self.layer2_input,
            n_in=nkerns[1] * image_size[0] * image_size[1],
            n_out=500,
            activation=softplus
        )

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=2)
        # the cost we minimize during training is the NLL of the model

        # create a list of all model parameters to be fit by gradient descent
        self.param_arrays = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        self.param_names = ["l3w", "l3b", "l2w", "l2b", "l1w", "l1b", "l0w", "l0b"]
        self.weight_arrays = [self.layer3.W,
                              self.layer2.W, self.layer1.W, self.layer0.W,
                              self.layer2.b, self.layer1.b, self.layer0.b]

        #print([w.eval().shape for w in self.weight_arrays])
        #print([np.prod(w.eval().shape) for w in self.weight_arrays])

        #penalized loss function
        self.err = self.layer3.negative_log_likelihood(self.y) / 1000
        self.penalty = self.lambduh * T.sum([T.sum(w ** 2) for w in self.weight_arrays])
        self.cost = self.penalty + self.err

        # create a list of gradients
        self.grads = T.grad(self.cost, self.param_arrays)

        #=========  Set up Theano equipment specific to training =========
        if path is None:
            # train_model is a function that updates the model parameters by
            # SGD. Since this model has many parameters, it would be tedious to
            # manually create an update rule for each model parameter. We thus
            # create the updates list by automatically looping over all
            # (params[i], grads[i]) pairs.
            self.iter = theano.shared(1)
            self.train_updates = [
                (param_i, param_i - (self.learning_rate / self.iter) * grad_i)
                for param_i, grad_i in zip(self.param_arrays, self.grads)
            ]
            batch_template_xy = self.get_training_batch(batch_size = batch_size)
            self.train_set_x_T = theano.shared(batch_template_xy[0])
            self.train_set_y_T = theano.shared(batch_template_xy[1])
            self.train_model = theano.function(
                [],
                [],
                updates=self.train_updates,
                givens={self.x:self.train_set_x_T, self.y:self.train_set_y_T}
            )

            self.train_verbose = theano.function(
                [],
                [self.cost, self.err, self.penalty],
                updates=self.train_updates,
                givens={self.x:self.train_set_x_T, self.y:self.train_set_y_T}
            )

        #========= Loading =========
        if not path is None:
            loaded_param_arrays = loaded_dict["param_arrays"]
            self.loading_updates = [
                (w, theano.shared(w_load))
                for w, w_load in zip(self.param_arrays, loaded_param_arrays)
                ]

            load_model = theano.function( [], [], updates=self.loading_updates)
            load_model()

        return

    def save(self, net_path):
        #reveal = theano.function([input], output)
        print("Saving net. When this is loaded again, it will not be capable of training.")
        dict_to_save = {
            "init_image_size":self.init_image_size,
            "nkerns":self.nkerns,
            "filter_diam":self.filter_diam,
            "maxpool_size":self.maxpool_size,
            "lambduh":self.lambduh,
            "batch_size":self.batch_size,

            "param_arrays":[w.eval() for w in self.param_arrays]
        }
        with open(net_path, "wb") as f:
            pkl.dump(file = f, obj = dict_to_save)
        return

    def plot(self):
        plt.clf()
        width = 2
        height = int(np.ceil(len(self.param_arrays) / width))
        for i, w in enumerate(self.param_arrays):
            plt.subplot(height, width, i + 1)
            self.plot_utility(w.eval(), self.param_names[i])
        plt.show()
        return

    def plot_utility(self, w, name):
        if w.shape == (2,):
            plt.plot((w[0], -w[1]))
        elif len(w.shape) == 1:
            plt.plot(range(len(w)), w)
        elif len(w.shape) == 2 and w.shape[1] == 2:
            plt.plot(range(w.shape[0]), w[:, 1])
        else:
            first_half_dims = int(np.product(w.shape[0:int(np.floor(len(w.shape) / 2.0))]))
            plt.imshow(w.reshape(first_half_dims, -1))
        plt.title( name)# + ", shape = " + str(w.shape) + \
                  #", (max, min) = " + str((np.max(w), -np.max(-w))) )
        return

    def fit(self, n_batches):
        start_time = timeit.default_timer()
        cum_costs = [0, 0]
        cum_errs = [0, 0]
        cum_penalties = [0, 0]
        for i in range(n_batches):
            train_set_x, train_set_y = self.get_training_batch(batch_size = self.batch_size)
            self.train_set_x_T.set_value(train_set_x)
            self.train_set_y_T.set_value(train_set_y)
            self.iter.set_value(i + 1)
            if i % 5 == 0:
                print("Training batch ", i, " of ", n_batches, "; batch_size = ", self.batch_size)
                print("First 5 labels :", train_set_y[0:5], "first pixel:", train_set_x[0, 0, 0, 0])
                cost, err, penalty = self.train_verbose()
                cum_costs.append(cost)
                cum_errs.append(err)
                cum_penalties.append(penalty)
                cum_costs[-1] += cum_costs[-2]
                cum_errs[-1] += cum_errs[-2]
                cum_penalties[-1] += cum_penalties[-2]
                print('Cost, error, penalty on this batch is ', cost, err, penalty)
            else:
                self.train_model()
        end_time = timeit.default_timer()
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
        return np.array(cum_costs), np.array(cum_errs), np.array(cum_penalties)

    def predict_proba(self, X):
        """
        :param X: numpy array of RGB images with shape (num_test_examples, 3, 96, 96)
        or rather (num_test_examples, image_size[2], image_size[0], image_size[1])
        :return: numpy array of size num_test_examples containing probabilities.
        """
        n_examples= X.shape[0]

        #pad x with zeros so the last batch is complete
        pad_len = self.batch_size - (n_examples % self.batch_size)
        pad_shape = list(X.shape)
        pad_shape[0] = pad_len
        X = np.concatenate([X, np.zeros(pad_shape)])

        #set up Theano function
        self.test_set_x_T = theano.shared(X[0:self.batch_size, :, :, :])
        self.test_model = theano.function(
            inputs=[],
            outputs=self.layer3.p_y_given_x,
            givens={self.x: self.test_set_x_T}
        )

        #run test in batches
        p = []
        num_test_batches = int(np.ceil(n_examples / self.batch_size))
        for i in range(num_test_batches):
            indices = (i * self.batch_size, (i + 1) *  self.batch_size)
            self.test_set_x_T = theano.shared(X[indices[0]:indices[1], :, :, :])
            p.extend(self.test_model())

        #strip off the padding
        p = np.array(p[0:n_examples])
        return np.array(p)

    def update_image_size(self, original):
        return int(np.floor((original - self.filter_diam + 1) / self.maxpool_size))

    def do_conv_pool(self, layer0_input, image_size, nkern, nkern_prev):
        layer0 = LeNetConvPoolLayer(
                rng,
                input=layer0_input,
                image_shape=(self.batch_size, nkern_prev, image_size[0], image_size[1]),
                filter_shape=(nkern, nkern_prev, self.filter_diam, self.filter_diam),
                poolsize=(self.maxpool_size, self.maxpool_size)
                )
        for i in range(2):
            image_size[i] = self.update_image_size(image_size[i])
            assert image_size[i] <= 96
        return layer0, image_size


