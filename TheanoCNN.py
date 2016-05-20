# Much of this code is taken verbatim from tutorials on the Theano website.

from __future__ import print_function
from __future__ import division

import os
import sys
import timeit

import numpy as np
rng = np.random.RandomState(23455)


import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from TheanoExtras import LogisticRegression, HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
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
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

class LeNet():
    def __init__(self, image_size, get_training_batch, nkerns=[20, 50],
                 filter_diam = 9, maxpool_size = 2,
                 learning_rate=0.1, batch_size=500):
        """
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_batches: int
        :param n_batches: maximal number of batches to run the optimizer

        :type get_training_batch: function
        :param get_training_batch: function that returns tuples of the form (images, labels).
        Its sole input is batch_size. See the documentation of
        AstroImageMunger.get_batch for more details.

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
        self.filter_diam = filter_diam
        self.batch_size = batch_size
        self.filter_diam = filter_diam
        self.maxpool_size = maxpool_size
        self.rng = np.random.RandomState(23455)
        self.batch_size = batch_size
        self.get_training_batch = get_training_batch

        print('... building the model')
        #=========  Set up Theano basics  =========

        #Symbolic input to Theano
        self.x = T.dtensor4("x")
        self.y = T.ivector("y")

        #set up two convolutional pooling layers
        self.layer0, image_size = self.do_conv_pool(self.x,             image_size, nkern = nkerns[0], nkern_prev = image_size[2])
        self.layer1, image_size = self.do_conv_pool(self.layer0.output, image_size, nkern = nkerns[1], nkern_prev = nkerns[0])
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        self.layer2_input = self.layer1.output.flatten(2)
        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            rng,
            input=self.layer2_input,
            n_in=nkerns[1] * image_size[0] * image_size[1],
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=2)
        # the cost we minimize during training is the NLL of the model
        batch_template_xy = self.get_training_batch(batch_size = batch_size)

        #=========  Set up Theano equipment specific to training =========

        self.cost = self.layer3.negative_log_likelihood(self.y)
        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        self.updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, self.grads)
        ]
        self.train_set_x_T = theano.shared(batch_template_xy[0])
        self.train_set_y_T = theano.shared(batch_template_xy[1])
        self.train_model = theano.function(
            [],
            self.cost,
            updates=self.updates,
            givens={self.x:self.train_set_x_T, self.y:self.train_set_y_T}
        )

        return

    def fit(self, n_batches):
        start_time = timeit.default_timer()
        for i in range(n_batches):
            print("Training batch ", i, " of ", n_batches)
            train_set_x, train_set_y = self.get_training_batch(batch_size = self.batch_size)
            self.train_set_x_T.set_value(train_set_x)
            self.train_set_y_T.set_value(train_set_y)
            self.train_model()
        end_time = timeit.default_timer()
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
        return

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
        return int(np.ceil((original - self.filter_diam + 1) / self.maxpool_size))

    def do_conv_pool(self, layer0_input, image_size, nkern, nkern_prev):
        layer0 = LeNetConvPoolLayer(
                rng,
                input=layer0_input,
                image_shape=(self.batch_size, nkern_prev, image_size[0], image_size[1]),
                filter_shape=(nkern, nkern_prev, self.filter_diam, self.filter_diam),
                poolsize=(2, 2)
                )
        for i in range(2):
            image_size[i] = self.update_image_size(image_size[i])
        return layer0, image_size