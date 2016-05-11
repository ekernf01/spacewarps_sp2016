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
    def __init__(self, filter_diam = 9, maxpool_size = 2):
        self.filter_diam = filter_diam
        self.maxpool_size = maxpool_size
        return

    def update_image_size(self, original):
        return int(np.ceil((original - self.filter_diam + 1) / self.maxpool_size))

    def do_conv_pool(self, layer0_input, image_size, batch_size, nkern, nkern_prev):
        layer0 = LeNetConvPoolLayer(
                rng,
                input=layer0_input,
                image_shape=(batch_size, nkern_prev, image_size[0], image_size[1]),
                filter_shape=(nkern, nkern_prev, self.filter_diam, self.filter_diam),
                poolsize=(2, 2)
                )
        for i in range(2):
            image_size[i] = self.update_image_size(image_size[i])

        return layer0, image_size

    def eval_lenet5(self, get_batch, image_size,
                    learning_rate=0.1, n_batches=20,
                    nkerns=[20, 50], batch_size=500):
        """
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type next_datum: function
        :param next_datum: function that returns tuples of the form (image, label)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """

        rng = np.random.RandomState(23455)

        ######################
        # BUILD MODEL        #
        ######################
        print('... building the model')

        # Form input of shape (batch_size, 1, image_size[0], image_size[1])

        test_set_x,  test_set_y  = get_batch(batch_size = batch_size, CV_type="test")

        #Feed it to Theano
        x = T.dtensor4("x")
        y = T.ivector("y")
        index = T.lscalar()  # index to a [mini]batch

        #set up two convolutional pooling layers
        layer0, image_size = self.do_conv_pool(x,             image_size, batch_size, nkern = nkerns[0], nkern_prev = image_size[2])
        layer1, image_size = self.do_conv_pool(layer0.output, image_size, batch_size, nkern = nkerns[1], nkern_prev = nkerns[0])

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * image_size[0] * image_size[1],
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

        print("...built layers. Initializing cost function.")

        # the cost we minimize during training is the NLL of the model
        cost = layer3.negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            inputs = [],
            outputs = layer3.errors(y),
            givens = {x: test_set_x, y: test_set_y}
        )

        # create a list of all model parameters to be fit by gradient descent
        params = layer3.params + layer2.params + layer1.params + layer0.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]
        train_set_x, train_set_y = get_batch(batch_size = batch_size, CV_type="train")
        train_set_x_T = theano.shared(train_set_x)
        train_set_y_T = theano.shared(train_set_y)
        train_model = theano.function(
            [],
            cost,
            updates=updates,
            givens={x:train_set_x_T, y:train_set_y_T}
        )
        # end-snippet-1

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')

        start_time = timeit.default_timer()
        for i in range(n_batches):
            train_set_x, train_set_y = get_batch(batch_size = batch_size, CV_type="train")
            train_set_x_T.set_value(train_set_x)
            train_set_y_T.set_value(train_set_y)
            train_model()
        test_loss = test_model()

        end_time = timeit.default_timer()
        print('Optimization complete. Test loss is: ', test_loss)
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

        return(test_loss)