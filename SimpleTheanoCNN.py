# Much of this code is taken verbatim from tutorials on the Theano website.

from __future__ import print_function
from __future__ import division

import os
import sys
import timeit

import numpy as np
import pickle as pkl
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from TheanoExtras import LogisticRegression
from theano.tensor.nnet import softplus

import warnings
import matplotlib.pyplot as plt
rng = np.random.RandomState(23455)



class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize, fixed_filter):
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
        if fixed_filter is None:
            self.W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = theano.shared(
                np.asarray(
                    fixed_filter,
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
    def __init__(self, image_size = None, get_training_batch = None, nkerns=[1,1],
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
        warnings.warn("In SimpleTheanoCNN, ignoring number of kernels.")
        nkerns = [1, 1]
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
            self.fixed_filter = loaded_dict["fixed_filter"]

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

        if path is None:
            assert not any((image_size is None, get_training_batch is None))
            self.fixed_filter = self.initialize_filter()

        print('... building the model')
        #=========  Set up Theano basics  =========

        #Symbolic input to Theano
        self.x = T.dtensor4("x")
        self.y = T.ivector("y")



        #set up convolutional pooling layer
        self.layer0, image_size = self.do_conv_pool(self.x,             image_size,
                                                    nkern = self.nkerns[1], nkern_prev = image_size[2],
                                                    fixed_filter = self.fixed_filter)

        n_into_lr = self.nkerns[1] * np.prod(image_size[0:2])
        print(image_size)
        print(n_into_lr)
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        self.layer3_input = self.layer0.output.flatten(2)
        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=self.layer3_input, n_in=n_into_lr, n_out=2)

        # create a list of all model parameters to be fit by gradient descent
        warnings.warn("Leaving weights fixed in SimpleTheanoCNN.")
        self.param_arrays = self.layer3.params + [self.layer0.b]
        self.param_names = ["l3w", "l3b", "l0b"]
        self.weight_arrays = [self.layer3.W,
                              self.layer0.W,
                              self.layer0.b]

        #print([w.eval().shape for w in self.weight_arrays])
        #print([np.prod(w.eval().shape) for w in self.weight_arrays])

        #penalized loss function
        self.cost = self.layer3.negative_log_likelihood(self.y) / 1000

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
                (param_i, param_i - (self.learning_rate / (1 + self.iter * self.lambduh) * grad_i))
                for param_i, grad_i in zip(self.param_arrays, self.grads)
            ]
            batch_template_xy = self.get_training_batch(batch_size = self.batch_size)
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
                [self.cost],
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

    def save(self, net_path, filename):
        #reveal = theano.function([input], output)
        print("Saving net. When this is loaded again, it will not be capable of training.")
        dict_to_save = {
            "init_image_size":self.init_image_size,
            "nkerns":self.nkerns,
            "filter_diam":self.filter_diam,
            "maxpool_size":self.maxpool_size,
            "lambduh":self.lambduh,
            "batch_size":self.batch_size,
            "fixed_filter":self.fixed_filter,
            "param_arrays":[w.eval() for w in self.param_arrays]
        }
        if not os.path.exists(net_path):
            os.mkdir(net_path)
        with open(os.path.join(net_path, filename), "wb") as f:
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
        cum_costs = []
        cum_errs = []
        cum_penalties = []
        for i in range(n_batches):
            train_set_x, train_set_y = self.get_training_batch(batch_size = self.batch_size)
            self.train_set_x_T.set_value(train_set_x)
            self.train_set_y_T.set_value(train_set_y)
            self.iter.set_value(i + 1)
            if i % 5 == 0:
                print("Training batch ", i, " of ", n_batches, "; batch_size = ", self.batch_size)
                print("First 5 labels :", train_set_y[0:5], "first pixel:", train_set_x[0, 0, 0, 0])
                cost, err, penalty = self.train_verbose()[0], 0, 0
                cum_costs.append(cost)
                cum_errs.append(err)
                cum_penalties.append(penalty)
                if len(cum_costs) >= 2:
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

    def do_conv_pool(self, layer0_input, image_size, nkern, nkern_prev, fixed_filter = None):
        layer0 = LeNetConvPoolLayer(
                rng,
                input=layer0_input,
                image_shape=(self.batch_size, nkern_prev, image_size[0], image_size[1]),
                filter_shape=(nkern, nkern_prev, self.filter_diam, self.filter_diam),
                poolsize=(self.maxpool_size, self.maxpool_size),
                fixed_filter = fixed_filter
                )
        for i in range(2):
            image_size[i] = self.update_image_size(image_size[i])
            assert image_size[i] <= 96
        return layer0, image_size


    def initialize_filter(self, num_samples = 10000):
        """
        Computes filters of the form \mu \Sigma^{-1} , where
        \mu is the sample mean and \Sigma the sample covariance
        of randomly drawn patches from the dataset.
        """
        print("initializing filters to whitened mean of data")
        images = self.get_training_batch(batch_size = self.batch_size)[0]
        fd = self.filter_diam
        filter_num_pixels = np.prod(fd*fd*3)
        mu = np.zeros(filter_num_pixels)
        sigma = np.zeros((filter_num_pixels, filter_num_pixels))
        X = np.zeros((num_samples, filter_num_pixels))
        # Get 10 patches from each image
        for i in range(num_samples):
            if i % 10 * self.batch_size == 0:
                images = self.get_training_batch(batch_size=self.batch_size)[0]
            p_x = np.random.choice(range(self.init_image_size[0] - fd))
            p_y = np.random.choice(range(self.init_image_size[1] - fd))
            X[i, :] = np.reshape(images[i % self.batch_size, :, p_x:(p_x + fd), p_y:(p_y + fd)], filter_num_pixels)

        mu = np.sum(X, axis = 0)
        mu = mu / num_samples
        for i in range(num_samples):
            X[i, :] = X[i, :] - mu
        sigma = X.T.dot(X) / num_samples + 0.01 * np.identity(filter_num_pixels)

        fixed_filter = np.linalg.solve(sigma, mu)
        fixed_filter = np.reshape(fixed_filter, (fd, fd, 3))
        fixed_filter = fixed_filter.transpose((2,1,0))
        fixed_filter = np.expand_dims(fixed_filter, axis=0)
        return fixed_filter


