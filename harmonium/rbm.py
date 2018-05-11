import tensorflow as tf
import numpy as np
import collections
import pickle
from harmonium.core import sample_bernoulli, sample_gaussian


class RBMLayer(object):
    activations = {'sigmoid': tf.nn.sigmoid, 'linear': None, 'relu': tf.nn.relu}
    #samplers = {'sigmoid': sample_bernoulli, 'linear': sample_gaussian, 'relu': lambda x: sample_gaussian(relu=True)}
    samplers = {'sigmoid': sample_bernoulli, 'linear': lambda x: x, 'relu': lambda x: x}

    def __init__(self, units,
                 activation=None,
                 use_bias=False,
                 bias=None,
                 sampled=False):
        """

        :param units: int, number of units
        :param activation: string, 'sigmoid', 'linear' or 'relu'
        :param use_bias: boolean, flag to use bias, if false bias set to zero and never updated
        :param bias: 1d-array, bias initial value, if None, bias initialized with zeros
        :param name: string, layer name
        :param sampled: boolean,
        """
        self.units = units
        self.use_bias = use_bias
        self.default_sampled = sampled
        self.bias_initializer = bias


        if activation in self.activations:
            self.activation = self.activations[activation]
            self.sampler = self.samplers[activation]
            self.binary = activation == "sigmoid"
        else:
            raise ValueError('Unknown activation identifier {}'.format(activation))

        self.session = None

    def initialize(self, session, name='layer'):
        self.session = session
        self.name = name
        if self.use_bias:
            self.bias = tf.get_variable(name=name + '_bias', shape=(self.units,),
                                        initializer=tf.zeros_initializer() if self.bias_initializer is None
                                        else tf.constant_initializer(self.bias_initializer))

    def call(self, input, weights, transpose_weights=False, sampled=None):
        sampled = self.default_sampled if sampled is None else sampled

        kernel = tf.matmul(input, weights, transpose_b=transpose_weights, name='{}_kernel'.format(self.name))
        if self.use_bias:
            kernel = tf.add(kernel, self.bias)

        return self.nonlinearity(kernel, sampled)

    def nonlinearity(self, kernel, sampled):
        output = kernel if self.activation is None else self.activation(kernel, name='{}_activation'.format(self.name))
        if sampled:
            output = self.sampler(output)
        return output



    def get_bias(self):
        return self.session.run(self.bias)






class RBMModel(object):
    def __init__(self, visible, hidden, session=None, weights=None, weights_stddev=0.01, scope='RBM'):
        """
        :param visible: RBMLayer, visible layer
        :param hidden: RBMLayer, hidden layer
        :param weights: 2d-array, weights for initialization
        :param weights_stddev: float, if weights aren't provided, RBM weights are initialized with
        gaussian random values with mean=0 and stddev=weights_stddev
        """

        self.session = tf.Session() if session is None else session
        self.hidden = hidden
        self.visible = visible

        with tf.variable_scope(scope) as scope:
            self.W = tf.get_variable(name='weights', shape=(self.visible.units, self.hidden.units),
                                     initializer=tf.random_normal_initializer(stddev=weights_stddev) if weights is None
                                     else tf.constant_initializer(weights))

            self.visible.initialize(self.session, name='visible')
            self.hidden.initialize(self.session, name='hidden')
            self.input = tf.placeholder("float", [None, self.visible.units], name='input')

        self.scope = scope



    def energy(self, visible_state, hidden_state, scope='energy'):
        with tf.variable_scope(scope):
            visible_state = tf.stop_gradient(visible_state, name="visible_state")
            hidden_state = tf.stop_gradient(hidden_state, name="hidden_state")
            energy = -tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.matmul(visible_state, self.W, name='visible_weights'),
                                                               hidden_state, name='weights_hidden')
                                                   , axis=1, name='energy_sum'), name="batch_energy_mean")

            if self.visible.use_bias:
                if self.visible.binary:
                    energy = tf.add(energy, -tf.reduce_mean(
                        tf.reduce_sum(tf.multiply(self.visible.bias, visible_state, name='visible_bias_energy'), axis=1)))
                else:
                    v = visible_state - self.visible.bias
                    energy = tf.add(energy,  tf.reduce_mean(tf.reduce_sum(tf.multiply(v, v) / 2, axis=1)))


            if self.hidden.use_bias:
                if self.hidden.binary:
                    energy = tf.add(energy, -tf.reduce_mean(
                        tf.reduce_sum(tf.multiply(self.hidden.bias, hidden_state, name='hidden_bias_energy'), axis=1)))
                else:
                    h = hidden_state - self.hidden.bias
                    energy = tf.add(energy, tf.reduce_mean(tf.reduce_sum(tf.multiply(h, h) / 2, axis=1)))

        return energy

    def burn_in(self, visible_state=None, hidden_state=None, n=1, sampled=None):
        assert n > 0, 'Number of steps to burn in should be greater than zero'
        if hidden_state is None:
            hidden_state = self.hidden.call(visible_state,  self.W, sampled=sampled)
        burned_in_hidden_state = hidden_state
        for i in range(n+1):
            burned_in_visible_state = self.visible.call(burned_in_hidden_state, self.W, transpose_weights=True,
                                                        sampled=sampled)
            burned_in_hidden_state = self.hidden.call(burned_in_visible_state, self.W, sampled=sampled)
        return [burned_in_visible_state, burned_in_hidden_state]

    def get_weights(self):
        """
        :return: 2d-array, RBM weights
        """
        return self.session.run(self.W)



    def compile(self, optimizer,
                metrics=None, kernel_regularizer=None, bias_regularizer=None):
        """
        :param optimizer: optimizer instance, supports only cd instance
        :param metrics: unsupported
        :param config: config to initialize TensorFlow session
        :param unstack: boolean. This option allows to train very large RBMs. You can switch it to true, if you get
        OOM. Never do it otherwise, because it makes training really slow.
        :param kernel_regularizer: available l1/l2 regularizers or None
        :param bias_regularizer: available l1/l2 regularizers or None
        """


        self.summary_writer = tf.summary.FileWriter('./summary', self.session.graph)

        with tf.variable_scope(self.scope):
            with tf.name_scope(self.scope.original_name_scope):
                self.optimizer = optimizer(self)

                [energy, update] = self.optimizer.get_cost_update(
                    self.input)

                self.energy_val = energy
                self.cost_update, self.cost = tf.metrics.mean(energy)
                self.update = update

                if kernel_regularizer:
                    self.kernel_regularizer = kernel_regularizer(self, self.W)
                else:
                    self.kernel_regularizer = None

                if bias_regularizer is not None:
                    self.visible_bias_regularizer = bias_regularizer(self, self.visible.bias)
                    self.hidden_bias_regularizer = bias_regularizer(self, self.hidden.bias)
                else:
                    self.visible_bias_regularizer = None
                    self.hidden_bias_regularizer = None

        self.kernel_regularizer = None
        self.visible_bias_regularizer = None
        self.hidden_bias_regularizer = None

        self.session.run(tf.global_variables_initializer())

    def fit(self, x, batch_size=32, nb_epoch=10, verbose=1, validation_data=None, shuffle=False):
        """
        Do RBM fitting on provided training set
        :param x: 2d-array, training set
        :param batch_size: int, minibatch size
        :param nb_epoch: int, number of epochs
        :param verbose: 0 for no output, 1 for output per minibatch, 2 for output per epoch
        :param validation_data: 2d-array, validation data (unused right now)
        :param shuffle: boolean, flag to shuffle training data every epoch
        """
        if verbose > 0:
            print("Fitting RBM on {} samples with {} batch size and {} epochs".format(len(x), batch_size, nb_epoch))

        session_run = [self.update, self.cost_update, self.cost]

        if self.kernel_regularizer is not None:
            session_run.append(self.kernel_regularizer)

        if self.visible_bias_regularizer is not None:
            session_run.append(self.visible_bias_regularizer)

        if self.hidden_bias_regularizer is not None:
            session_run.append(self.hidden_bias_regularizer)


        samples_num = len(x)
        index_array = np.arange(samples_num)

        batches_num = int(len(x) / batch_size) + (1 if len(x) % batch_size > 0 else 0)

        for j in range(nb_epoch):
            if verbose > 0:
                self.log("Epoch {}/{}", j + 1, nb_epoch)

            if shuffle:
                np.random.shuffle(index_array)

            batches = [(i * batch_size, min(samples_num, (i + 1) * batch_size)) for i in range(0, batches_num)]
            free_energy = 0
            self.session.run([tf.local_variables_initializer()])
            for batch_indices in batches:
                batch = x[index_array[batch_indices[0]:batch_indices[1]]]

                res = self.session.run(session_run, feed_dict={self.input: batch})

                free_energy = res[2]

                if verbose == 1:
                    self.log('{}/{} free energy: {}'.format(batch_indices[1], len(x), free_energy))
            if verbose > 0:
                self.log('Epoch complete, free energy {}'.format(free_energy))

        if verbose > 0:
            self.log('Fitting completed')

    def generate(self, x, n=1, sampled=None):
        """
        Returns visible state after applying n Gibbs sampling steps
        :param x: 2d-array, visible unit states
        :param n: int, number of Gibbs sampling steps
        :param sampled: boolean, if true, do sampling from units states
        :return: 2d-array, generated visible state
        """
        visible, _ = self.burn_in(self.input, n=n, sampled=sampled)
        return self.session.run(visible, feed_dict={self.input: x})

    def hidden_state(self, x, sampled=None):
        """
        Returns hidden state after applying n Gibbs sampling steps
        :param x: 2d-array, visible unit states
        :param n: int, number of Gibbs sampling steps
        :param sampled: boolean, if true, do sampling from units states
        :return: 2d-array, generated visible state
        """
        hidden = self.hidden.call(self.input, self.W, sampled=sampled)
        return self.session.run(hidden, feed_dict={self.input: x})

    def log(self, str, *args):
        print(str.format(*args))
