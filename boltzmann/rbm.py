import tensorflow as tf
import numpy as np
import collections

from boltzmann.core import sample


class LatentVariablesModel(object):

    def __init__(self):
        self.session = None

    def _hidden_state(self, visible_state):
        pass

    def _visible_state(self, hidden_state):
        pass

    def burn_in(self, visible_state=None, hidden_state=None, n=1):
        assert n > 0, 'Number of steps to burn in should be greater than zero'
        if hidden_state is None:
            hidden_state = self._hidden_state(visible_state)
        burned_in_hidden_state = hidden_state
        for i in range(n):
            burned_in_visible_state = self._hidden_state(burned_in_hidden_state)
            burned_in_hidden_state = self._hidden_state(burned_in_visible_state)
        return [burned_in_visible_state, burned_in_hidden_state]

    def hidden_state(self, visible_state):
        return self._get_session().run(self._hidden_state(visible_state))

    def visible_state(self, hidden_state):
        return self._get_session().run(self._visible_state(hidden_state))

    def __call__(self, input_state, **kwargs):
        return self.visible_state(self._hidden_state(input_state))

    def _get_session(self):
        return self.session

    def _set_session(self, session):
        self.session = session

class EnrgyBasedModel(LatentVariablesModel):
    def _energy(self):
        pass

class RBMLayer(object):
    activations = {'sigmoid': tf.nn.sigmoid, 'linear': None, 'relu': tf.nn.relu}

    def __init__(self, units,
                 activation=None,
                 use_bias=False,
                 bias=None,
                 name=None,
                 sampled=False):
        """

        :param units: int, number of units
        :param activation: string, 'sigmoid', 'linear' or 'relu'
        :param use_bias: boolean, flag to use bias, if false bias set to zero and never updated
        :param bias: 1d-array, bias initial value, if None, bias initialized with zeros
        :param name: string, layer name
        :param sampled: boolean,
        """
        self.gaussian = True
        self.units = units
        self.use_bias = use_bias
        self.default_sampled = sampled
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros([units]) if bias is None else bias,
                                    name=None if name is None else name + '_bias')

        if activation in self.activations:
            self.activation = self.activations[activation]
            if activation == 'sigmoid':
                self.gaussian = False
        else:
            raise ValueError('Unknown activation identifier {}'.format(activation))

        self.session = None

    def call(self, input, weights, transpose_weights=False, sampled=None):
        sampled = self.default_sampled if sampled is None else sampled

        if not self.gaussian and sampled:
            raise ValueError('Sampling is available only for logistic units')

        kernel = tf.matmul(input, weights, transpose_b=transpose_weights)
        if self.use_bias:
            kernel = tf.add(kernel, self.bias)

        output = kernel if self.activation is None else self.activation(kernel)
        if sampled:
            output = sample(output)

        return output



    def get_bias(self):
        return self.session.run(self.bias)






class RBMModel(object):
    def __init__(self, visible, hidden, weights=None, weights_stddev=0.01):
        """
        :param visible: RBMLayer, visible layer
        :param hidden: RBMLayer, hidden layer
        :param weights: 2d-array, weights for initialization
        :param weights_stddev: float, if weights aren't provided, RBM weights are initialized with
        gaussian random values with mean=0 and stddev=weights_stddev
        """
        self.hidden = hidden
        self.visible = visible

        if weights is None:
            self.W = tf.Variable(
                tf.random_normal([self.visible.units, self.hidden.units], mean=0.0, stddev=weights_stddev,
                                 name="weights"))
        else:
            self.W = tf.Variable(weights,
                                 name="weights")

    def energy(self, visible_state, hidden_state):
        energy = -tf.reduce_sum(
            tf.reduce_mean(tf.multiply(tf.matmul(visible_state, self.W), hidden_state), axis=0))

        if self.visible.use_bias:
            if self.visible.gaussian:
                v = visible_state - self.visible.bias
                free_energy = tf.add(energy,  tf.reduce_mean(tf.reduce_sum(tf.multiply(v, v) / 2, axis=1)))
            else:
                free_energy = tf.add(energy, -tf.reduce_mean(
                    tf.reduce_sum(tf.multiply(self.visible.bias, visible_state), axis=1)))

        if self.hidden.use_bias:
            if self.hidden.gaussian:
                h = hidden_state - self.hidden.bias
                free_energy = tf.add(energy, tf.reduce_mean(tf.reduce_sum(tf.multiply(h, h) / 2, axis=1)))
            else:
                free_energy = tf.add(energy, -tf.reduce_mean(
                    tf.reduce_sum(tf.multiply(self.hidden.bias, hidden_state), axis=1)))

        return energy

    #f gibbs_sample(self, x, n=1, sampled=None):
    #  visible = x
    #     hidden = self.hidden.call(visible, self.W, sampled=sampled)
    #     start = GibbsSample(visible=visible, hidden=hidden)
    #     for i in range(n):
    #         visible = self.visible.call(hidden, self.W, transpose_weights=True,
    #                                     sampled=sampled)
    #         hidden = self.hidden.call(visible, self.W, sampled=sampled)
    #
    #     end = GibbsSample(visible=visible, hidden=hidden)
    #     return GibbsChain(start=start, end=end)

    def burn_in(self, visible_state=None, hidden_state=None, n=1):
        assert n > 0, 'Number of steps to burn in should be greater than zero'
        if hidden_state is None:
            hidden_state = self.hidden.call(visible_state)
        burned_in_hidden_state = hidden_state
        for i in range(n):
            burned_in_visible_state = self.visible.call(burned_in_hidden_state)
            burned_in_hidden_state = self.hidden.call(burned_in_visible_state)
        return [burned_in_visible_state, burned_in_hidden_state]

    def get_weights(self):
        """
        :return: 2d-array, RBM weights
        """
        return self.session.run(self.W)

    def compile(self, optimizer,
                metrics=None, config=None, unstack=False, kernel_regularizer=None, bias_regularizer=None):
        """
        :param optimizer: optimizer instance, supports only cd instance
        :param metrics: unsupported
        :param config: config to initialize TensorFlow session
        :param unstack: boolean. This option allows to train very large RBMs. You can switch it to true, if you get
        OOM. Never do it otherwise, because it makes training really slow.
        :param kernel_regularizer: available l1/l2 regularizers or None
        :param bias_regularizer: available l1/l2 regularizers or None
        """
        self.optimizer = optimizer(self)
        if config is not None:
            self.session = tf.Session()
        else:
            self.session = tf.Session(config=config)

        self.unstack = unstack
        self.visible.session = self.session
        self.hidden.session = self.session

        self.input = tf.placeholder("float", [None, self.visible.units], name='input')
        [free_energy, weight_update, visible_bias_update, hidden_bias_update] = self.optimizer.get_cost_update(
            self.input)

        # weight_update = tf.add(weight_update, kernel_regularizer(self.W))



        if unstack:
            self.batch_size = tf.placeholder("float", [], name='batch_size')
            self.w_gradient = tf.Variable(tf.zeros([self.visible.units, self.hidden.units]), name="gradient")
            self.reset_w_gradient = self.w_gradient.assign(tf.zeros([self.visible.units, self.hidden.units]))
            self.partial_w_update = self.w_gradient.assign_add(weight_update)
            self.update = self.W.assign_add(tf.div(self.w_gradient, self.batch_size))

            if self.hidden.use_bias:
                self.hb_gradient = tf.Variable(tf.zeros([self.hidden.units]), name="gradient_hb")
                self.reset_hb_gradient = self.hb_gradient.assign(tf.zeros([self.hidden.units]))
                self.partial_hb_update = self.hb_gradient.assign_add(hidden_bias_update)
                self.update_hb = self.hidden.bias.assign_add(tf.div(self.hb_gradient, self.batch_size))
            if self.visible.use_bias:
                self.vb_gradient = tf.Variable(tf.zeros([self.visible.units]), name="gradient_vb")
                self.reset_vb_gradient = self.vb_gradient.assign(tf.zeros([self.visible.units]))
                self.partial_vb_update = self.vb_gradient.assign_add(visible_bias_update)
                self.update_vb = self.visible.bias.assign_add(tf.div(self.vb_gradient, self.batch_size))

        else:
            self.update = self.W.assign_add(weight_update)
            if self.hidden.use_bias:
                self.update_hb = self.hidden.bias.assign_add(hidden_bias_update)
            if self.visible.use_bias:
                self.update_vb = self.visible.bias.assign_add(visible_bias_update)

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


        self.cost = free_energy
        self.session.run(tf.global_variables_initializer())

    def fit(self, x, batch_size=32, nb_epoch=10, verbose=1, validation_data=None, shuffle=True):
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

        if self.unstack:
            session_run = [self.partial_w_update, self.cost]
            if self.visible.use_bias:
                session_run.append(self.partial_vb_update)
            if self.hidden.use_bias:
                session_run.append(self.partial_hb_update)
        else:
            session_run = [self.update, self.cost]
            if self.visible.use_bias:
                session_run.append(self.update_vb)
            if self.hidden.use_bias:
                session_run.append(self.update_hb)

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
            for batch_indices in batches:
                if self.unstack:
                    for index in range(batch_indices[0], batch_indices[1]):
                        batch = x[index]
                        partial_cost = self.session.run(session_run, feed_dict={self.input: [batch]})[1]
                        free_energy += partial_cost
                    free_energy /= batch_size
                    self.session.run(self.update,
                                     feed_dict={self.batch_size: float(batch_indices[1] - batch_indices[0])})
                    self.session.run(self.reset_w_gradient)
                    if self.hidden.use_bias:
                        self.session.run(self.update_hb,
                                         feed_dict={self.batch_size: float(batch_indices[1] - batch_indices[0])})
                        self.session.run(self.reset_hb_gradient)
                    if self.visible.use_bias:
                        self.session.run(self.update_vb,
                                         feed_dict={self.batch_size: float(batch_indices[1] - batch_indices[0])})
                        self.session.run(self.reset_vb_gradient)
                else:
                    batch = x[index_array[batch_indices[0]:batch_indices[1]]]
                    free_energy = self.session.run(session_run, feed_dict={self.input: batch})[1]

                if verbose == 1:
                    self.log('{}/{} free energy: {}'.format(batch_indices[1], len(x), free_energy))
            if verbose > 0:
                self.log('Epoch complete, last free energy {}'.format(free_energy))

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
        visible = self.gibbs_sample(self.input, n, sampled=sampled).end.visible
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
