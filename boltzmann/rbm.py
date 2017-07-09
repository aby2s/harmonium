import tensorflow as tf
import numpy as np
import collections

GibbsSample = collections.namedtuple('GibbsSample', 'visible hidden')
GibbsChain = collections.namedtuple('GibbsChain', 'start end')
CostUpdate = collections.namedtuple('CostUpdate', 'energy weight_update visible_bias_update hidden_bias_update')


class CD(object):
    def __init__(self, model, n=1, lr=0.1):
        self.model = model
        self.n = n
        self.lr = lr
        self.tensorArr = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)

    def get_cost(self, state_multi, visible_state, hidden_state):
        free_energy = tf.reduce_sum(
            tf.reduce_mean(tf.multiply(state_multi, self.model.W), axis=0))

        if self.model.visible.use_bias:
            if self.model.visible.probabilistic:
                free_energy = tf.add(free_energy, tf.reduce_mean(
                    tf.reduce_sum(tf.multiply(self.model.visible.bias, visible_state), axis=1)))
            else:
                v = visible_state - self.model.visible.bias
                free_energy = tf.add(free_energy, - tf.reduce_mean(tf.reduce_sum(tf.multiply(v, v) / 2, axis=1)))

        if self.model.hidden.use_bias:
            if self.model.visible.probabilistic:
                free_energy = tf.add(free_energy, tf.reduce_mean(
                    tf.reduce_sum(tf.multiply(self.model.hidden.bias, hidden_state), axis=1)))
            else:
                h = hidden_state - self.model.hidden.bias
                free_energy = tf.add(free_energy, -tf.reduce_mean(tf.reduce_sum(tf.multiply(h, h) / 2, axis=1)))

        return -free_energy

    def multiply_states(self, visible, hidden):
        return tf.matmul(tf.expand_dims(visible, 2),
                         tf.expand_dims(hidden, 1))

    def get_cost_update(self, visible):
        gibbs_chain = self.model.gibbs_sample(visible, n=self.n)
        # hidden = states[0][1]
        # visible = states[0][0]
        # hg = states[1][1]
        # vg = states[1][0]
        # -tf.reduce_sum(tf.reduce_mean(tf.multiply(tf.matmul(visible, self.model.W), hidden), 0))
        chain_start_multi = self.multiply_states(gibbs_chain.start.visible, gibbs_chain.start.hidden)
        chain_end_multi = self.multiply_states(gibbs_chain.end.visible, gibbs_chain.end.hidden)
        return CostUpdate(energy=self.get_cost(chain_start_multi, gibbs_chain.start.visible, gibbs_chain.start.hidden),
                          weight_update=tf.reduce_mean(chain_start_multi - chain_end_multi, 0) * self.lr,
                          visible_bias_update=tf.reduce_mean(gibbs_chain.start.visible - gibbs_chain.end.visible,
                                                             0) * self.lr,
                          hidden_bias_update=tf.reduce_mean(gibbs_chain.start.hidden - gibbs_chain.end.hidden,
                                                            0) * self.lr)


def cd(n=1, lr=0.1):
    return lambda model: CD(model, n=n, lr=lr)


class RBMLayer(object):
    activations = {'sigmoid': tf.nn.sigmoid, 'linear': None, 'relu': tf.nn.relu}

    def __init__(self, units,
                 activation=None,
                 use_bias=False,
                 bias=None,
                 name=None,
                 sampled=False):
        self.probabilistic = False
        self.units = units
        self.use_bias = use_bias
        self.default_sampled = sampled
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros([units]) if bias is None else bias,
                                    name=None if name is None else name + '_bias')

        if activation in self.activations:
            self.activation = self.activations[activation]
            if activation == 'sigmoid':
                self.probabilistic = True
        else:
            raise ValueError('Unknown activation identifier {}'.format(activation))

        self.session = None

    def call(self, input, weights, transpose_weights=False, sampled=None):
        sampled = self.default_sampled if sampled is None else sampled

        if not self.probabilistic and sampled:
            raise ValueError('Sampling is available only for logistic units')

        kernel = tf.matmul(input, weights, transpose_b=transpose_weights)
        if self.use_bias:
            kernel = tf.add(kernel, self.bias)

        output = kernel if self.activation is None else self.activation(kernel)
        if sampled:
            output = self.sample(output)

        return output

    def sample(self, probability):
        shape = tf.shape(probability)
        return tf.where(probability - tf.random_uniform(shape) > 0.0,
                        tf.ones(shape), tf.zeros(shape))

    def get_bias(self):
        return self.session.run(self.bias)


def l2(l):
    return lambda W, grad: W.assign_add(grad - 2 * l * W)


def _l1(W, grad, l):
    w_update = W + grad
    return W.assign(tf.where(tf.abs(w_update) > l, w_update - l * tf.sign(W), tf.zeros(tf.shape(W))))


def l1(l):
    return lambda W, grad: _l1(W, grad, l)


class RBMModel(object):
    def __init__(self, visible, hidden, weights=None, weights_stddev=0.01):
        self.hidden = hidden
        self.visible = visible

        if weights is None:
            self.W = tf.Variable(
                tf.random_normal([self.visible.units, self.hidden.units], mean=0.0, stddev=weights_stddev,
                                 name="weights"))
        else:
            self.W = tf.Variable(weights,
                                 name="weights")

    def gibbs_sample(self, x, n=1, sampled=None):
        visible = x
        hidden = self.hidden.call(visible, self.W, sampled=sampled)
        start = GibbsSample(visible=visible, hidden=hidden)
        for i in range(n):
            visible = self.visible.call(hidden, self.W, transpose_weights=True,
                                        sampled=sampled)
            hidden = self.hidden.call(visible, self.W, sampled=sampled)

        end = GibbsSample(visible=visible, hidden=hidden)
        return GibbsChain(start=start, end=end)

    def get_weights(self):
        return self.session.run(self.W)

    def compile(self, optimizer,
                metrics=None, config=None, unstack=False, kernel_regularizer=None, bias_regularizer=None):
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

        if bias_regularizer is not None:
            visible_bias_update = tf.add(visible_bias_update,
                                         bias_regularizer(self.visible.bias))
            hidden_bias_update = tf.add(hidden_bias_update, bias_regularizer(self.hidden.bias))

        if unstack:
            self.batch_size = tf.placeholder("float", [], name='batch_size')
            self.w_gradient = tf.Variable(tf.zeros([self.visible.units, self.hidden.units]), name="gradient")
            self.reset_w_gradient = self.w_gradient.assign(tf.zeros([self.visible.units, self.hidden.units]))
            self.partial_w_update = kernel_regularizer(self.w_gradient,
                                                       weight_update) if kernel_regularizer is not None else self.w_gradient.assign_add(
                weight_update)
            self.update = self.W.assign_add(tf.div(self.w_gradient, self.batch_size))

            if self.hidden.use_bias:
                self.hb_gradient = tf.Variable(tf.zeros([self.hidden.units]), name="gradient_hb")
                self.reset_hb_gradient = self.hb_gradient.assign(tf.zeros([self.hidden.units]))
                self.partial_hb_update = bias_regularizer(self.hb_gradient,
                                                          hidden_bias_update) if bias_regularizer is not None \
                    else self.hb_gradient.assign_add(hidden_bias_update)

                self.update_hb = self.hidden.bias.assign_add(tf.div(self.hb_gradient, self.batch_size))
            if self.visible.use_bias:
                self.vb_gradient = tf.Variable(tf.zeros([self.visible.units]), name="gradient_vb")
                self.reset_vb_gradient = self.vb_gradient.assign(tf.zeros([self.visible.units]))
                self.partial_vb_update = bias_regularizer(self.vb_gradient,
                                                          visible_bias_update) if bias_regularizer is not None \
                    else self.vb_gradient.assign_add(visible_bias_update)
                self.update_vb = self.visible.bias.assign_add(tf.div(self.vb_gradient, self.batch_size))

        else:
            self.update = kernel_regularizer(self.W,
                                             weight_update) if kernel_regularizer is not None else self.W.assign_add(
                weight_update)
            if self.hidden.use_bias:
                self.update_hb = bias_regularizer(self.W, hidden_bias_update) if bias_regularizer is not None \
                    else self.hidden.bias.assign_add(hidden_bias_update)
            if self.visible.use_bias:
                self.update_vb = bias_regularizer(self.W, visible_bias_update) if bias_regularizer is not None \
                    else self.visible.bias.assign_add(visible_bias_update)

        self.cost = free_energy
        self.session.run(tf.global_variables_initializer())

    def fit(self, x, batch_size=32, nb_epoch=10, verbose=1, validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):
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

        samples_num = len(x)
        index_array = np.arange(samples_num)

        batches_num = int(len(x) / batch_size) + (1 if len(x) % batch_size > 0 else 0)

        free_energy = 0
        for j in range(nb_epoch):
            if verbose > 0:
                self.log("Epoch {}/{}, free energy {}", j + 1, nb_epoch, free_energy)

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
                    print('{}/{} free energy: {}'.format(batch_indices[1], len(x), free_energy))

        if verbose > 0:
            print('Fitting completed')

    def generate(self, x, n=1, sampled=None):
        visible = self.gibbs_sample(self.input, n + 1, sampled=sampled).end.visible
        return self.session.run(visible, feed_dict={self.input: x})

    def hidden_state(self, x, n=1, sampled=None):
        hidden = self.hidden.call(self.input, self.W, sampled=sampled)
        return self.session.run(hidden, feed_dict={self.input: x})

    def log(self, str, *args):
        print(str.format(*args))
