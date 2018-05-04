import tensorflow as tf

class CD(object):
    def __init__(self, model, n=1, lr=0.1, momentum=None):
        self.model = model
        self.n = n
        self.lr = lr
        self.trace_data = dict()

        self.states = []

        self.momentum = momentum
        # if momentum:
        #     self.weight_velocity = None
        #     self.vb_velocity = None
        #     self.hb_velocity = None


    # def get_cost(self, state_multi, visible_state, hidden_state):
    #     free_energy = tf.reduce_sum(
    #         tf.reduce_mean(tf.multiply(state_multi, self.model.W), axis=0))
    #
    #     if self.model.visible.use_bias:
    #         if self.model.visible.probabilistic:
    #             free_energy = tf.add(free_energy, tf.reduce_mean(
    #                 tf.reduce_sum(tf.multiply(self.model.visible.bias, visible_state), axis=1)))
    #         else:
    #             v = visible_state - self.model.visible.bias
    #             free_energy = tf.add(free_energy, - tf.reduce_mean(tf.reduce_sum(tf.multiply(v, v) / 2, axis=1)))
    #
    #     if self.model.hidden.use_bias:
    #         if self.model.visible.probabilistic:
    #             free_energy = tf.add(free_energy, tf.reduce_mean(
    #                 tf.reduce_sum(tf.multiply(self.model.hidden.bias, hidden_state), axis=1)))
    #         else:
    #             h = hidden_state - self.model.hidden.bias
    #             free_energy = tf.add(free_energy, -tf.reduce_mean(tf.reduce_sum(tf.multiply(h, h) / 2, axis=1)))
    #
    #     return -free_energy
    #
    # def multiply_states(self, visible, hidden):
    #     return tf.matmul(tf.expand_dims(visible, 2),
    #                      tf.expand_dims(hidden, 1))
    #
    # def get_velocity(self, previous, g):
    #     return g if previous is None else previous * self.momentum + g

    def get_cost_update(self, sample):
        positive_visible_state = sample
        positive_hidden_state = self.model.hidden.call(positive_visible_state, self.model.W)
        negative_visible_state, negative_hidden_state = self.sample_negative(positive_visible_state, positive_hidden_state)
        data_energy = self.model.energy(positive_visible_state, positive_hidden_state, name='data')
        model_energy = self.model.energy(negative_visible_state, negative_hidden_state, name='model')
        loss = data_energy - model_energy

        if self.momentum:
            self.optimizer = tf.train.MomentumOptimizer(self.lr, self.momentum)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr, name='optimizer')

        # vars = [self.model.W]
        # if self.model.hidden.use_bias:
        #     vars.append(self.model.hidden.bias)
        #
        # if self.model.visible.use_bias:
        #     vars.append(self.model.visible.bias)
        grads = self.optimizer.compute_gradients(loss, gate_gradients=tf.train.Optimizer.GATE_GRAPH) #we need gradient ascent here
        for g,v in grads:
            self.trace_data["grad_{}".format(v.name)] = g


        self.trace_data["positive_visible_state"] = positive_visible_state
        self.trace_data["positive_hidden_state"] = positive_hidden_state
        self.trace_data["negative_visible_state"] = negative_visible_state
        self.trace_data["negative_hidden_state"] = negative_hidden_state
        self.trace_data["energy"] = data_energy
        self.trace_data["global_step"] = tf.Variable(initial_value = 0)

        update = self.optimizer.apply_gradients(grads, global_step = self.trace_data["global_step"])
        # update = self.optimizer.minimize(loss)
        #self.grads_and_vars = grads_and_vars
        return [data_energy, update]


        # negative_grad = self.multiply_states(negative_sample.end.visible, negative_sample.end.hidden)
        #
        # energy = self.get_cost(positive_grad, negative_sample.start.visible, negative_sample.start.hidden)
        #
        # weight_update = tf.reduce_mean(positive_grad - negative_grad, 0) * self.lr
        # visible_bias_update = tf.reduce_mean(negative_sample.start.visible - negative_sample.end.visible, 0) * self.lr
        # hidden_bias_update = tf.reduce_mean(negative_sample.start.hidden - negative_sample.end.hidden, 0) * self.lr
        # if self.momentum:
        #     self.weight_velocity = self.get_velocity(self.weight_velocity, weight_update)
        #     self.vb_velocity = self.get_velocity(self.vb_velocity, visible_bias_update)
        #     self.hb_velocity = self.get_velocity(self.hb_velocity, hidden_bias_update)
        #     return CostUpdate(
        #         energy=energy,
        #         weight_update=self.weight_velocity,
        #         visible_bias_update=self.vb_velocity,
        #         hidden_bias_update=self.hb_velocity)
        #
        # else:
        #     return CostUpdate(
        #         energy=energy,
        #         weight_update=weight_update,
        #         visible_bias_update=visible_bias_update,
        #         hidden_bias_update=hidden_bias_update)

    def sample_negative(self, visible, hidden):
        return self.model.burn_in(visible, hidden_state=hidden, n=self.n)


class PCD(CD):
    def __init__(self, model, n=1, lr=0.1, momentum=None):
        self.visible_negative = None
        self.hidden_negative = None

        super(PCD, self).__init__(model, n, lr, momentum)

    def sample_negative(self, visible, hidden):
        if self.visible_negative is None:
            self.visible_negative = self.model.hidden.nonlinearity(tf.random_normal(tf.shape(self.model.input)), sampled=True)


        [visible_negative, hidden_negative] = self.model.burn_in(self.visible_negative, hidden_state=self.hidden_negative, n=self.n)
        self.visible_negative = visible_negative
        self.hidden_negative = hidden_negative
        return  [visible_negative, hidden_negative]

def cd(n=1, lr=0.1, momentum=None):
    """
    Creates contrastive divergence optimizer
    :param lr: float, learning rate
    :param n: int, number of Gibbs sampling steps
    :return: contrastive divergence optimizer
    """
    return lambda model: CD(model, n=n, lr=lr, momentum=momentum)

def pcd(n=1, lr=0.1, momentum=None):
    """
    Creates contrastive divergence optimizer
    :param lr: float, learning rate
    :param n: int, number of Gibbs sampling steps
    :return: contrastive divergence optimizer
    """
    return lambda model: PCD(model, n=n, lr=lr, momentum=momentum)