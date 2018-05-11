import tensorflow as tf

class CD(object):
    def __init__(self, model, n=1, lr=0.1, momentum=None):
        self.model = model
        self.n = n
        self.lr = lr
        self.momentum = momentum

    def get_cost_update(self, sample):
        positive_visible_state = sample
        positive_hidden_state = self.model.hidden.call(positive_visible_state, self.model.W)
        negative_visible_state, negative_hidden_state = self.sample_negative(positive_visible_state, positive_hidden_state)
        data_energy = self.model.energy(positive_visible_state, positive_hidden_state, scope='data_energy')
        model_energy = self.model.energy(negative_visible_state, negative_hidden_state, scope='model_energy')
        loss = tf.subtract(data_energy, model_energy, name='loss')

        if self.momentum:
            self.optimizer = tf.train.MomentumOptimizer(self.lr, self.momentum)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr, name='optimizer')

        update = self.optimizer.minimize(loss)
        return [data_energy, update]

    def sample_negative(self, visible, hidden):
        return self.model.burn_in(visible, hidden_state=hidden, n=self.n)


class PCD(CD):
    def __init__(self, model, n=1, lr=0.1, momentum=None):
        self.visible_negative = None
        self.hidden_negative = None

        super(PCD, self).__init__(model, n, lr, momentum)

    def sample_negative(self, visible, hidden):
        if self.visible_negative is None:
            self.visible_negative = visible


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