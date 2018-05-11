import tensorflow as tf

class Regularizer(object):
    def __init__(self, l):
        self.l = l

    def __call__(self, model, learnable):
        raise NotImplementedError


class L2(Regularizer):
    """
    Creates L2 regularizer
    :param l: regularization coefficient
    :return:
    """
    def __init__(self, l):
        super(L2, self).__init__(l)

    def __call__(self, model, learnable):
        return learnable.assign_add(- self.l * learnable)


class L1(Regularizer):
    """
    Creates L1 regularizer
    :param l: regularization coefficient
    :return:
    """
    def __init__(self, l):
        super(L1, self).__init__(l)

    def __call__(self, model, learnable):
        return learnable.assign(tf.where(tf.abs(learnable) > self.l, learnable - self.l * tf.sign(learnable),
                                            tf.zeros(tf.shape(learnable))))


class SparsityTarget(Regularizer):
    def __init__(self, l, p):
        """
        Creates sparsity target regularizer
        :param l: regularization coefficient
        :param p: sparsity target
        :return:
        """
        super(SparsityTarget, self).__init__(l)
        self.p = p

    def __call__(self, model, learnable):
        q = tf.reduce_mean(model.hidden.call(model.input, model.W), 0)
        return learnable.assign(tf.add(learnable, self.l * (self.p-q)))