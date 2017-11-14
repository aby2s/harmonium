import tensorflow as tf
import collections

GibbsSample = collections.namedtuple('GibbsSample', 'visible hidden')
GibbsChain = collections.namedtuple('GibbsChain', 'start end')
CostUpdate = collections.namedtuple('CostUpdate', 'energy weight_update visible_bias_update hidden_bias_update')

def sample(probability):
    shape = tf.shape(probability)
    return tf.where(probability - tf.random_uniform(shape) > 0.0,
                    tf.ones(shape), tf.zeros(shape))