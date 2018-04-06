import tensorflow as tf
import collections

def sample_bernoulli(probability):
    # return tf.where(probability - tf.random_uniform(shape) > 0.0,
    #                 tf.ones(shape), tf.zeros(shape))
    return tf.where(probability > 0.5,
                    tf.ones_like(probability), tf.zeros_like(probability))

def sample_gaussian(mean, stddev=1.0, relu=False):
    return tf.nn.relu(mean+tf.random_normal(tf.shape(mean), stddev=stddev))

