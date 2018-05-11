import tensorflow as tf
import collections

def sample_bernoulli(probability):
    return tf.where(probability - tf.random_uniform(tf.shape(probability)) > 0.0,
                    tf.ones_like(probability), tf.zeros_like(probability))

def sample_gaussian(mean, stddev=1.0, relu=False):
    sample = mean+tf.random_normal(tf.shape(mean), stddev=stddev)
    if relu:
        return tf.nn.relu(sample)
    else:
        return sample

