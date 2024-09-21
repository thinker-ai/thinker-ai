#thinker_ai/agent/memory/humanoid_memory/dnc_new/util.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def batch_invert_permutation(permutations):
    with tf.name_scope('batch_invert_permutation'):
        permutations = tf.cast(permutations, tf.int32)
        return tf.map_fn(tf.math.invert_permutation, permutations)


def batch_gather(values, indices):
    with tf.name_scope('batch_gather'):
        return tf.gather(values, indices, batch_dims=1)


def one_hot(length, index):
    result = np.zeros(length)
    result[index] = 1
    return result


def reduce_prod(x, axis, name="reduce_prod"):
    with tf.name_scope(name):
        return tf.reduce_prod(x, axis=axis)
