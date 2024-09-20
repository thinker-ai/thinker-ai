from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def batch_invert_permutation(permutations):
    """返回每个批次的逆置换序列。

    Args:
        permutations: `Tensor`，形状为 `[batch_size, dim]`，表示每个批次的置换序列。

    Returns:
        `Tensor`，形状为 `[batch_size, dim]`，表示每个批次置换序列的逆序列。
    """
    with tf.name_scope('batch_invert_permutation'):
        permutations = tf.cast(permutations, tf.int32)
        return tf.map_fn(tf.math.invert_permutation, permutations)

def batch_gather(values, indices):
    """对每个批次执行 `tf.gather` 操作。

    Args:
        values: `Tensor`，形状为 `[batch_size, seq_length, ...]`。
        indices: `Tensor`，形状为 `[batch_size, indices_length]`，指定要提取的索引。

    Returns:
        `Tensor`，形状为 `[batch_size, indices_length, ...]`，提取的值。
    """
    with tf.name_scope('batch_gather'):
        return tf.gather(values, indices, batch_dims=1)


def one_hot(length, index):
    """返回长度为 `length` 的 one-hot 编码向量，在 `index` 位置为 1，其余位置为 0。

    Args:
        length: 整数，表示向量的长度。
        index: 整数，表示 1 所在的位置。

    Returns:
        `ndarray`，长度为 `length` 的 one-hot 向量。
    """
    result = np.zeros(length)
    result[index] = 1
    return result


def reduce_prod(x, axis, name="reduce_prod"):
    """在指定轴上计算乘积。

    Args:
        x: `Tensor`，要计算乘积的张量。
        axis: 整数，指定计算乘积的轴。
        name: 操作名称（可选）。

    Returns:
        `Tensor`，沿指定轴的乘积结果。
    """
    with tf.name_scope(name):
        return tf.reduce_prod(x, axis=axis)
