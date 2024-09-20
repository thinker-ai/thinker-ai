"""Tests for utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc_new import util
from thinker_ai.agent.memory.humanoid_memory.dnc_new.util import reduce_prod, one_hot


class BatchInvertPermutation(tf.test.TestCase):

    def test(self):
        # Tests that the _batch_invert_permutation function correctly inverts a
        # batch of permutations.
        batch_size = 5
        length = 7

        permutations = np.empty([batch_size, length], dtype=int)
        for i in range(batch_size):
            permutations[i] = np.random.permutation(length)

        inverse = util.batch_invert_permutation(tf.constant(permutations, tf.int32))

        # No need for session in TF 2.x, use eager execution
        inverse = inverse.numpy()

        for i in range(batch_size):
            for j in range(length):
                self.assertEqual(permutations[i][inverse[i][j]], j)


class BatchGather(tf.test.TestCase):

    def test(self):
        values = np.array([[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 7]])
        indexs = np.array([[1, 2, 0, 3], [3, 0, 1, 2], [0, 2, 1, 3]])
        target = np.array([[1, 4, 3, 1], [6, 5, 9, 2], [5, 5, 3, 7]])

        result = util.batch_gather(tf.constant(values), tf.constant(indexs))

        # Use eager execution
        result = result.numpy()

        self.assertAllEqual(target, result)


class OneHotTest(tf.test.TestCase):

    def testOneHot(self):
        length = 10
        index = 3
        result = one_hot(length, index)
        expected_result = np.zeros(length)
        expected_result[index] = 1
        self.assertAllEqual(result, expected_result)

    def testIndexOutOfBounds(self):
        length = 10
        index = 15  # 超出边界的索引
        with self.assertRaises(IndexError):
            _ = one_hot(length, index)


class ReduceProdTest(tf.test.TestCase):

    def testReduceProd(self):
        x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
        axis = 1  # 沿着 axis=1 计算乘积
        result = reduce_prod(x, axis)
        expected_result = np.array([6, 120], dtype=np.float32)  # 期望结果

        # 直接获取结果
        result_eval = result.numpy()  # 使用 .numpy() 获取结果
        np.testing.assert_allclose(result_eval, expected_result)

    def testReduceProdEmpty(self):
        x = tf.constant([], dtype=tf.float32)
        axis = 0
        result = reduce_prod(x, axis)

        # 检查空输入的结果是否为标量值1
        result_eval = result.numpy()
        self.assertEqual(result_eval, 1.0)  # 验证结果为1


if __name__ == '__main__':
    tf.test.main()
