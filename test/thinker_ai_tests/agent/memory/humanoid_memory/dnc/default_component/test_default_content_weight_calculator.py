"""Tests for CosineWeights."""

import numpy as np
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import DefaultContentWeightCalculator


class CosineWeightsTest(tf.test.TestCase):
    def test_cosine_weights_basic(self):
        """
        测试 CosineWeights 类的基本功能，包括输出形状、权重范围和归一化。
        """
        batch_size = 2
        memory_size = 4
        num_heads = 3
        word_size = 5

        # 创建 CosineWeights 实例，使用默认的 softmax 作为 strength_op
        cosine_weights = DefaultContentWeightCalculator(num_heads=num_heads, word_size=word_size)

        # 创建随机的 memory、keys 和 strengths
        memory = tf.random.uniform([batch_size, memory_size, word_size], minval=-1.0, maxval=1.0)
        keys = tf.random.uniform([batch_size, num_heads, word_size], minval=-1.0, maxval=1.0)
        strengths = tf.random.uniform([batch_size, num_heads], minval=0.1, maxval=2.0)

        # 调用 compute 方法
        weights = cosine_weights.compute(keys, strengths, memory)

        # 检查输出形状
        expected_shape = [batch_size, num_heads, memory_size]
        self.assertAllEqual(weights.shape, expected_shape)

        # 检查权重范围
        self.assertGreaterEqual(tf.reduce_min(weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(weights).numpy(), 1.0)

        # 检查权重在 memory_size 维度上的和为1
        sum_weights = tf.reduce_sum(weights, axis=-1)  # [batch_size, num_heads]
        self.assertAllClose(sum_weights.numpy(), np.ones_like(sum_weights.numpy()), atol=1e-6)

    def test_cosine_weights_arbitrary_batch_dims(self):
        """
        测试 CosineWeights 类在任意批次维度下的行为，例如 [batch_dim1, batch_dim2, ...]。
        """
        # 测试任意批次维度，例如 [batch_dim1, batch_dim2, ...]
        batch_dims = [3, 2]
        memory_size = 5
        num_heads = 4
        word_size = 6

        # 创建 CosineWeights 实例
        cosine_weights = DefaultContentWeightCalculator(num_heads=num_heads, word_size=word_size)

        # 创建随机的 memory、keys 和 strengths
        memory_shape = batch_dims + [memory_size, word_size]
        keys_shape = batch_dims + [num_heads, word_size]
        strengths_shape = batch_dims + [num_heads]

        memory = tf.random.uniform(memory_shape, minval=-1.0, maxval=1.0)
        keys = tf.random.uniform(keys_shape, minval=-1.0, maxval=1.0)
        strengths = tf.random.uniform(strengths_shape, minval=0.1, maxval=2.0)

        # 调用 compute 方法
        weights = cosine_weights.compute(keys, strengths, memory)

        # 检查输出形状
        expected_shape = batch_dims + [num_heads, memory_size]
        self.assertAllEqual(weights.shape, expected_shape)

        # 检查权重范围
        self.assertGreaterEqual(tf.reduce_min(weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(weights).numpy(), 1.0)

        # 检查权重在 memory_size 维度上的和为1
        sum_weights = tf.reduce_sum(weights, axis=-1)  # [batch_dims..., num_heads]
        self.assertAllClose(sum_weights.numpy(), np.ones_like(sum_weights.numpy()), atol=1e-6)

    def test_cosine_weights_zero_norm(self):
        """
        测试 memory 或 keys 的零向量情况，确保不发生除零错误，并正确处理。
        """
        # 测试 memory 或 keys 的零向量情况，确保不发生除零错误
        batch_size = 1
        memory_size = 2
        num_heads = 1
        word_size = 3

        # 创建 CosineWeights 实例
        cosine_weights = DefaultContentWeightCalculator(num_heads=num_heads, word_size=word_size)

        # 创建 memory、keys 和 strengths，其中 memory 包含零向量
        memory = tf.constant([[[0.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0]]], dtype=tf.float32)  # [1, 2, 3]
        keys = tf.constant([[[1.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 1, 3]
        strengths = tf.constant([[1.0]], dtype=tf.float32)  # [1, 1]

        # 调用 compute 方法
        weights = cosine_weights.compute(keys, strengths, memory)

        # 检查输出形状
        expected_shape = [1, 1, 2]
        self.assertAllEqual(weights.shape, expected_shape)

        # 检查权重范围
        self.assertGreaterEqual(tf.reduce_min(weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(weights).numpy(), 1.0)

        # 检查权重在 memory_size 维度上的和为1
        sum_weights = tf.reduce_sum(weights, axis=-1)  # [1, 1]
        self.assertAllClose(sum_weights.numpy(), np.ones_like(sum_weights.numpy()), atol=1e-6)

        # 计算预期值
        expected_similarity = np.array([[[0.0, 0.57735026]]], dtype=np.float32)
        expected_weights = tf.nn.softmax(expected_similarity * strengths.numpy(), axis=-1).numpy()
        self.assertAllClose(weights.numpy(), expected_weights, atol=1e-6)

    def test_cosine_weights_all_zero_norm(self):
        """
        测试所有 memory 向量的余弦相似度为零，期望均匀分配权重。
        """
        # 新增测试：所有 memory 向量的余弦相似度为零，期望均匀分配权重
        batch_size = 1
        memory_size = 2
        num_heads = 1
        word_size = 3

        # 创建 CosineWeights 实例
        cosine_weights = DefaultContentWeightCalculator(num_heads=num_heads, word_size=word_size)

        # 创建 memory、keys 和 strengths，其中所有 memory 向量都是零向量
        memory = tf.constant([[[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 2, 3]
        keys = tf.constant([[[1.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 1, 3]
        strengths = tf.constant([[1.0]], dtype=tf.float32)  # [1, 1]

        # 调用 compute 方法
        weights = cosine_weights.compute(keys, strengths, memory)

        # 检查输出形状
        expected_shape = [1, 1, 2]
        self.assertAllEqual(weights.shape, expected_shape)

        # 检查权重范围
        self.assertGreaterEqual(tf.reduce_min(weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(weights).numpy(), 1.0)

        # 检查权重在 memory_size 维度上的和为1
        sum_weights = tf.reduce_sum(weights, axis=-1)  # [1, 1]
        self.assertAllClose(sum_weights.numpy(), np.ones_like(sum_weights.numpy()), atol=1e-6)

        # 由于所有 memory 向量的余弦相似度为零，softmax 应均匀分配权重
        expected_weights = np.array([[[0.5, 0.5]]], dtype=np.float32)
        self.assertAllClose(weights.numpy(), expected_weights, atol=1e-6)

    def test_cosine_weights_zero_key_norm(self):
        """
        测试 keys 向量为零向量，确保不发生除零错误，并正确处理。
        """
        # 新增测试：keys 向量为零向量，确保不发生除零错误
        batch_size = 1
        memory_size = 2
        num_heads = 1
        word_size = 3

        # 创建 CosineWeights 实例
        cosine_weights = DefaultContentWeightCalculator(num_heads=num_heads, word_size=word_size)

        # 创建 memory、keys 和 strengths，其中 keys 向量为零向量
        memory = tf.constant([[[1.0, 1.0, 1.0],
                               [2.0, 2.0, 2.0]]], dtype=tf.float32)  # [1, 2, 3]
        keys = tf.constant([[[0.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 1, 3]
        strengths = tf.constant([[1.0]], dtype=tf.float32)  # [1, 1]

        # 调用 compute 方法
        weights = cosine_weights.compute(keys, strengths, memory)

        # 检查输出形状
        expected_shape = [1, 1, 2]
        self.assertAllEqual(weights.shape, expected_shape)

        # 检查权重范围
        self.assertGreaterEqual(tf.reduce_min(weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(weights).numpy(), 1.0)

        # 检查权重在 memory_size 维度上的和为1
        sum_weights = tf.reduce_sum(weights, axis=-1)  # [1, 1]
        self.assertAllClose(sum_weights.numpy(), np.ones_like(sum_weights.numpy()), atol=1e-6)

        # 由于 keys 是零向量，与所有 memory 无关，softmax 应均匀分配权重
        expected_weights = np.array([[[0.5, 0.5]]], dtype=np.float32)
        self.assertAllClose(weights.numpy(), expected_weights, atol=1e-6)


if __name__ == '__main__':
    tf.test.main()