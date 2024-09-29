import numpy as np
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import weighted_softmax


class WeightedSoftmaxTest(tf.test.TestCase):
    def testValues(self):
        """
        测试 weighted_softmax 的输出是否与 TensorFlow 的 softmax 相同（strength_op 为 identity）
        """
        batch_size = 5
        num_heads = 3
        memory_size = 7

        # 使用 NumPy 生成输入数据
        activations_data = np.random.randn(batch_size, num_heads, memory_size).astype(np.float32)
        strengths_data = np.ones((batch_size, num_heads), dtype=np.float32)

        # 定义输入张量
        activations = tf.convert_to_tensor(activations_data)
        strengths = tf.convert_to_tensor(strengths_data)

        # 调用加权 Softmax 函数，使用 identity 作为 strength_op
        observed = weighted_softmax(activations, strengths, strength_op=tf.identity)

        # TensorFlow 自带的 softmax 函数用于对比
        expected = tf.nn.softmax(activations, axis=-1)

        # 使用 eager execution 模式直接运行
        observed_val = observed.numpy()
        expected_val = expected.numpy()

        # 验证 observed 和 expected 是否接近
        self.assertAllClose(observed_val, expected_val, atol=1e-6)

    def test_weighted_softmax_with_softmax_strength(self):
        """
        测试 weighted_softmax 使用 softmax 作为 strength_op 时的输出
        """
        batch_size = 2
        num_heads = 2
        memory_size = 3

        # 使用固定的输入数据
        activations = tf.constant([[[1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0]],
                                   [[1.0, 0.0, -1.0],
                                    [0.0, 1.0, -1.0]]], dtype=tf.float32)  # [2, 2, 3]

        strengths = tf.constant([[1.0, 2.0],
                                 [3.0, 4.0]], dtype=tf.float32)  # [2, 2]

        # 定义 strength_op 为 softmax
        strength_op = tf.nn.softmax

        # 计算 weighted_softmax
        observed = weighted_softmax(activations, strengths, strength_op)

        # 手动计算加权分数
        adjusted_strengths = strength_op(tf.expand_dims(strengths, axis=-1))  # [2, 2, 1]
        weighted_scores = activations * adjusted_strengths  # [2, 2, 3]
        expected = tf.nn.softmax(weighted_scores, axis=-1)  # [2, 2, 3]

        # 验证
        self.assertAllClose(observed.numpy(), expected.numpy(), atol=1e-6)

    def test_weighted_softmax_zero_strength(self):
        """
        测试 strength 为零时，确保 weighted_softmax 的行为
        """
        batch_size = 1
        num_heads = 1
        memory_size = 3

        activations = tf.constant([[[1.0, 2.0, 3.0]]], dtype=tf.float32)  # [1, 1, 3]
        strengths = tf.constant([[0.0]], dtype=tf.float32)  # [1, 1]

        # 定义 strength_op 为 identity
        strength_op = tf.identity

        # 计算 weighted_softmax
        observed = weighted_softmax(activations, strengths, strength_op)

        # 手动计算加权分数
        weighted_scores = activations * tf.expand_dims(strengths, axis=-1)  # [1, 1, 3]
        expected = tf.nn.softmax(weighted_scores, axis=-1)  # softmax([0,0,0]) = [1/3,1/3,1/3]

        # 验证
        self.assertAllClose(observed.numpy(), expected.numpy(), atol=1e-6)

    def test_weighted_softmax_extreme_strengths(self):
        """
        测试 weighted_softmax 在极小和极大 strengths 值下的行为。
        """
        # 定义输入参数
        batch_size = 1
        num_heads = 1
        memory_size = 3

        # 创建固定的 activations
        activations = tf.constant([[[1.0, 2.0, 3.0]]], dtype=tf.float32)  # [1, 1, 3]

        # 测试极小的 strengths 值
        strengths_small = tf.constant([[1e-6]], dtype=tf.float32)  # [1, 1]
        expected_small = tf.nn.softmax(activations * tf.expand_dims(strengths_small, axis=-1),
                                       axis=-1)  # 应接近 softmax([0, 0, 0]) = [1/3, 1/3, 1/3]

        # 计算 weighted_softmax
        observed_small = weighted_softmax(activations, strengths_small, strength_op=tf.identity)

        # 验证
        self.assertAllClose(observed_small.numpy(), expected_small.numpy(), atol=1e-4)

        # 测试极大的 strengths 值
        strengths_large = tf.constant([[1e6]], dtype=tf.float32)  # [1, 1]
        # 由于 strengths 很大，weighted_scores 会非常大，softmax 会接近于 [0, 0, 1]
        expected_large = tf.nn.softmax(activations * tf.expand_dims(strengths_large, axis=-1), axis=-1)
        # 由于 1e6 * 1 = 1e6, 1e6 * 2 = 2e6, 1e6 * 3 = 3e6
        # softmax([1e6, 2e6, 3e6]) ≈ [0, 0, 1]
        expected_large = tf.constant([[[0.0, 0.0, 1.0]]], dtype=tf.float32)

        # 计算 weighted_softmax
        observed_large = weighted_softmax(activations, strengths_large, strength_op=tf.identity)

        # 验证
        self.assertAllClose(observed_large.numpy(), expected_large.numpy(), atol=1e-4)

    def test_weighted_softmax_with_softmax_strength_op(self):
        """
        测试 weighted_softmax 使用 softmax 作为 strength_op 时的输出是否等同于标准 softmax
        """
        batch_size = 2
        num_heads = 2
        memory_size = 4

        # 生成随机 activations 和 strengths
        activations_data = np.random.randn(batch_size, num_heads, memory_size).astype(np.float32)
        strengths_data = np.random.randn(batch_size, num_heads).astype(np.float32)

        activations = tf.convert_to_tensor(activations_data)
        strengths = tf.convert_to_tensor(strengths_data)

        # 使用 softmax 作为 strength_op
        strength_op = tf.nn.softmax

        # 计算 weighted_softmax
        observed = weighted_softmax(activations, strengths, strength_op=strength_op)

        # 由于 strength_op=tf.nn.softmax 会将 weights_expanded 全部变为 1（因为 softmax 在最后一个维度为1时输出1）
        # 所以 weighted_scores = scores * 1 = scores
        expected = tf.nn.softmax(activations, axis=-1)

        # 验证
        self.assertAllClose(observed.numpy(), expected.numpy(), atol=1e-6)
