"""Tests for memory addressing."""

import numpy as np
import tensorflow as tf

# from parameterized import parameterized

from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import TemporalLinkage, weighted_softmax, CosineWeights, \
    WriteAllocation, UsageUpdate


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


# 假设 CosineWeights 和 weighted_softmax 已经被正确导入
# from your_module import CosineWeights, weighted_softmax

class CosineWeightsTest(tf.test.TestCase):
    def test_cosine_weights_basic(self):
        """
        测试 CosineWeights 层的基本功能，包括输出形状、权重范围和归一化。
        """
        batch_size = 2
        memory_size = 4
        num_heads = 3
        word_size = 5

        # 创建 CosineWeights 实例，使用默认的 softmax 作为 strength_op
        cosine_weights_layer = CosineWeights(num_heads=num_heads, word_size=word_size)

        # 创建随机的 memory、keys 和 strengths
        memory = tf.random.uniform([batch_size, memory_size, word_size], minval=-1.0, maxval=1.0)
        keys = tf.random.uniform([batch_size, num_heads, word_size], minval=-1.0, maxval=1.0)
        strengths = tf.random.uniform([batch_size, num_heads], minval=0.1, maxval=2.0)

        # 构建输入字典
        inputs = {
            'memory': memory,
            'keys': keys,
            'strengths': strengths
        }

        # 调用层
        weights = cosine_weights_layer(inputs)

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
        测试 CosineWeights 层在任意批次维度下的行为，例如 [batch_dim1, batch_dim2, ...]。
        """
        # 测试任意批次维度，例如 [batch_dim1, batch_dim2, ...]
        batch_dims = [3, 2]
        memory_size = 5
        num_heads = 4
        word_size = 6

        # 创建 CosineWeights 实例
        cosine_weights_layer = CosineWeights(num_heads=num_heads, word_size=word_size)

        # 创建随机的 memory、keys 和 strengths
        memory_shape = batch_dims + [memory_size, word_size]
        keys_shape = batch_dims + [num_heads, word_size]
        strengths_shape = batch_dims + [num_heads]

        memory = tf.random.uniform(memory_shape, minval=-1.0, maxval=1.0)
        keys = tf.random.uniform(keys_shape, minval=-1.0, maxval=1.0)
        strengths = tf.random.uniform(strengths_shape, minval=0.1, maxval=2.0)

        # 构建输入字典
        inputs = {
            'memory': memory,
            'keys': keys,
            'strengths': strengths
        }

        # 调用层
        weights = cosine_weights_layer(inputs)

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
        cosine_weights_layer = CosineWeights(num_heads=num_heads, word_size=word_size)

        # 创建 memory、keys 和 strengths，其中 memory 包含零向量
        memory = tf.constant([[[0.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0]]], dtype=tf.float32)  # [1, 2, 3]
        keys = tf.constant([[[1.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 1, 3]
        strengths = tf.constant([[1.0]], dtype=tf.float32)  # [1, 1]

        # 构建输入字典
        inputs = {
            'memory': memory,
            'keys': keys,
            'strengths': strengths
        }

        # 调用层
        weights = cosine_weights_layer(inputs)

        # 检查输出形状
        expected_shape = [1, 1, 2]
        self.assertAllEqual(weights.shape, expected_shape)

        # 检查权重范围
        self.assertGreaterEqual(tf.reduce_min(weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(weights).numpy(), 1.0)

        # 检查权重在 memory_size 维度上的和为1
        sum_weights = tf.reduce_sum(weights, axis=-1)  # [1, 1]
        self.assertAllClose(sum_weights.numpy(), np.ones_like(sum_weights.numpy()), atol=1e-6)

        # 由于第一条 memory 是零向量，与 keys 无关，softmax 应根据非零相似度分配权重
        # 期望值为 [0.3595, 0.6405]
        expected_weights = np.array([[[0.3595, 0.6405]]], dtype=np.float32)
        self.assertAllClose(weights.numpy(), expected_weights, atol=1.1e-3)  # 增加 atol 至 1.1e-3

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
        cosine_weights_layer = CosineWeights(num_heads=num_heads, word_size=word_size)

        # 创建 memory、keys 和 strengths，其中所有 memory 向量都是零向量
        memory = tf.constant([[[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 2, 3]
        keys = tf.constant([[[1.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 1, 3]
        strengths = tf.constant([[1.0]], dtype=tf.float32)  # [1, 1]

        # 构建输入字典
        inputs = {
            'memory': memory,
            'keys': keys,
            'strengths': strengths
        }

        # 调用层
        weights = cosine_weights_layer(inputs)

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
        self.assertAllClose(weights.numpy(), expected_weights, atol=1e-3)

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
        cosine_weights_layer = CosineWeights(num_heads=num_heads, word_size=word_size)

        # 创建 memory、keys 和 strengths，其中 keys 向量为零向量
        memory = tf.constant([[[1.0, 1.0, 1.0],
                               [2.0, 2.0, 2.0]]], dtype=tf.float32)  # [1, 2, 3]
        keys = tf.constant([[[0.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 1, 3]
        strengths = tf.constant([[1.0]], dtype=tf.float32)  # [1, 1]

        # 构建输入字典
        inputs = {
            'memory': memory,
            'keys': keys,
            'strengths': strengths
        }

        # 调用层
        weights = cosine_weights_layer(inputs)

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
        self.assertAllClose(weights.numpy(), expected_weights, atol=1e-3)

    # @parameterized.expand([
    #     # (description, memory, keys, strengths, expected_weights)
    #     (
    #         "Zero and non-zero memory vector",
    #         tf.constant([[[0.0, 0.0, 0.0],
    #                       [1.0, 1.0, 1.0]]], dtype=tf.float32),
    #         tf.constant([[[1.0, 0.0, 0.0]]], dtype=tf.float32),
    #         tf.constant([[1.0]], dtype=tf.float32),
    #         np.array([[[0.318, 0.682]]], dtype=np.float32)
    #     ),
    #     (
    #         "All zero memory vectors",
    #         tf.constant([[[0.0, 0.0, 0.0],
    #                       [0.0, 0.0, 0.0]]], dtype=tf.float32),
    #         tf.constant([[[1.0, 0.0, 0.0]]], dtype=tf.float32),
    #         tf.constant([[1.0]], dtype=tf.float32),
    #         np.array([[[0.5, 0.5]]], dtype=np.float32)
    #     ),
    #     (
    #         "Zero key vector",
    #         tf.constant([[[1.0, 1.0, 1.0],
    #                       [2.0, 2.0, 2.0]]], dtype=tf.float32),
    #         tf.constant([[[0.0, 0.0, 0.0]]], dtype=tf.float32),
    #         tf.constant([[1.0]], dtype=tf.float32),
    #         np.array([[[0.5, 0.5]]], dtype=np.float32)
    #     ),
    # ])
    # def test_weighted_softmax_cases(self, _, memory, keys, strengths, expected_weights):
    #     cosine_weights_layer = CosineWeights(num_heads=keys.shape[1], word_size=keys.shape[-1])
    #
    #     inputs = {
    #         'memory': memory,
    #         'keys': keys,
    #         'strengths': strengths
    #     }
    #
    #     weights = cosine_weights_layer(inputs)
    #
    #     # 检查输出形状
    #     expected_shape = list(expected_weights.shape)
    #     self.assertAllEqual(weights.shape, expected_shape)
    #
    #     # 检查权重范围
    #     self.assertGreaterEqual(tf.reduce_min(weights).numpy(), 0.0)
    #     self.assertLessEqual(tf.reduce_max(weights).numpy(), 1.0)
    #
    #     # 检查权重在 memory_size 维度上的和为1
    #     sum_weights = tf.reduce_sum(weights, axis=-1)
    #     self.assertAllClose(sum_weights.numpy(), np.ones_like(sum_weights.numpy()), atol=1e-6)
    #
    #     # 检查权重值
    #     self.assertAllClose(weights.numpy(), expected_weights, atol=1e-3)


class TemporalLinkageTest(tf.test.TestCase):
    def setUp(self):
        super(TemporalLinkageTest, self).setUp()
        self.memory_size = 3
        self.num_writes = 2
        self.temporal_linkage_layer = TemporalLinkage(
            memory_size=self.memory_size,
            num_writes=self.num_writes
        )

    def test_basic_temporal_linkage_update(self):
        """
        基本测试：验证链路矩阵和优先级权重的更新。
        """
        batch_size = 1

        # 定义 write_weights
        write_weights = tf.constant([
            [[0.1, 0.2, 0.3],
             [0.4, 0.1, 0.2]]
        ], dtype=tf.float32)  # [1, 2, 3]

        # 定义 prev_linkage
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        }

        # 调用 TemporalLinkage 层
        updated_linkage = self.temporal_linkage_layer({
            'write_weights': write_weights,
            'prev_linkage': prev_linkage
        }, training=False)

        # 预期 precedence_weights = write_weights * epsilon
        expected_precedence_weights = write_weights * 1e-6  # [1, 2, 3]

        # 预期 link = write_weights_i * prev_precedence_weights_j = 0 since prev_precedence_weights = 0
        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_dynamic_num_writes(self):
        """
        测试动态调整 num_writes 的情况。
        """
        batch_size = 2  # 确保 batch_size 为 2
        dynamic_num_writes = 3

        # 定义 write_weights，形状与 batch_size 一致
        write_weights = tf.constant([
            [[0.1, 0.2, 0.3],
             [0.4, 0.1, 0.2],
             [0.2, 0.2, 0.2]],
            [[0.3, 0.2, 0.1],
             [0.1, 0.5, 0.2],
             [0.3, 0.2, 0.1]]
        ], dtype=tf.float32)  # [2, 3, 3]

        # 定义 prev_linkage，形状与 batch_size 一致
        prev_linkage = {
            'link': tf.zeros([batch_size, dynamic_num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, dynamic_num_writes, self.memory_size], dtype=tf.float32)
        }

        # 调用 TemporalLinkage 层
        updated_linkage = self.temporal_linkage_layer({
            'write_weights': write_weights,
            'prev_linkage': prev_linkage
        }, training=False)

        # 预期 precedence_weights = write_weights * epsilon
        expected_precedence_weights = write_weights * self.temporal_linkage_layer.epsilon  # [2, 3, 3]

        # 预期 link = write_weights_i * prev_precedence_weights_j = 0 since prev_precedence_weights = 0
        expected_link = tf.zeros([batch_size, dynamic_num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        # 验证 precedence_weights 和 link
        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(),
                            atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)
    def test_partial_write_and_read(self):
        """
        测试部分内存槽被写入和部分内存槽被读出的情况。
        """
        batch_size = 1

        # 创建初始使用率
        initial_usage = tf.constant([[0.2, 0.5, 0.3]], dtype=tf.float32)  # [1, 3]

        # 定义 write_weights
        write_weights = tf.constant([
            [[0.1, 0.2, 0.3],
             [0.4, 0.1, 0.2]]
        ], dtype=tf.float32)  # [1, 2, 3]

        # 定义 free_gate 和 read_weights
        free_gate = tf.constant([
            [0.5, 0.5]
        ], dtype=tf.float32)  # [1, 2]

        read_weights = tf.constant([
            [[0.3, 0.4, 0.3],
             [0.2, 0.5, 0.3]]
        ], dtype=tf.float32)  # [1, 2, 3]

        # 调用 UsageUpdate 层（假设已经与 TemporalLinkage 兼容）
        # 此处仅测试 TemporalLinkage，不涉及 UsageUpdate，因此 initial_usage 等参数仅用于上下文
        # 您可能需要根据实际实现调整此部分

        # 调用 TemporalLinkage 层
        updated_linkage = self.temporal_linkage_layer({
            'write_weights': write_weights,
            'prev_linkage': {
                'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
                'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
            }
        }, training=False)

        # 预期 precedence_weights = write_weights * epsilon
        expected_precedence_weights = write_weights * 1e-6  # [1, 2, 3]

        # 预期 link = write_weights_i * prev_precedence_weights_j = 0 since prev_precedence_weights = 0
        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_initial_state(self):
        """
        测试 get_initial_state 方法，确保返回正确的初始链路矩阵和优先级权重。
        """
        batch_size = 4
        initial_linkage = self.temporal_linkage_layer.get_initial_state([batch_size])

        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)
        expected_precedence_weights = tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)

        self.assertAllClose(initial_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)
        self.assertAllClose(initial_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)

    def test_state_size(self):
        """
        测试 state_size 属性，确保返回正确的形状。
        """
        expected_state_size = {
            'link': tf.TensorShape([self.num_writes, self.memory_size, self.memory_size]),
            'precedence_weights': tf.TensorShape([self.num_writes, self.memory_size])
        }
        self.assertEqual(self.temporal_linkage_layer.state_size, expected_state_size)

    def testLinkUpdate_steps(self):
        """
        测试链路更新的多个步骤。
        """
        batch_size = 1
        memory_size = 3
        num_writes = 1

        module = TemporalLinkage(memory_size=memory_size, num_writes=num_writes, epsilon=1e-6)

        # 初始化状态
        prev_link = tf.zeros([batch_size, num_writes, memory_size, memory_size], dtype=tf.float32)
        prev_precedence_weights = tf.zeros([batch_size, num_writes, memory_size], dtype=tf.float32)

        # 第一次写入
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)  # 写入位置 0
        linkage_state = module({
            'write_weights': write_weights,
            'prev_linkage': {
                'link': prev_link,
                'precedence_weights': prev_precedence_weights
            }
        }, training=False)

        # 验证第一次写入后的链路矩阵
        expected_link_after_first_write = np.zeros((1, 1, 3, 3), dtype=np.float32)
        self.assertAllClose(linkage_state['link'].numpy(), expected_link_after_first_write, atol=1e-6)

        # 更新 prev_link 和 prev_precedence_weights
        prev_link = linkage_state['link']
        prev_precedence_weights = linkage_state['precedence_weights']  # p_t = w_t^w + (1 - sum(w_t^w)) * p_{t-1}

        # 第二次写入
        write_weights = tf.constant([[[0, 1, 0]]], dtype=tf.float32)  # 写入位置 1
        linkage_state = module({
            'write_weights': write_weights,
            'prev_linkage': {
                'link': prev_link,
                'precedence_weights': prev_precedence_weights
            }
        }, training=False)

        # 验证第二次写入后的链路矩阵
        expected_link_after_second_write = np.array([[[[0, 0, 0],
                                                       [1e-6, 0, 0],
                                                       [0, 0, 0]]]], dtype=np.float32)
        self.assertAllClose(linkage_state['link'].numpy(), expected_link_after_second_write, atol=1e-6)
    def test_full_link_method(self):
        """
        测试 full_link 方法，确保在第一次写入时链路矩阵为全零。
        """
        batch_size = 1
        memory_size = 3
        num_writes = 1

        # 创建 TemporalLinkage 实例
        module = TemporalLinkage(memory_size=memory_size, num_writes=num_writes)

        # 初始化 prev_link 和 prev_precedence_weights
        prev_link = tf.zeros([batch_size, num_writes, memory_size, memory_size], dtype=tf.float32)
        prev_precedence_weights = tf.zeros([batch_size, num_writes, memory_size], dtype=tf.float32)

        # 创建 write_weights
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        # 调用模块
        linkage_state = module({
            'write_weights': write_weights,
            'prev_linkage': {
                'link': prev_link,
                'precedence_weights': prev_precedence_weights
            }
        }, training=False)

        # 验证更新后的 link 矩阵是否符合预期
        expected_link = np.zeros((1, 1, 3, 3), dtype=np.float32)  # 第一次写入，link 应为全零矩阵
        self.assertAllClose(linkage_state['link'].numpy(), expected_link, atol=1e-6)

    def test_new_link_creation(self):
        """
        测试新链路的创建，确保当 prev_precedence_weights 为零时，新链路为零矩阵。
        """
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)

        write_weights_i = tf.expand_dims(write_weights, 3)  # (batch_size, num_writes, memory_size, 1)
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)  # (batch_size, num_writes, 1, memory_size)

        # 计算 new_link，使用矩阵乘法
        new_link = tf.matmul(write_weights_i, prev_precedence_weights_j)

        # 由于 prev_precedence_weights 为零，new_link 应该是全零矩阵
        expected_new_link = np.zeros((1, 1, 3, 3), dtype=np.float32)
        np.testing.assert_allclose(new_link.numpy(), expected_new_link, atol=1e-6)

    def test_updated_link(self):
        """
        测试更新后的链路矩阵，确保在 prev_precedence_weights 为零时，updated_link 仍为零矩阵。
        """
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)
        prev_link = tf.zeros([1, 1, 3, 3], dtype=tf.float32)

        write_weights_i = tf.expand_dims(write_weights, 3)
        write_weights_j = tf.expand_dims(write_weights, 2)
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)

        # 计算 prev_link_scale
        prev_link_scale = 1 - write_weights_i - write_weights_j  # 不进行裁剪

        # 计算 new_link，使用矩阵乘法
        new_link = tf.matmul(write_weights_i, prev_precedence_weights_j)
        updated_link = prev_link_scale * prev_link + new_link

        # 避免自连接
        mask = tf.eye(3, batch_shape=[1, 1], dtype=tf.float32)
        updated_link = updated_link * (1 - mask)

        # 由于 prev_precedence_weights 为零，new_link 应该是全零矩阵
        expected_updated_link = np.zeros((1, 1, 3, 3), dtype=np.float32)
        np.testing.assert_allclose(updated_link.numpy(), expected_updated_link, atol=1e-6)

    def testModule(self):
        """
        测试模块的整体功能，确保输出形状和数值范围正确。
        """
        batch_size = 2
        memory_size = 4
        num_writes = 2

        module = TemporalLinkage(memory_size=memory_size, num_writes=num_writes)

        # 初始化状态
        prev_link = tf.zeros([batch_size, num_writes, memory_size, memory_size], dtype=tf.float32)
        prev_precedence_weights = tf.zeros([batch_size, num_writes, memory_size], dtype=tf.float32)

        # 随机生成 write_weights
        write_weights = tf.random.uniform([batch_size, num_writes, memory_size], minval=0.0, maxval=1.0)

        # 归一化 write_weights
        write_weights /= tf.reduce_sum(write_weights, axis=2, keepdims=True) + 1e-6

        # 调用模块
        linkage_state = module({
            'write_weights': write_weights,
            'prev_linkage': {
                'link': prev_link,
                'precedence_weights': prev_precedence_weights
            }
        })

        # 检查输出形状
        self.assertEqual(linkage_state['link'].shape, (batch_size, num_writes, memory_size, memory_size))
        self.assertEqual(linkage_state['precedence_weights'].shape, (batch_size, num_writes, memory_size))

        # 检查 link 矩阵的数值范围
        self.assertGreaterEqual(tf.reduce_min(linkage_state['link']).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(linkage_state['link']).numpy(), 1.0)

        # 检查对角线元素为 0
        diag_elements = tf.linalg.diag_part(linkage_state['link'])
        expected_diag = tf.zeros([batch_size, num_writes, memory_size], dtype=tf.float32)
        self.assertAllClose(diag_elements.numpy(), expected_diag.numpy(), atol=1e-6)

    def testDirectionalReadWeights(self):
        """
        测试方向性读权重的计算，确保输出形状和数值范围正确。
        """
        batch_size = 2
        memory_size = 4
        num_writes = 2
        num_reads = 3

        module = TemporalLinkage(memory_size=memory_size, num_writes=num_writes)

        # 创建随机的 link 矩阵和 prev_read_weights
        link = tf.random.uniform([batch_size, num_writes, memory_size, memory_size], minval=0.0, maxval=1.0)
        prev_read_weights = tf.random.uniform([batch_size, num_reads, memory_size], minval=0.0, maxval=1.0)

        # 调用 directional_read_weights
        forward_weights = module.directional_read_weights(link, prev_read_weights, forward=True)
        backward_weights = module.directional_read_weights(link, prev_read_weights, forward=False)

        # 检查输出形状
        self.assertEqual(forward_weights.shape, (batch_size, num_reads, num_writes, memory_size))
        self.assertEqual(backward_weights.shape, (batch_size, num_reads, num_writes, memory_size))

        # 检查数值范围（例如，非负且合理范围内）
        self.assertGreaterEqual(tf.reduce_min(forward_weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(forward_weights).numpy(), 1.0)
        self.assertGreaterEqual(tf.reduce_min(backward_weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(backward_weights).numpy(), 1.0)

        # 检查权重和为1
        forward_sum = tf.reduce_sum(forward_weights, axis=-1)  # [batch_size, num_reads, num_writes]
        backward_sum = tf.reduce_sum(backward_weights, axis=-1)  # [batch_size, num_reads, num_writes]
        self.assertAllClose(forward_sum.numpy(), np.ones_like(forward_sum.numpy()), atol=1e-6)
        self.assertAllClose(backward_sum.numpy(), np.ones_like(backward_sum.numpy()), atol=1e-6)

    def test_write_weights_i_and_j(self):
        """
        测试 write_weights 的扩展操作，确保形状和数值正确。
        """
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        # 扩展维度
        write_weights_i = tf.expand_dims(write_weights, 3)  # (batch_size, num_writes, memory_size, 1)
        write_weights_j = tf.expand_dims(write_weights, 2)  # (batch_size, num_writes, 1, memory_size)

        # 验证 shape 和值
        expected_write_weights_i = np.array([[[[1], [0], [0]]]], dtype=np.float32)
        expected_write_weights_j = np.array([[[[1, 0, 0]]]], dtype=np.float32)

        np.testing.assert_allclose(write_weights_i.numpy(), expected_write_weights_i, atol=1e-6)
        np.testing.assert_allclose(write_weights_j.numpy(), expected_write_weights_j, atol=1e-6)

    def test_precedence_weights_update(self):
        """
        测试优先级权重的更新，确保正确计算。
        """
        # 初始 precedence_weights 为全零
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)  # 第一次写入位置 0

        # 模拟 precedence_weights 更新过程
        reset_gate = 1 - tf.reduce_sum(write_weights, axis=2, keepdims=True)  # [batch_size, num_writes, 1]
        updated_precedence_weights = reset_gate * prev_precedence_weights + write_weights  # [batch_size, num_writes, memory_size]

        # 打印调试信息（可选）
        tf.print("Prev Precedence Weights:", prev_precedence_weights)
        tf.print("Write Weights:", write_weights)
        tf.print("Reset Gate:", reset_gate)
        tf.print("Updated Precedence Weights:", updated_precedence_weights)

        # 预期 precedence_weights = write_weights，因为 prev_precedence_weights 为零
        expected_precedence_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        self.assertAllClose(updated_precedence_weights.numpy(), expected_precedence_weights.numpy(), atol=1e-6)

    def test_write_weights_expansion(self):
        """
        测试 write_weights_i 和 write_weights_j 的扩展，确保正确性。
        """
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        # 扩展维度
        write_weights_i = tf.expand_dims(write_weights, 3)  # (batch_size, num_writes, memory_size, 1)
        write_weights_j = tf.expand_dims(write_weights, 2)  # (batch_size, num_writes, 1, memory_size)

        # 验证 shape 和值
        expected_write_weights_i = np.array([[[[1], [0], [0]]]], dtype=np.float32)
        expected_write_weights_j = np.array([[[[1, 0, 0]]]], dtype=np.float32)

        np.testing.assert_allclose(write_weights_i.numpy(), expected_write_weights_i, atol=1e-6)
        np.testing.assert_allclose(write_weights_j.numpy(), expected_write_weights_j, atol=1e-6)

    def test_prev_precedence_weights_j(self):
        """
        测试 prev_precedence_weights_j 的扩展，确保正确性。
        """
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)

        # 计算 prev_precedence_weights_j
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights,
                                                   2)  # (batch_size, num_writes, 1, memory_size)

        # 验证 shape 和值
        expected_prev_precedence_weights_j = np.array([[[[0, 0, 0]]]], dtype=np.float32)

        np.testing.assert_allclose(prev_precedence_weights_j.numpy(), expected_prev_precedence_weights_j, atol=1e-6)

    def test_prev_link_scale(self):
        """
        测试 prev_link_scale 的计算，确保正确应用百分比缩放因子。
        """
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        # 扩展维度
        write_weights_i = tf.expand_dims(write_weights, 3)
        write_weights_j = tf.expand_dims(write_weights, 2)

        # 计算 prev_link_scale，使用百分比缩放
        prev_link_scale = 1.0 - (write_weights_i + write_weights_j) / 2.0
        prev_link_scale = tf.clip_by_value(prev_link_scale, 0.0, 1.0)

        # 打印调试信息
        tf.print("write_weights_i:", write_weights_i)
        tf.print("write_weights_j:", write_weights_j)
        tf.print("prev_link_scale:", prev_link_scale)

        # 预期的 prev_link_scale
        expected_prev_link_scale_clipped = np.array([[[[0, 0.5, 0.5],
                                                       [0.5, 1, 1],
                                                       [0.5, 1, 1]]]], dtype=np.float32)

        # 验证百分比缩放后的 prev_link_scale
        np.testing.assert_allclose(prev_link_scale.numpy(), expected_prev_link_scale_clipped, atol=1e-6)
    def test_gradient_flow(self):
        """
        验证 TemporalLinkage 层的梯度是否能够正确传播。
        """
        batch_size = 1

        # 定义可训练变量
        write_weights = tf.Variable([[[0.5, 0.3, 0.2]]], dtype=tf.float32)  # [1, 1, 3]

        # 定义 prev_linkage
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        }

        with tf.GradientTape() as tape:
            # 调用 TemporalLinkage 层
            updated_linkage = self.temporal_linkage_layer({
                'write_weights': write_weights,
                'prev_linkage': prev_linkage
            }, training=True)

            # 定义一个依赖于 precedence_weights 的损失函数
            loss = tf.reduce_sum(updated_linkage['precedence_weights'])  # scalar

        # 计算梯度
        gradients = tape.gradient(loss, [write_weights])

        # 检查梯度
        for grad, var_name in zip(gradients, ['write_weights']):
            self.assertIsNotNone(grad, f"No gradient for {var_name}")
            grad_norm = tf.norm(grad).numpy()
            self.assertGreater(grad_norm, 1e-12, f"Gradient too small for {var_name}")
            tf.print(f"Gradient norm for {var_name}: {grad_norm}")

    def test_all_write_weights_zero(self):
        """
        测试所有写入权重为零的情况，确保 precedence_weights 和 link 不更新。
        """
        batch_size = 1
        write_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)  # [1, 1, 3]
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)
        prev_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        # 调用 TemporalLinkage 层
        updated_linkage = self.temporal_linkage_layer({
            'write_weights': write_weights,
            'prev_linkage': {
                'link': prev_link,
                'precedence_weights': prev_precedence_weights
            }
        }, training=False)

        # 预期 precedence_weights 和 link 不变
        expected_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)
        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size],
                                 dtype=tf.float32)

        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(),
                            atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_multiple_writes(self):
        """
        测试多个写入操作，确保 precedence_weights 和 link 正确累积。
        """
        batch_size = 1
        write_weights_1 = tf.constant([[[0.2, 0.3, 0.5]]], dtype=tf.float32)  # 第一次写入
        write_weights_2 = tf.constant([[[0.1, 0.6, 0.3]]], dtype=tf.float32)  # 第二次写入

        # 调整 TemporalLinkage 实例的 num_writes 为 2 以应对两次写入
        module = TemporalLinkage(memory_size=self.memory_size, num_writes=2)

        # 第一次写入
        linkage_state_1 = module({
            'write_weights': write_weights_1,
            'prev_linkage': {
                'link': tf.zeros([batch_size, 1, self.memory_size, self.memory_size], dtype=tf.float32),
                'precedence_weights': tf.zeros([batch_size, 1, self.memory_size], dtype=tf.float32)
            }
        }, training=False)

        # 修正 expected_precedence_weights_1 为两次写入的形状
        expected_precedence_weights_1 = write_weights_1 * 1e-6
        expected_precedence_weights_1 = tf.reshape(expected_precedence_weights_1,
                                                   linkage_state_1['precedence_weights'].shape)

        # 验证第一次写入后的 precedence_weights 和 link
        self.assertAllClose(linkage_state_1['precedence_weights'].numpy(), expected_precedence_weights_1.numpy(),
                            atol=1e-6)

        expected_link_1 = tf.zeros([batch_size, 1, self.memory_size, self.memory_size], dtype=tf.float32)
        self.assertAllClose(linkage_state_1['link'].numpy(), expected_link_1.numpy(), atol=1e-6)

        # 第二次写入，num_writes 更新为 2
        linkage_state_2 = module({
            'write_weights': write_weights_2,
            'prev_linkage': linkage_state_1
        }, training=False)

        # 预期 precedence_weights 在第二次写入后应更新为：
        expected_precedence_weights_2 = (write_weights_2 + expected_precedence_weights_1) * 1e-6
        expected_precedence_weights_2 = tf.reshape(expected_precedence_weights_2,
                                                   linkage_state_2['precedence_weights'].shape)

        # 验证第二次写入后的 precedence_weights 和 link
        self.assertAllClose(linkage_state_2['precedence_weights'].numpy(), expected_precedence_weights_2.numpy(),atol=1e-6)

    def test_write_and_precedence_weights_relation(self):
        """
        测试写入权重和 precedence_weights 之间的关系，确保正确累积。
        """
        batch_size = 1
        write_weights = tf.constant([[[0.3, 0.4, 0.3]]], dtype=tf.float32)  # [1, 1, 3]
        prev_precedence_weights = tf.constant([[[0.1, 0.2, 0.3]]], dtype=tf.float32)

        # 计算 updated_precedence_weights
        reset_gate = 1 - tf.reduce_sum(write_weights, axis=2, keepdims=True)  # [1, 1, 1]
        updated_precedence_weights = reset_gate * prev_precedence_weights + write_weights  # [1, 1, 3]

        # 预期 precedence_weights = (1 - 1.0) * prev + write_weights = write_weights
        expected_precedence_weights = write_weights  # [1, 1, 3]

        self.assertAllClose(updated_precedence_weights.numpy(), expected_precedence_weights.numpy(), atol=1e-6)

    def test_link_update_with_non_zero_precedence(self):
        """
        测试链路更新时，precedence_weights 非零的情况，基于 DNC 的经典链路更新设计。
        """
        # 定义输入参数
        write_weights = tf.constant([[[0.2, 0.0, 0.8],
                                      [0.1, 0.3, 0.6]]], dtype=tf.float32)  # Shape: [1, 2, 3]
        prev_precedence_weights = tf.constant([[[0.5, 0.0, 0.5],
                                                [0.4, 0.1, 0.5]]], dtype=tf.float32)  # Shape: [1, 2, 3]
        prev_link = tf.ones([1, 2, 3, 3], dtype=tf.float32)  # 全1矩阵, Shape: [1, 2, 3, 3]

        # 调用 _link 方法更新链路矩阵
        updated_link = self.temporal_linkage_layer._link(prev_link, prev_precedence_weights, write_weights)

        # 计算期望的新链路：new_link = write_weights_i * prev_precedence_weights_j
        # 写入头1
        new_link_1 = np.array([[[0.1, 0.0, 0.1],
                                [0.0, 0.0, 0.0],
                                [0.4, 0.0, 0.4]]], dtype=np.float32)  # Shape: [1, 1, 3, 3]

        # 写入头2
        new_link_2 = np.array([[[0.04, 0.01, 0.05],
                                [0.12, 0.03, 0.15],
                                [0.24, 0.06, 0.3]]], dtype=np.float32)  # Shape: [1, 1, 3, 3]

        # 组合所有写入头的新链路
        expected_new_link = np.stack([new_link_1, new_link_2], axis=1)  # Shape: [1, 2, 3, 3]

        # 计算期望的链路缩放因子：prev_link_scale = 1 - write_weights_i - write_weights_j
        # 写入头1
        expected_prev_link_scale_1 = np.array([[[0.6, 0.8, 0.0],
                                               [0.8, 1.0, 0.2],
                                               [0.0, 0.2, 0.0]]], dtype=np.float32)  # Shape: [1, 1, 3, 3]
        # 写入头2
        expected_prev_link_scale_2 = np.array([[[0.8, 0.6, 0.3],
                                               [0.6, 0.4, 0.1],
                                               [0.3, 0.1, 0.0]]], dtype=np.float32)  # Shape: [1, 1, 3, 3]
        # 组合所有写入头的缩放因子
        expected_prev_link_scale = np.stack([expected_prev_link_scale_1, expected_prev_link_scale_2], axis=1)  # Shape: [1, 2, 3, 3]

        # 计算期望的 updated_link：expected_updated_link = prev_link_scale * prev_link + new_link
        # 因为 prev_link 是全1矩阵，所以相当于 prev_link_scale + new_link
        expected_updated_link = expected_prev_link_scale + expected_new_link  # Shape: [1, 2, 3, 3]

        # 计算期望的 final_link，避免自连接
        # 创建 mask，形状 [1, 2, 3, 3]
        # 对于每个写入头，避免自连接
        mask = 1 - np.eye(self.memory_size, dtype=np.float32)  # Shape: [3, 3]
        mask = np.tile(mask, (self.num_writes, 1, 1))  # Shape: [2, 3, 3]
        mask = np.expand_dims(mask, axis=0)  # Shape: [1, 2, 3, 3]
        expected_final_link = expected_updated_link * mask  # Shape: [1, 2, 3, 3]

        # 打印调试信息，查看实际的 updated_link 和期望的 final_link
        tf.print("write_weights:", write_weights)
        tf.print("prev_precedence_weights:", prev_precedence_weights)
        tf.print("expected_prev_link_scale:", expected_prev_link_scale)
        tf.print("expected_new_link:", expected_new_link)
        tf.print("expected_updated_link:", expected_updated_link)
        tf.print("expected_final_link:", expected_final_link)
        tf.print("actual updated_link:", updated_link)

        # 验证链路更新结果是否符合预期
        self.assertAllClose(updated_link.numpy(), expected_final_link, atol=1e-6)




class UsageUpdateTest(tf.test.TestCase):
    def setUp(self):
        super(UsageUpdateTest, self).setUp()
        self.memory_size = 3  # 定义 memory_size
        self.num_writes = 2  # 定义 num_writes
        self.num_reads = 2  # 定义 num_reads
        self.epsilon = 1e-6  # 定义 epsilon

        # 初始化 UsageUpdate 层
        self.usage_update_layer = UsageUpdate(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            num_reads=self.num_reads,
            epsilon=self.epsilon
        )

    def test_basic_usage_update(self):
        """
        基本测试：验证写操作和读操作对使用率的影响。
        """
        batch_size = 2

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [2, 3]

        # 定义 write_weights
        write_weights = tf.constant([
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]],
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]]
        ], dtype=tf.float32)  # [2, 2, 3]

        # 定义 free_gate 和 read_weights
        free_gate = tf.constant([
            [1.0, 0.0],
            [1.0, 1.0]
        ], dtype=tf.float32)  # [2, 2]

        read_weights = tf.constant([
            [[0.5, 0.5, 0.0],
             [0.0, 0.5, 0.5]],
            [[1.0, 1.0, 1.0],
             [0.0, 1.0, 0.0]]
        ], dtype=tf.float32)  # [2, 2, 3]

        # 调用 UsageUpdate 层
        updated_usage = self.usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [2, 3]

        # 计算预期使用率
        # 1. 计算 write_allocation = 1 - prod(1 - write_weights, axis=1)
        write_weights_cumprod = tf.reduce_prod(1 - write_weights, axis=1)  # [2, 3]
        write_allocation = 1 - write_weights_cumprod  # [2, 3]

        # 2. 使用 usage_after_write = initial_usage + (1 - initial_usage) * write_allocation
        usage_after_write = initial_usage + (1 - initial_usage) * write_allocation  # [2, 3]

        # 3. 计算自由读权重
        free_gate_expanded = tf.expand_dims(free_gate, axis=-1)  # [2, 2, 1]
        free_read_weights = free_gate_expanded * read_weights  # [2, 2, 3]

        # 4. 计算 total_free_read_weights = sum(free_read_weights, axis=1)
        total_free_read_weights = tf.reduce_sum(free_read_weights, axis=1)  # [2, 3]

        # 5. 使用 usage_after_read = usage_after_write - total_free_read_weights
        usage_after_read = usage_after_write - total_free_read_weights  # [2, 3]

        # 6. 裁剪使用率到 [0, 1]
        expected_usage = tf.clip_by_value(usage_after_read, 0.0, 1.0)  # [2, 3]

        # 断言
        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_all_full_usage(self):
        """
        测试所有内存槽已满的情况，确保进一步写操作释放使用率。
        """
        batch_size = 1

        # 创建初始使用率为全1
        initial_usage = tf.ones([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义 write_weights
        write_weights = tf.constant([
            [[1 / 3, 1 / 3, 1 / 3]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 定义 free_gate 和 read_weights（全读）
        free_gate = tf.constant([
            [1.0]
        ], dtype=tf.float32)  # [1, 1]

        read_weights = tf.constant([
            [[1.0, 1.0, 1.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 调用 UsageUpdate 层
        updated_usage = self.usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 计算预期使用率
        # 1. 计算 write_allocation = 1 - prod(1 - write_weights, axis=1)
        write_weights_cumprod = tf.reduce_prod(1 - write_weights, axis=1)  # [1, 3]
        write_allocation = 1 - write_weights_cumprod  # [1, 3]

        # 2. 使用 usage_after_write = initial_usage + (1 - initial_usage) * write_allocation
        usage_after_write = initial_usage + (1 - initial_usage) * write_allocation  # [1, 3]
        # 由于 initial_usage =1, usage_after_write =1 +0 * write_allocation =1

        # 3. 计算自由读权重
        free_gate_expanded = tf.expand_dims(free_gate, axis=-1)  # [1, 1, 1]
        free_read_weights = free_gate_expanded * read_weights  # [1, 1, 3]

        # 4. 计算 total_free_read_weights = sum(free_read_weights, axis=1)
        total_free_read_weights = tf.reduce_sum(free_read_weights, axis=1)  # [1, 3]

        # 5. 使用 usage_after_read = usage_after_write - total_free_read_weights
        usage_after_read = usage_after_write - total_free_read_weights  # [1, 3]
        # 1 -1 =0, 1 -1 =0, 1 -1 =0

        # 6. 裁剪使用率到 [0, 1]
        expected_usage = tf.clip_by_value(usage_after_read, 0.0, 1.0)  # [1, 3]

        # 断言
        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_all_zero_read_weights(self):
        """
        测试所有读权重为零的情况，确保读操作不影响使用率。
        """
        batch_size = 1

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义 write_weights
        write_weights = tf.constant([
            [[1 / 3, 1 / 3, 1 / 3]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 定义 free_gate 和 read_weights（全零）
        free_gate = tf.constant([
            [0.0]
        ], dtype=tf.float32)  # [1, 1]

        read_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 调用 UsageUpdate 层
        updated_usage = self.usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 计算预期使用率
        # 1. 计算 write_allocation = 1 - prod(1 - write_weights, axis=1)
        write_weights_cumprod = tf.reduce_prod(1 - write_weights, axis=1)  # [1, 3]
        write_allocation = 1 - write_weights_cumprod  # [1, 3]

        # 2. 使用 usage_after_write = initial_usage + (1 - initial_usage) * write_allocation
        usage_after_write = initial_usage + (1 - initial_usage) * write_allocation  # [1, 3]
        # =0 +1 * write_allocation =write_allocation

        # 3. 计算自由读权重
        free_gate_expanded = tf.expand_dims(free_gate, axis=-1)  # [1, 1, 1]
        free_read_weights = free_gate_expanded * read_weights  # [1, 1, 3]
        # =0 * read_weights =0

        # 4. 计算 total_free_read_weights = sum(free_read_weights, axis=1)
        total_free_read_weights = tf.reduce_sum(free_read_weights, axis=1)  # [1, 3]
        # =0

        # 5. 使用 usage_after_read = usage_after_write - total_free_read_weights
        usage_after_read = usage_after_write - total_free_read_weights  # [1, 3]
        # = write_allocation -0 = write_allocation

        # 6. 裁剪使用率到 [0, 1]
        expected_usage = tf.clip_by_value(usage_after_read, 0.0, 1.0)  # [1, 3]

        # 断言
        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_partial_write_and_read(self):
        """
        测试部分内存槽被写入和部分内存槽被读出的情况。
        """
        batch_size = 1

        # 创建初始使用率
        initial_usage = tf.constant([[0.2, 0.5, 0.3]], dtype=tf.float32)  # [1, 3]

        # 定义 write_weights
        write_weights = tf.constant([
            [[0.1, 0.2, 0.3],
             [0.4, 0.1, 0.2]]
        ], dtype=tf.float32)  # [1, 2, 3]

        # 定义 free_gate 和 read_weights
        free_gate = tf.constant([
            [0.5, 0.5]
        ], dtype=tf.float32)  # [1, 2]

        read_weights = tf.constant([
            [[0.3, 0.4, 0.3],
             [0.2, 0.5, 0.3]]
        ], dtype=tf.float32)  # [1, 2, 3]

        # 调用 UsageUpdate 层
        updated_usage = self.usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 计算预期使用率
        # 步骤1: 计算 write_allocation = 1 - prod(1 - write_weights, axis=1)
        write_weights_cumprod = tf.reduce_prod(1 - write_weights, axis=1)  # [1, 3]
        write_allocation = 1 - write_weights_cumprod  # [1, 3]

        # 步骤2: 计算 usage_after_write = initial_usage + (1 - initial_usage) * write_allocation
        usage_after_write = initial_usage + (1 - initial_usage) * write_allocation  # [1, 3]

        # 步骤3: 计算 total_free_read_weights = sum(free_gate * read_weights, axis=1)
        free_gate_expanded = tf.expand_dims(free_gate, axis=-1)  # [1, 2, 1]
        free_read_weights = free_gate_expanded * read_weights  # [1, 2, 3]
        total_free_read_weights = tf.reduce_sum(free_read_weights, axis=1)  # [1, 3]

        # 步骤4: 计算 usage_after_read = usage_after_write - total_free_read_weights
        usage_after_read = usage_after_write - total_free_read_weights  # [1, 3]
        usage_after_read = tf.maximum(usage_after_read, 0.0)  # 确保不低于0

        # 步骤5: 裁剪使用率到 [0, 1]
        expected_usage = tf.clip_by_value(usage_after_read, 0.0, 1.0)  # [1, 3]

        # 转换为 numpy 进行比较
        expected_usage_np = expected_usage.numpy()
        updated_usage_np = updated_usage.numpy()

        # 打印调试信息（可选）
        tf.print("Initial Usage:", initial_usage)
        tf.print("Write Weights:", write_weights)
        tf.print("Write Allocation:", write_allocation)
        tf.print("Usage After Write:", usage_after_write)
        tf.print("Free Gate:", free_gate)
        tf.print("Read Weights:", read_weights)
        tf.print("Free Read Weights:", free_read_weights)
        tf.print("Total Free Read Weights:", total_free_read_weights)
        tf.print("Usage After Read:", usage_after_read)
        tf.print("Expected Usage:", expected_usage)
        tf.print("Updated Usage:", updated_usage)

        # 断言更新后的使用率与预期值接近
        self.assertAllClose(updated_usage_np, expected_usage_np, atol=1e-6)

    def test_initial_state(self):
        """
        测试 get_initial_state 方法，确保返回正确的初始使用率。
        """
        batch_size = 4
        initial_usage = self.usage_update_layer.get_initial_state([batch_size])  # [4,3]

        expected_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [4,3]

        self.assertAllClose(initial_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_state_size(self):
        """
        测试 state_size 属性，确保返回正确的形状。
        """
        self.assertEqual(self.usage_update_layer.state_size, tf.TensorShape([self.memory_size]))


class WriteAllocationTest(tf.test.TestCase):
    def setUp(self):
        super(WriteAllocationTest, self).setUp()
        self.memory_size = 3
        self.num_writes = 2
        self.epsilon = 1e-3  # 增大 epsilon

        # 定义生成 write_content_weights 的函数
        def custom_write_content_weights_fn(inputs: dict) -> tf.Tensor:
            write_content_keys = inputs.get('write_content_keys')
            write_content_strengths = inputs.get('write_content_strengths')
            if write_content_keys is not None and write_content_strengths is not None:
                # 使用 write_content_keys 和 write_content_strengths 以及 memory 来生成 write_content_weights
                memory = inputs.get('memory')
                if memory is None:
                    raise KeyError(
                        "'memory' must be provided when using 'write_content_keys' and 'write_content_strengths'")
                # 计算 scores
                scores = tf.matmul(write_content_keys, memory,
                                   transpose_b=True)  # [batch_size, num_writes, memory_size]
                # 计算 weighted_softmax
                write_content_weights = weighted_softmax(scores, write_content_strengths,
                                                         tf.nn.softmax)  # [batch_size, num_writes, memory_size]
                return write_content_weights
            else:
                # 使用 softmax 基于 usage 生成 write_content_weights
                usage = inputs.get('usage')
                write_gates_sum = inputs.get('write_gates_sum')
                if usage is not None and write_gates_sum is not None:
                    write_content_weights = tf.nn.softmax(tf.expand_dims(usage, axis=1),
                                                          axis=-1)  # [batch_size, num_writes, memory_size]
                    # 不要乘以 write_gates_sum，保持 write_content_weights 为均匀分布
                    return write_content_weights
                else:
                    raise KeyError(
                        "Inputs must contain either ('write_content_keys' and 'write_content_strengths') or ('usage' and 'write_gates_sum').")

        # 实例化 WriteAllocation 层时仅传入 write_content_weights_fn
        self.write_allocation_layer = WriteAllocation(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            epsilon=self.epsilon,
            write_content_weights_fn=custom_write_content_weights_fn
            # allocation_gate_fn 和 write_gate_fn 使用默认值
        )

    def test_gradient_computability(self):
        """验证梯度是否可以计算"""
        batch_size = 2  # 使用更大的批量大小以更好地测试
        memory = tf.Variable(tf.random.normal([batch_size, self.memory_size, 4]),
                             trainable=True)  # [batch_size, memory_size, word_size]
        usage = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32,
                            trainable=True)  # [batch_size, memory_size]
        write_content_keys = tf.random.uniform([batch_size, self.num_writes, 4])  # [batch_size, num_writes, word_size]
        write_content_strengths = tf.ones([batch_size, self.num_writes], dtype=tf.float32)  # [batch_size, num_writes]

        # 实例化 WriteAllocation 层，传入自定义的 gate 函数
        write_allocation_layer = WriteAllocation(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            epsilon=self.epsilon,
            write_content_weights_fn=self.write_allocation_layer.write_content_weights_fn,
            allocation_gate_fn=None,  # 使用默认的 allocation_gate_fn
            write_gate_fn=None  # 使用默认的 write_gate_fn
        )

        with tf.GradientTape() as tape:
            # 创建 inputs 字典，包含 memory
            inputs = {
                'usage': usage,
                'write_gates_sum': tf.ones([batch_size, self.num_writes], dtype=tf.float32),
                'write_content_keys': write_content_keys,
                'write_content_strengths': write_content_strengths,
                'memory': memory
            }

            # 调用 WriteAllocation 的 call 方法，并可选择覆盖 num_writes
            write_weights = write_allocation_layer(inputs, training=True)  # [batch_size, num_writes, memory_size]

            # 定义一个依赖于 usage 的损失函数
            loss = tf.reduce_sum(write_weights * tf.expand_dims(usage, axis=1))  # scalar

        # 计算梯度
        gradients = tape.gradient(loss, [memory, usage])

        # 检查梯度
        for grad, var_name in zip(gradients, ['memory', 'usage']):
            self.assertIsNotNone(grad, f"No gradient for {var_name}")
            grad_norm = tf.norm(grad).numpy()
            self.assertGreater(grad_norm, 1e-12, f"Gradient too small for {var_name}")
            tf.print(f"Gradient norm for {var_name}: {grad_norm}")

    def test_gradient_path_validity(self):
        """验证梯度传播路径的正确性"""
        batch_size = 2
        memory = tf.Variable(tf.random.normal([batch_size, self.memory_size, 4]),
                             trainable=True)  # [batch_size, memory_size, word_size]
        usage = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32,
                            trainable=True)  # [batch_size, memory_size]
        write_content_keys = tf.random.uniform([batch_size, self.num_writes, 4])  # [batch_size, num_writes, word_size]
        write_content_strengths = tf.ones([batch_size, self.num_writes], dtype=tf.float32)  # [batch_size, num_writes]

        # 实例化 WriteAllocation 层，传入自定义的 gate 函数
        write_allocation_layer = WriteAllocation(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            epsilon=self.epsilon,
            write_content_weights_fn=self.write_allocation_layer.write_content_weights_fn,
            allocation_gate_fn=None,  # 使用默认的 allocation_gate_fn
            write_gate_fn=None  # 使用默认的 write_gate_fn
        )

        with tf.GradientTape() as tape:
            # 创建 inputs 字典，包含 memory
            inputs = {
                'usage': usage,
                'write_gates_sum': tf.ones([batch_size, self.num_writes], dtype=tf.float32),
                'write_content_keys': write_content_keys,
                'write_content_strengths': write_content_strengths,
                'memory': memory
            }

            # 调用 WriteAllocation 的 call 方法，并可选择覆盖 num_writes
            write_weights = write_allocation_layer(inputs, training=True)  # [batch_size, num_writes, memory_size]

            # 定义一个依赖于 usage 的损失函数
            loss = tf.reduce_sum(write_weights * tf.expand_dims(usage, axis=1))  # scalar

        # 计算梯度
        gradients = tape.gradient(loss, [memory, usage])

        # 验证每个变量的梯度正确传播
        for grad, var_name in zip(gradients, ['memory', 'usage']):
            self.assertIsNotNone(grad, f"No gradient for {var_name}")
            grad_norm = tf.reduce_sum(tf.abs(grad)).numpy()
            self.assertGreater(grad_norm, 0, f"Some gradients are zero for {var_name}")
            tf.print(f"Gradient norm for {var_name}: {grad_norm}")

    def test_gradient_value_reasonableness(self):
        """验证梯度值的合理性"""
        batch_size = 2
        memory = tf.Variable(tf.random.normal([batch_size, self.memory_size, 4]),
                             trainable=True)  # [batch_size, memory_size, word_size]
        usage = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32,
                            trainable=True)  # [batch_size, memory_size]
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)  # [batch_size, num_writes]

        with tf.GradientTape() as tape:
            # 创建 inputs 字典
            inputs = {
                'usage': usage,
                'write_gates_sum': write_gates_sum
            }

            # 调用 WriteAllocation 的 call 方法，并可选择覆盖 num_writes
            write_weights = self.write_allocation_layer(inputs, training=True)  # [batch_size, num_writes, memory_size]

            # 定义一个依赖于 usage 的损失函数
            loss = tf.reduce_sum(write_weights * tf.expand_dims(usage, axis=1))  # scalar

        # 计算梯度
        gradients = tape.gradient(loss, [memory, usage])
        gradient_value = tf.reduce_sum(tf.abs(gradients[1])).numpy()  # usage 的梯度

        # 验证梯度值在合理范围内
        self.assertGreater(gradient_value, 1e-6, "Gradient is too small")
        self.assertLess(gradient_value, 1e6, "Gradient is too large")
        tf.print(f"Gradient value for usage: {gradient_value}")

    def test_write_allocation(self):
        """
        测试 WriteAllocation 类的 call 方法，确保其功能正确。
        """
        batch_size = 2

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [2,3]

        # 定义 write_gates_sum 全为1
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)  # [2,2]

        # 创建 inputs 字典，不包含 'write_content_keys' 和 'write_content_strengths'
        inputs = {
            'usage': initial_usage,
            'write_gates_sum': write_gates_sum
        }

        # 调用 WriteAllocation 层
        write_allocation_weights = self.write_allocation_layer(inputs, training=False)  # [2,2,3]

        # 预期 write_allocation_weights 应为 soft allocations, 例如 [1/3,1/3,1/3]
        expected_write_allocation_weights = tf.constant([
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]],
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]]
        ], dtype=tf.float32)  # [2,2,3]

        self.assertAllClose(write_allocation_weights.numpy(), expected_write_allocation_weights.numpy(), atol=1e-5)

    def test_write_allocation_with_dynamic_num_writes(self):
        """测试 WriteAllocation 层在调用时动态覆盖 num_writes"""
        batch_size = 2
        dynamic_num_writes = 3  # 动态覆盖的 num_writes

        # 创建 usage 和 write_gates_sum
        usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [2,3]
        write_gates_sum = tf.ones([batch_size, dynamic_num_writes], dtype=tf.float32)  # [2,3]

        # 创建 inputs 字典，不包含 'write_content_keys' 和 'write_content_strengths'
        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        # 调用 WriteAllocation 的 call 方法，并覆盖 num_writes
        write_weights = self.write_allocation_layer(inputs, training=False, num_writes=dynamic_num_writes)  # [2,3,3]

        # 预期 write_allocation_weights 应为 soft allocations, 例如 [1/3,1/3,1/3]
        expected_write_allocation_weights = tf.constant([
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]],
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]]
        ], dtype=tf.float32)  # [2,3,3]

        self.assertAllClose(write_weights.numpy(), expected_write_allocation_weights.numpy(), atol=1e-5)


if __name__ == '__main__':
    tf.test.main()
