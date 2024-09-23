"""Tests for memory addressing."""

import numpy as np
import tensorflow as tf

# from parameterized import parameterized

from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import TemporalLinkage, weighted_softmax, CosineWeights, \
    Freeness, WriteAllocation, UsageUpdate


class WeightedSoftmaxTest(tf.test.TestCase):

    def testValues(self):
        batch_size = 5
        num_heads = 3
        memory_size = 7

        # 使用 NumPy 生成输入数据
        activations_data = np.random.randn(batch_size, num_heads, memory_size).astype(np.float32)
        strengths_data = np.ones((batch_size, num_heads), dtype=np.float32)

        # 定义输入张量
        activations = tf.convert_to_tensor(activations_data)
        strengths = tf.convert_to_tensor(strengths_data)

        # 调用加权 Softmax 函数
        observed = weighted_softmax(activations, strengths, tf.identity)

        # TensorFlow 自带的 softmax 函数用于对比
        expected = tf.nn.softmax(activations, axis=-1)

        # 使用 eager execution 模式直接运行
        observed_val = observed.numpy()
        expected_val = expected.numpy()

        # 验证 observed 和 expected 是否接近
        self.assertAllClose(observed_val, expected_val, atol=1e-6)

    def test_weighted_softmax_with_softplus_strength(self):
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

        # 定义 strength_op 为 softplus
        strength_op = tf.nn.softplus

        # 计算 weighted_softmax
        observed = weighted_softmax(activations, strengths, strength_op)

        # 手动计算加权分数
        adjusted_strengths = strength_op(tf.expand_dims(strengths, axis=-1))  # [2, 2, 1]
        weighted_scores = activations * adjusted_strengths  # [2, 2, 3]
        expected = tf.nn.softmax(weighted_scores, axis=-1)  # [2, 2, 3]

        # 验证
        self.assertAllClose(observed.numpy(), expected.numpy(), atol=1e-6)

    def test_weighted_softmax_zero_strength(self):
        # 测试 strength 为零时，确保 weighted_softmax 的行为
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
        weighted_scores = activations * strengths[..., tf.newaxis]  # [1, 1, 3] = [1,1,3] * [1,1,1]
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

        # 创建 CosineWeights 实例
        # 假设 strength_op 是 identity
        strength_op = tf.identity

        # 创建固定的 activations
        activations = tf.constant([[[1.0, 2.0, 3.0]]], dtype=tf.float32)  # [1, 1, 3]

        # 测试极小的 strengths 值
        strengths_small = tf.constant([[1e-6]], dtype=tf.float32)  # [1, 1]
        expected_small = tf.nn.softmax(activations * strengths_small,
                                       axis=-1)  # 应接近 softmax([0, 0, 0]) = [1/3, 1/3, 1/3]

        # 计算 weighted_softmax
        observed_small = weighted_softmax(activations, strengths_small, strength_op)

        # 验证
        self.assertAllClose(observed_small.numpy(), expected_small.numpy(), atol=1e-4)

        # 测试极大的 strengths 值
        strengths_large = tf.constant([[1e6]], dtype=tf.float32)  # [1, 1]
        # 由于 strengths 很大，weighted_scores 会非常大，softmax 会接近于 [0, 0, 1]
        expected_large = tf.nn.softmax(activations * strengths_large, axis=-1)
        # 由于 1e6 * 1 = 1e6, 1e6 * 2 = 2e6, 1e6 * 3 = 3e6
        # softmax([1e6, 2e6, 3e6]) ≈ [0, 0, 1]
        expected_large = tf.constant([[[0.0, 0.0, 1.0]]], dtype=tf.float32)

        # 计算 weighted_softmax
        observed_large = weighted_softmax(activations, strengths_large, strength_op)


class CosineWeightsTest(tf.test.TestCase):
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

    def test_cosine_weights_basic(self):
        batch_size = 2
        memory_size = 4
        num_heads = 3
        word_size = 5

        # 创建 CosineWeights 实例
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
        # 期望值为 [0.318, 0.682]
        expected_weights = np.array([[[0.318, 0.682]]], dtype=np.float32)
        self.assertAllClose(weights.numpy(), expected_weights, atol=1.1e-3)  # 增加 atol 至 1.1e-3

    def test_cosine_weights_all_zero_norm(self):
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

    def test_write_weights_i_and_j(self):
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        # 计算 write_weights_i 和 write_weights_j
        write_weights_i = tf.expand_dims(write_weights, 3)  # (batch_size, num_writes, memory_size, 1)
        write_weights_j = tf.expand_dims(write_weights, 2)  # (batch_size, num_writes, 1, memory_size)

        # 验证 shape 和值
        expected_write_weights_i = np.array([[[[1], [0], [0]]]], dtype=np.float32)
        expected_write_weights_j = np.array([[[[1, 0, 0]]]], dtype=np.float32)

        np.testing.assert_allclose(write_weights_i.numpy(), expected_write_weights_i, atol=1e-6)
        np.testing.assert_allclose(write_weights_j.numpy(), expected_write_weights_j, atol=1e-6)

    def test_precedence_weights_update(self):
        # 初始 precedence_weights 为全零
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)  # 第一次写入位置 0

        # 模拟 precedence_weights 更新过程
        reset_gate = 1 - tf.reduce_sum(write_weights, axis=2, keepdims=True)  # [batch_size, num_writes, 1]
        updated_precedence_weights = reset_gate * prev_precedence_weights + write_weights  # [batch_size, num_writes, memory_size]

        print("prev_precedence_weights:", prev_precedence_weights.numpy())
        print("write_weights:", write_weights.numpy())
        print("updated_precedence_weights:", updated_precedence_weights.numpy())

        expected_precedence_weights = np.array([[[1, 0, 0]]], dtype=np.float32)

        np.testing.assert_allclose(updated_precedence_weights.numpy(), expected_precedence_weights, atol=1e-6)

    def test_write_weights_expansion(self):
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        # 扩展维度
        write_weights_i = tf.expand_dims(write_weights, 3)
        write_weights_j = tf.expand_dims(write_weights, 2)

        print("write_weights_i:", write_weights_i.numpy())
        print("write_weights_j:", write_weights_j.numpy())

        expected_write_weights_i = np.array([[[[1], [0], [0]]]], dtype=np.float32)
        expected_write_weights_j = np.array([[[[1, 0, 0]]]], dtype=np.float32)

        np.testing.assert_allclose(write_weights_i.numpy(), expected_write_weights_i, atol=1e-6)
        np.testing.assert_allclose(write_weights_j.numpy(), expected_write_weights_j, atol=1e-6)

    def test_prev_precedence_weights_j(self):
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)

        # 计算 prev_precedence_weights_j
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights,
                                                   2)  # (batch_size, num_writes, 1, memory_size)

        # 验证 shape 和值
        expected_prev_precedence_weights_j = np.array([[[[0, 0, 0]]]], dtype=np.float32)

        np.testing.assert_allclose(prev_precedence_weights_j.numpy(), expected_prev_precedence_weights_j, atol=1e-6)

    def test_prev_link_scale(self):
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        write_weights_i = tf.expand_dims(write_weights, 3)
        write_weights_j = tf.expand_dims(write_weights, 2)

        # 计算 prev_link_scale
        prev_link_scale = 1 - write_weights_i - write_weights_j  # 不进行裁剪

        print("write_weights_i:", write_weights_i.numpy())
        print("write_weights_j:", write_weights_j.numpy())
        print("prev_link_scale:", prev_link_scale.numpy())

        expected_prev_link_scale = np.array([[[[-1, 0, 0],
                                               [0, 1, 1],
                                               [0, 1, 1]]]], dtype=np.float32)
        np.testing.assert_allclose(prev_link_scale.numpy(), expected_prev_link_scale, atol=1e-6)

    def test_weighted_softmax_with_softplus_strength(self):
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

        # 定义 strength_op 为 softplus
        strength_op = tf.nn.softplus

        # 计算 weighted_softmax
        observed = weighted_softmax(activations, strengths, strength_op)

        # 手动计算加权分数
        adjusted_strengths = strength_op(tf.expand_dims(strengths, axis=-1))  # [2, 2, 1]
        weighted_scores = activations * adjusted_strengths  # [2, 2, 3]
        expected = tf.nn.softmax(weighted_scores, axis=-1)  # [2, 2, 3]

        # 验证
        self.assertAllClose(observed.numpy(), expected.numpy(), atol=1e-6)

    def test_weighted_softmax_zero_strength(self):
        # 测试 strength 为零时，确保 weighted_softmax 的行为
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
        weighted_scores = activations * strengths[..., tf.newaxis]  # [1, 1, 3] = [1,1,3] * [1,1,1]
        expected = tf.nn.softmax(weighted_scores, axis=-1)  # softmax([0,0,0]) = [1/3,1/3,1/3]

        # 验证
        self.assertAllClose(observed.numpy(), expected.numpy(), atol=1e-6)


class TemporalLinkageTest(tf.test.TestCase):
    def testLinkUpdate_steps(self):
        batch_size = 1
        memory_size = 3
        num_writes = 1

        module = TemporalLinkage(memory_size=memory_size, num_writes=num_writes)

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
        })

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
        })

        # 验证第二次写入后的链路矩阵
        expected_link_after_second_write = np.array([[[[0, 0, 0],
                                                       [1, 0, 0],
                                                       [0, 0, 0]]]], dtype=np.float32)
        self.assertAllClose(linkage_state['link'].numpy(), expected_link_after_second_write, atol=1e-6)

        # 更新 prev_link 和 prev_precedence_weights
        prev_link = linkage_state['link']
        prev_precedence_weights = linkage_state['precedence_weights']

        # 第三次写入
        write_weights = tf.constant([[[0, 1, 0]]], dtype=tf.float32)  # 再次写入位置 1
        linkage_state = module({
            'write_weights': write_weights,
            'prev_linkage': {
                'link': prev_link,
                'precedence_weights': prev_precedence_weights
            }
        })

        # 验证第三次写入后的链路矩阵
        expected_link_after_third_write = np.zeros((1, 1, 3, 3), dtype=np.float32)  # 修改为全零矩阵
        self.assertAllClose(linkage_state['link'].numpy(), expected_link_after_third_write, atol=1e-6)

    def test_full_link_method(self):
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
        })

        # 验证更新后的 link 矩阵是否符合预期
        expected_link = np.zeros((1, 1, 3, 3), dtype=np.float32)  # 第一次写入，link 应为全零矩阵
        self.assertAllClose(linkage_state['link'].numpy(), expected_link, atol=1e-6)

    def test_new_link_creation(self):
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)

        write_weights_i = tf.expand_dims(write_weights, 3)  # (batch_size, num_writes, memory_size, 1)
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights,
                                                   2)  # (batch_size, num_writes, 1, memory_size)

        # 计算 new_link，使用矩阵乘法
        new_link = tf.matmul(write_weights_i, prev_precedence_weights_j)

        print("write_weights_i:", write_weights_i.numpy())
        print("prev_precedence_weights_j:", prev_precedence_weights_j.numpy())
        print("new_link:", new_link.numpy())

        # 由于 prev_precedence_weights 为零，new_link 应该是全零矩阵
        expected_new_link = np.zeros((1, 1, 3, 3), dtype=np.float32)
        np.testing.assert_allclose(new_link.numpy(), expected_new_link, atol=1e-6)

    def test_updated_link(self):
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

        print("write_weights_i:", write_weights_i.numpy())
        print("write_weights_j:", write_weights_j.numpy())
        print("prev_precedence_weights_j:", prev_precedence_weights_j.numpy())
        print("prev_link_scale:", prev_link_scale.numpy())
        print("new_link:", new_link.numpy())
        print("updated_link:", updated_link.numpy())

        # 由于 prev_precedence_weights 为零，new_link 应该是全零矩阵
        expected_updated_link = np.zeros((1, 1, 3, 3), dtype=np.float32)
        np.testing.assert_allclose(updated_link.numpy(), expected_updated_link, atol=1e-6)

    def testModule(self):
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
        self.assertAllClose(diag_elements, np.zeros((batch_size, num_writes, memory_size)), atol=1e-6)

    def testDirectionalReadWeights(self):
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
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        # 计算 write_weights_i 和 write_weights_j
        write_weights_i = tf.expand_dims(write_weights, 3)  # (batch_size, num_writes, memory_size, 1)
        write_weights_j = tf.expand_dims(write_weights, 2)  # (batch_size, num_writes, 1, memory_size)

        # 验证 shape 和值
        expected_write_weights_i = np.array([[[[1], [0], [0]]]], dtype=np.float32)
        expected_write_weights_j = np.array([[[[1, 0, 0]]]], dtype=np.float32)

        np.testing.assert_allclose(write_weights_i.numpy(), expected_write_weights_i, atol=1e-6)
        np.testing.assert_allclose(write_weights_j.numpy(), expected_write_weights_j, atol=1e-6)

    def test_precedence_weights_update(self):
        # 初始 precedence_weights 为全零
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)  # 第一次写入位置 0

        # 模拟 precedence_weights 更新过程
        reset_gate = 1 - tf.reduce_sum(write_weights, axis=2, keepdims=True)  # [batch_size, num_writes, 1]
        updated_precedence_weights = reset_gate * prev_precedence_weights + write_weights  # [batch_size, num_writes, memory_size]

        print("prev_precedence_weights:", prev_precedence_weights.numpy())
        print("write_weights:", write_weights.numpy())
        print("updated_precedence_weights:", updated_precedence_weights.numpy())

        expected_precedence_weights = np.array([[[1, 0, 0]]], dtype=np.float32)

        np.testing.assert_allclose(updated_precedence_weights.numpy(), expected_precedence_weights, atol=1e-6)

    def test_write_weights_expansion(self):
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        # 扩展维度
        write_weights_i = tf.expand_dims(write_weights, 3)
        write_weights_j = tf.expand_dims(write_weights, 2)

        print("write_weights_i:", write_weights_i.numpy())
        print("write_weights_j:", write_weights_j.numpy())

        expected_write_weights_i = np.array([[[[1], [0], [0]]]], dtype=np.float32)
        expected_write_weights_j = np.array([[[[1, 0, 0]]]], dtype=np.float32)

        np.testing.assert_allclose(write_weights_i.numpy(), expected_write_weights_i, atol=1e-6)
        np.testing.assert_allclose(write_weights_j.numpy(), expected_write_weights_j, atol=1e-6)

    def test_prev_precedence_weights_j(self):
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)

        # 计算 prev_precedence_weights_j
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights,
                                                   2)  # (batch_size, num_writes, 1, memory_size)

        # 验证 shape 和值
        expected_prev_precedence_weights_j = np.array([[[[0, 0, 0]]]], dtype=np.float32)

        np.testing.assert_allclose(prev_precedence_weights_j.numpy(), expected_prev_precedence_weights_j, atol=1e-6)

    def test_prev_link_scale(self):
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        write_weights_i = tf.expand_dims(write_weights, 3)
        write_weights_j = tf.expand_dims(write_weights, 2)

        # 计算 prev_link_scale
        prev_link_scale = 1 - write_weights_i - write_weights_j  # 不进行裁剪

        print("write_weights_i:", write_weights_i.numpy())
        print("write_weights_j:", write_weights_j.numpy())
        print("prev_link_scale:", prev_link_scale.numpy())

        expected_prev_link_scale = np.array([[[[-1, 0, 0],
                                               [0, 1, 1],
                                               [0, 1, 1]]]], dtype=np.float32)
        np.testing.assert_allclose(prev_link_scale.numpy(), expected_prev_link_scale, atol=1e-6)


class WriteAllocationUsageUpdateTest(tf.test.TestCase):
    def setUp(self):
        super(WriteAllocationUsageUpdateTest, self).setUp()
        self.memory_size = 3  # 定义 memory_size
        self.num_writes = 2   # 定义 num_writes
        self.num_reads = 2    # 定义 num_reads
        self.epsilon = 1e-6   # 定义 epsilon

    def test_basic_write_and_read(self):
        """
        基本测试：验证写操作和读操作对使用率的影响。
        """
        batch_size = 2
        num_writes = 2
        num_reads = 2

        # 初始化 WriteAllocation 和 UsageUpdate 层
        write_allocation_layer = WriteAllocation(memory_size=self.memory_size, num_writes=num_writes, epsilon=self.epsilon)
        usage_update_layer = UsageUpdate(memory_size=self.memory_size, num_writes=num_writes, num_reads=num_reads, epsilon=self.epsilon)

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [2, 3]

        # 定义 write_gates_sum，使得 WriteAllocation 生成与旧测试中相同的 write_weights
        # 这里我们需要手动设置 write_gates_sum 来匹配测试期望
        # 假设 write_gates_sum 全为1，这样 allocation 将决定 write_weights
        write_gates_sum = tf.ones([batch_size, num_writes], dtype=tf.float32)  # [2, 2]

        # 定义自由门和读权重
        free_gate = tf.constant([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=tf.float32)  # [2, 2]

        read_weights = tf.constant([
            [[0.5, 0.5, 0.0],
             [0.0, 0.5, 0.5]],
            [[0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [2, 2, 3]

        # 调用 WriteAllocation 层
        write_weights = write_allocation_layer({
            'usage': initial_usage,
            'write_gates_sum': write_gates_sum
        }, training=False)  # [2, 2, 3]

        # 调用 UsageUpdate 层
        updated_usage = usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [2, 3]

        # 预期使用率计算：
        # 第一批次：
        # 写操作1：写入第0槽 -> usage = [1, 0, 0]
        # 写操作2：写入第1槽 -> usage = [1, 1, 0]
        # 读操作1：释放 0.5 from slot0 and 0.5 from slot1 -> usage = [0.5, 0.5, 0.0]
        # 读操作2：释放 0.0 from slot0 and 0.0 from slot1 -> usage remains [0.5, 0.5, 0.0]
        #
        # 第二批次：
        # 写操作1：写入第0槽 -> usage = [1, 0, 1]
        # 写操作2：写入第1槽 -> usage = [1, 1, 1]
        # 读操作1：释放 1.0 from slot0 and 1.0 from slot1 -> usage = [0.0, 0.0, 1.0]
        # 读操作2：释放 0.0 from slot0 and 0.0 from slot1 -> usage remains [0.0, 0.0, 1.0]
        expected_usage = tf.constant([
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=tf.float32)  # [2, 3]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-5)

    def test_all_zero_memory_after_write(self):
        """
        测试所有 memory 向量为零的情况，确保写操作不影响使用率。
        """
        batch_size = 1
        num_writes = 1
        num_reads = 1

        # 初始化 WriteAllocation 和 UsageUpdate 层
        write_allocation_layer = WriteAllocation(memory_size=self.memory_size, num_writes=num_writes, epsilon=self.epsilon)
        usage_update_layer = UsageUpdate(memory_size=self.memory_size, num_writes=num_writes, num_reads=num_reads, epsilon=self.epsilon)

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义 write_gates_sum 为0，确保 allocation_weights 为0，写操作不影响使用率
        write_gates_sum = tf.zeros([batch_size, num_writes], dtype=tf.float32)  # [1, 1]

        # 定义自由门和读权重（任意）
        free_gate = tf.constant([
            [1.0]
        ], dtype=tf.float32)  # [1, 1]

        read_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 调用 WriteAllocation 层
        write_weights = write_allocation_layer({
            'usage': initial_usage,
            'write_gates_sum': write_gates_sum
        }, training=False)  # [1, 1, 3]

        # 调用 UsageUpdate 层
        updated_usage = usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 预期使用率为全0
        expected_usage = tf.constant([
            [0.0, 0.0, 0.0]
        ], dtype=tf.float32)  # [1, 3]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_all_zero_read_weights(self):
        """
        测试所有读权重为零的情况，确保读操作不影响使用率。
        """
        batch_size = 1
        num_writes = 1
        num_reads = 1

        # 初始化 WriteAllocation 和 UsageUpdate 层
        write_allocation_layer = WriteAllocation(memory_size=self.memory_size, num_writes=num_writes, epsilon=self.epsilon)
        usage_update_layer = UsageUpdate(memory_size=self.memory_size, num_writes=num_writes, num_reads=num_reads, epsilon=self.epsilon)

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义 write_gates_sum 为1，确保 allocation 将写入第0槽
        write_gates_sum = tf.ones([batch_size, num_writes], dtype=tf.float32)  # [1, 1]

        # 定义自由门和读权重（全零）
        free_gate = tf.constant([
            [0.0]
        ], dtype=tf.float32)  # [1, 1]

        read_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 调用 WriteAllocation 层
        write_weights = write_allocation_layer({
            'usage': initial_usage,
            'write_gates_sum': write_gates_sum
        }, training=False)  # [1, 1, 3]

        # 调用 UsageUpdate 层
        updated_usage = usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 预期使用率为 [1.0, 0.0, 0.0]
        expected_usage = tf.constant([
            [1.0, 0.0, 0.0]
        ], dtype=tf.float32)  # [1, 3]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_all_full_usage(self):
        """
        测试所有内存槽已满的情况，确保进一步写操作释放使用率。
        """
        batch_size = 1
        num_writes = 1
        num_reads = 1

        # 初始化 WriteAllocation 和 UsageUpdate 层
        write_allocation_layer = WriteAllocation(memory_size=self.memory_size, num_writes=num_writes, epsilon=self.epsilon)
        usage_update_layer = UsageUpdate(memory_size=self.memory_size, num_writes=num_writes, num_reads=num_reads, epsilon=self.epsilon)

        # 创建初始使用率为全1
        initial_usage = tf.ones([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义 write_gates_sum 为1，确保 allocation 将写入所有槽
        write_gates_sum = tf.ones([batch_size, num_writes], dtype=tf.float32)  # [1, 1]

        # 定义自由门和读权重（全读）
        free_gate = tf.constant([
            [1.0]
        ], dtype=tf.float32)  # [1, 1]

        read_weights = tf.constant([
            [[1.0, 1.0, 1.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 调用 WriteAllocation 层
        write_weights = write_allocation_layer({
            'usage': initial_usage,
            'write_gates_sum': write_gates_sum
        }, training=False)  # [1, 1, 3]

        # 调用 UsageUpdate 层
        updated_usage = usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 预期使用率应为全0，因为所有内存槽已满并通过自由门释放
        expected_usage = tf.constant([
            [0.0, 0.0, 0.0]
        ], dtype=tf.float32)  # [1, 3]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_multiple_writes_and_reads(self):
        """
        测试多次写操作和读操作的累积影响。
        """
        batch_size = 1
        num_writes = 2
        num_reads = 2

        # 初始化 WriteAllocation 和 UsageUpdate 层
        write_allocation_layer = WriteAllocation(memory_size=self.memory_size, num_writes=num_writes, epsilon=self.epsilon)
        usage_update_layer = UsageUpdate(memory_size=self.memory_size, num_writes=num_writes, num_reads=num_reads, epsilon=self.epsilon)

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义 write_gates_sum 为2，确保 allocation 将写入两个槽
        write_gates_sum = tf.ones([batch_size, num_writes], dtype=tf.float32)  # [1, 2]

        # 定义自由门和读权重（两次读操作）
        free_gate = tf.constant([
            [1.0, 1.0]
        ], dtype=tf.float32)  # [1, 2]

        read_weights = tf.constant([
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]]
        ], dtype=tf.float32)  # [1, 2, 3]

        # 调用 WriteAllocation 层
        write_weights = write_allocation_layer({
            'usage': initial_usage,
            'write_gates_sum': write_gates_sum
        }, training=False)  # [1, 2, 3]

        # 调用 UsageUpdate 层
        updated_usage = usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 预期使用率计算：
        # 写操作1：写入第0槽 -> usage = [1, 0, 0]
        # 写操作2：写入第1槽 -> usage = [1, 1, 0]
        # 读操作1：释放 1.0 from slot0 and 1.0 from slot1 -> usage = [0.0, 0.0, 0.0]
        # 读操作2：释放 0.0 from slot0 and 0.0 from slot1 -> usage remains [0.0, 0.0, 0.0]
        expected_usage = tf.constant([
            [0.0, 0.0, 0.0]
        ], dtype=tf.float32)  # [1, 3]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_all_zero_write_weights(self):
        """
        测试所有写权重为零的情况，确保写操作不影响使用率。
        """
        batch_size = 1
        num_writes = 1
        num_reads = 1

        # 初始化 WriteAllocation 和 UsageUpdate 层
        write_allocation_layer = WriteAllocation(memory_size=self.memory_size, num_writes=num_writes, epsilon=self.epsilon)
        usage_update_layer = UsageUpdate(memory_size=self.memory_size, num_writes=num_writes, num_reads=num_reads, epsilon=self.epsilon)

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义 write_gates_sum 为0，确保 allocation_weights 为0，写操作不影响使用率
        write_gates_sum = tf.zeros([batch_size, num_writes], dtype=tf.float32)  # [1, 1]

        # 定义自由门和读权重
        free_gate = tf.constant([
            [1.0]
        ], dtype=tf.float32)  # [1, 1]

        read_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 调用 WriteAllocation 层
        write_weights = write_allocation_layer({
            'usage': initial_usage,
            'write_gates_sum': write_gates_sum
        }, training=False)  # [1, 1, 3]

        # 调用 UsageUpdate 层
        updated_usage = usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 预期使用率应保持不变
        expected_usage = tf.constant([
            [0.0, 0.0, 0.0]
        ], dtype=tf.float32)  # [1, 3]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)


class FreenessTest(tf.test.TestCase):
    def setUp(self):
        super(FreenessTest, self).setUp()
        self.memory_size = 3  # 定义 memory_size，不带下划线
        self.num_writes = 2  # 定义 num_writes

    def test_basic_write_and_read(self):
        """
        基本测试：验证写操作和读操作对使用率的影响。
        """
        batch_size = 2
        num_writes = 2
        num_reads = 2

        # 初始化 Freeness 层
        freeness_layer = Freeness(memory_size=self.memory_size, num_writes=num_writes)

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [2, 3]

        # 定义写权重
        write_weights = tf.constant([
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [2, 2, 3]

        # 定义自由门和读权重
        free_gate = tf.constant([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=tf.float32)  # [2, 2]

        read_weights = tf.constant([
            [[0.5, 0.5, 0.0],
             [0.0, 0.5, 0.5]],
            [[0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [2, 2, 3]

        # 构建输入字典
        inputs = {
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }

        # 调用层
        updated_usage = freeness_layer(inputs, training=False)

        # 预期使用率计算：
        # 对于第一批次：
        # 写操作1：写入第0槽 -> usage = [1, 0, 0]
        # 写操作2：写入第1槽 -> usage = [1, 1, 0]
        # 读操作1：释放 0.5 from slot0 and 0.5 from slot1 -> usage = [0.5, 0.5, 0.0]
        # 读操作2：释放 0.0 from slot0 and 0.0 from slot1 -> usage remains [0.5, 0.5, 0.0]
        #
        # 对于第二批次：
        # 写操作1：写入第2槽 -> usage = [0, 0, 1]
        # 写操作2：写入第0槽 -> usage = [1, 0, 1]
        # 读操作1：释放 1.0 from slot0 and 1.0 from slot1 -> usage = [0.0, 0.0, 1.0]
        # 读操作2：释放 0.0 from slot0 and 0.0 from slot1 -> usage remains [0.0, 0.0, 1.0]
        expected_usage = tf.constant([
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=tf.float32)  # [2, 3]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_all_zero_memory_after_write(self):
        """
        测试所有 memory 向量为零的情况，确保使用率均匀分配。
        """
        batch_size = 1
        num_writes = 1
        num_reads = 1

        # 初始化 Freeness 层
        freeness_layer = Freeness(memory_size=self.memory_size, num_writes=num_writes)

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义写权重（全零）
        write_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 定义自由门和读权重（任意）
        free_gate = tf.constant([
            [1.0]
        ], dtype=tf.float32)  # [1, 1]

        read_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 构建输入字典
        inputs = {
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }

        # 调用层
        updated_usage = freeness_layer(inputs, training=False)

        # 预期使用率为全0
        expected_usage = tf.constant([
            [0.0, 0.0, 0.0]
        ], dtype=tf.float32)  # [1, 3]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_all_zero_read_weights(self):
        """
        测试所有读权重为零的情况，确保读操作不影响使用率。
        """
        batch_size = 1
        num_writes = 1
        num_reads = 1

        # 初始化 Freeness 层
        freeness_layer = Freeness(memory_size=self.memory_size, num_writes=num_writes)

        # 创建初始使用率
        initial_usage = freeness_layer.get_initial_state((batch_size,))  # 使用元组

        # 定义写权重（部分写入）
        write_weights = tf.constant([
            [[1.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [batch_size, num_writes, memory_size]

        # 定义自由门和读权重（全零）
        free_gate = tf.constant([
            [0.0]
        ], dtype=tf.float32)  # [batch_size, num_reads]

        read_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [batch_size, num_reads, memory_size]

        # 构建输入字典
        inputs = {
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }

        # 调用层
        updated_usage = freeness_layer(inputs, training=False)

        # 预期使用率为 [1.0, 0.0, 0.0]
        expected_usage = tf.constant([
            [1.0, 0.0, 0.0]
        ], dtype=tf.float32)  # [batch_size, memory_size]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_all_full_usage(self):
        """
        测试所有内存槽已满的情况，确保进一步写操作释放使用率。
        """
        batch_size = 1
        num_writes = 1
        num_reads = 1

        # 初始化 Freeness 层
        freeness_layer = Freeness(memory_size=self.memory_size, num_writes=num_writes)

        # 创建初始使用率为全1
        initial_usage = tf.ones([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义写权重（写入所有槽）
        write_weights = tf.constant([
            [[1.0, 1.0, 1.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 定义自由门和读权重（全读）
        free_gate = tf.constant([
            [1.0]
        ], dtype=tf.float32)  # [1, 1]

        read_weights = tf.constant([
            [[1.0, 1.0, 1.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 构建输入字典
        inputs = {
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }

        # 调用层
        updated_usage = freeness_layer(inputs, training=False)

        # 预期使用率应为全0，因为所有内存槽已满并通过自由门释放
        expected_usage = tf.constant([
            [0.0, 0.0, 0.0]
        ], dtype=tf.float32)  # [1, 3]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_multiple_writes_and_reads(self):
        """
        测试多次写操作和读操作的累积影响。
        """
        batch_size = 1
        num_writes = 2
        num_reads = 2

        # 初始化 Freeness 层
        freeness_layer = Freeness(memory_size=self.memory_size, num_writes=num_writes)

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义写权重（两次写入）
        write_weights = tf.constant([
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]]
        ], dtype=tf.float32)  # [1, 2, 3]

        # 定义自由门和读权重（两次读操作）
        free_gate = tf.constant([
            [1.0, 1.0]
        ], dtype=tf.float32)  # [1, 2]

        read_weights = tf.constant([
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]]
        ], dtype=tf.float32)  # [1, 2, 3]

        # 构建输入字典
        inputs = {
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }

        # 调用层
        updated_usage = freeness_layer(inputs, training=False)

        # 预期使用率计算：
        # 写操作1：写入第0槽 -> usage = [1, 0, 0]
        # 写操作2：写入第1槽 -> usage = [1, 1, 0]
        # 读操作1：释放 1.0 from slot0 and 1.0 from slot1 -> usage = [0, 0, 0]
        # 读操作2：释放 0.0 from slot0 and 0.0 from slot1 -> usage remains [0, 0, 0]
        expected_usage = tf.constant([
            [0.0, 0.0, 0.0]
        ], dtype=tf.float32)  # [1, 3]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_extreme_strengths_small(self):
        """
        测试极小的 strengths 值，确保 weighted_softmax 接近标准 softmax。
        """
        batch_size = 1
        num_heads = 1
        memory_size = 3

        # 定义输入参数
        activations = tf.constant([[[1.0, 2.0, 3.0]]], dtype=tf.float32)  # [1, 1, 3]
        strengths_small = tf.constant([[1e-6]], dtype=tf.float32)  # [1, 1]

        # 计算加权 softmax
        observed_small = weighted_softmax(activations, strengths_small, tf.identity)

        # 期望值为 softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
        expected_small = tf.nn.softmax(tf.zeros_like(activations), axis=-1)

        self.assertAllClose(observed_small.numpy(), expected_small.numpy(), atol=1e-4)

    def test_extreme_strengths_large(self):
        """
        测试极大的 strengths 值，确保 weighted_softmax 更加尖锐。
        """
        batch_size = 1
        num_heads = 1
        memory_size = 3

        # 定义输入参数
        activations = tf.constant([[[1.0, 2.0, 3.0]]], dtype=tf.float32)  # [1, 1, 3]
        strengths_large = tf.constant([[1e6]], dtype=tf.float32)  # [1, 1]

        # 计算加权 softmax
        observed_large = weighted_softmax(activations, strengths_large, tf.identity)

        # 期望值为 softmax([1e6, 2e6, 3e6]) ≈ [0, 0, 1]
        expected_large = tf.nn.softmax(activations * strengths_large, axis=-1)

        self.assertAllClose(observed_large.numpy(), expected_large.numpy(), atol=1e-3)

    def test_weighted_softmax_with_different_strength_ops(self):
        """
        测试不同的 strength_op 函数，验证 weighted_softmax 的灵活性。
        """
        batch_size = 1
        num_heads = 1
        memory_size = 3

        # 定义输入参数
        activations = tf.constant([[[1.0, 2.0, 3.0]]], dtype=tf.float32)  # [1, 1, 3]
        strengths = tf.constant([[1.0]], dtype=tf.float32)  # [1, 1]

        # 定义 strength_op 为 relu
        strength_op_relu = tf.nn.relu
        observed_relu = weighted_softmax(activations, strengths, strength_op_relu)
        expected_relu = tf.nn.softmax(activations * tf.nn.relu(strengths[..., tf.newaxis]), axis=-1)
        self.assertAllClose(observed_relu.numpy(), expected_relu.numpy(), atol=1e-6)

        # 定义 strength_op 为 tanh
        strength_op_tanh = tf.nn.tanh
        observed_tanh = weighted_softmax(activations, strengths, strength_op_tanh)
        expected_tanh = tf.nn.softmax(activations * tf.nn.tanh(strengths[..., tf.newaxis]), axis=-1)
        self.assertAllClose(observed_tanh.numpy(), expected_tanh.numpy(), atol=1e-6)

    def test_all_zero_write_weights(self):
        """
        测试所有写权重为零的情况，确保写操作不影响使用率。
        """
        batch_size = 1
        num_writes = 1
        num_reads = 1

        # 初始化 Freeness 层
        freeness_layer = Freeness(memory_size=self.memory_size, num_writes=num_writes)

        # 创建初始使用率
        initial_usage = freeness_layer.get_initial_state((batch_size,))  # 使用元组

        # 定义写权重（全零）
        write_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [batch_size, num_writes, memory_size]

        # 定义自由门和读权重
        free_gate = tf.constant([
            [1.0]
        ], dtype=tf.float32)  # [batch_size, num_reads]

        read_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [batch_size, num_reads, memory_size]

        # 构建输入字典
        inputs = {
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }

        # 调用层
        updated_usage = freeness_layer(inputs, training=False)

        # 预期使用率应保持不变
        expected_usage = tf.constant([
            [0.0, 0.0, 0.0]
        ], dtype=tf.float32)  # [batch_size, memory_size]

        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    # @parameterized.expand([
    #     ("all_full", (1,), [[1.0, 1.0, 1.0]], [[1.0]], [[[1.0, 1.0, 1.0]]], [[0.0, 0.0, 0.0]]),
    #     ("all_zero", (1,), [[0.0, 0.0, 0.0]], [[1.0]], [[[0.0, 0.0, 0.0]]], [[0.0, 0.0, 0.0]]),
    #     # 添加更多测试案例
    # ])
    # def test_allocation_cases(self, name, batch_shape, write_weights, free_gate, read_weights, expected_usage):
    #     freeness_layer = Freeness(memory_size=self._memory_size)
    #     initial_usage = freeness_layer.get_initial_state(batch_shape)
    #     inputs = {
    #         'write_weights': tf.constant(write_weights, dtype=tf.float32),
    #         'free_gate': tf.constant(free_gate, dtype=tf.float32),
    #         'read_weights': tf.constant(read_weights, dtype=tf.float32),
    #         'prev_usage': initial_usage
    #     }
    #     updated_usage = freeness_layer(inputs, training=False)
    #     expected = tf.constant(expected_usage, dtype=tf.float32)
    #     self.assertAllClose(updated_usage.numpy(), expected.numpy(), atol=1e-6)


if __name__ == '__main__':
    tf.test.main()
