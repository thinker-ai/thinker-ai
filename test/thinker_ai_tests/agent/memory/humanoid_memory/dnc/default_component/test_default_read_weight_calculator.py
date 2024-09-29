# test_default_read_weight_calculator.py

import numpy as np
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import DefaultReadWeightCalculator, DefaultTemporalLinkageUpdater

class DefaultReadWeightCalculatorTest(tf.test.TestCase):
    def test_compute_read_weights_basic(self):
        """
        测试 DefaultReadWeightCalculator 的基本功能，包括输出形状、权重范围和归一化。
        """
        batch_size = 2
        num_reads = 3
        num_writes = 2
        memory_size = 4

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes,
            epsilon=1e-6
        )

        # 创建 DefaultReadWeightCalculator 实例
        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        # 创建随机的输入张量
        read_content_weights = tf.random.uniform([batch_size, num_reads, memory_size], minval=0.0, maxval=1.0)
        prev_read_weights = tf.random.uniform([batch_size, num_reads, memory_size], minval=0.0, maxval=1.0)
        link = tf.random.uniform([batch_size, num_writes, memory_size, memory_size], minval=0.0, maxval=1.0)
        read_mode = tf.random.uniform([batch_size, num_reads, 1 + 2 * num_writes], minval=0.0, maxval=1.0)

        # 规范化 read_mode 使其和为1.0
        read_mode_sum = tf.reduce_sum(read_mode, axis=-1, keepdims=True) + 1e-8  # 防止除零
        read_mode_normalized = read_mode / read_mode_sum

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode_normalized,
            training=True
        )

        # 检查输出形状
        expected_shape = [batch_size, num_reads, memory_size]
        self.assertAllEqual(read_weights.shape, expected_shape)

        # 检查权重范围
        self.assertGreaterEqual(tf.reduce_min(read_weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(read_weights).numpy(), 1.0)

        # 检查权重在 memory_size 维度上的和为1
        sum_weights = tf.reduce_sum(read_weights, axis=-1)  # [batch_size, num_reads]
        self.assertAllClose(sum_weights.numpy(), np.ones_like(sum_weights.numpy()), atol=1e-6)

    def test_compute_read_weights_specific_modes(self):
        """
        测试 DefaultReadWeightCalculator 在特定 read_mode 下的行为。
        """
        batch_size = 1
        num_reads = 1
        num_writes = 1
        memory_size = 2

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes,
            epsilon=1e-6
        )

        # 创建 DefaultReadWeightCalculator 实例
        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        # 设置具体的 link 和 prev_read_weights
        link = tf.constant([[[[0.0, 1.0],
                             [0.0, 0.0]]]], dtype=tf.float32)  # [1, 1, 2, 2]
        prev_read_weights = tf.constant([[[1.0, 0.0]]], dtype=tf.float32)  # [1, 1, 2]
        read_content_weights = tf.constant([[[0.5, 0.5]]], dtype=tf.float32)  # [1, 1, 2]

        # 设置 read_mode: content_mode=1, forward_mode=0, backward_mode=0
        # 以确保 read_weights = content_mode * read_content_weights = [0.5, 0.5]
        read_mode = tf.constant([[[1.0, 0.0, 0.0]]], dtype=tf.float32)  # [1,1,3]

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,
            training=False
        )

        # 预期读取权重
        expected_read_weights = np.array([[[0.5, 0.5]]], dtype=np.float32)
        self.assertAllClose(read_weights.numpy(), expected_read_weights, atol=1e-6)

    def test_write_weights_gradients(self):
        """
        测试是否可以计算相对于输入的梯度。
        """
        batch_size = 2
        num_reads = 2
        num_writes = 2
        memory_size = 4

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes,
            epsilon=1e-6
        )

        # 创建 DefaultReadWeightCalculator 实例
        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        # 创建输入的变量
        read_content_weights = tf.Variable(
            tf.random.uniform([batch_size, num_reads, memory_size], minval=0.5, maxval=1.5)
        )
        prev_read_weights = tf.Variable(
            tf.random.uniform([batch_size, num_reads, memory_size], minval=0.0, maxval=1.0)
        )
        link = tf.Variable(
            tf.random.uniform([batch_size, num_writes, memory_size, memory_size], minval=0.0, maxval=1.0)
        )
        read_mode = tf.Variable(
            tf.random.uniform([batch_size, num_reads, 1 + 2 * num_writes], minval=0.0, maxval=1.0)
        )

        # 规范化 read_mode 使其和为1.0
        read_mode_sum = tf.reduce_sum(read_mode, axis=-1, keepdims=True) + 1e-8  # 防止除零
        read_mode_normalized = read_mode / read_mode_sum

        with tf.GradientTape() as tape:
            read_weights = read_weight_calculator.compute(
                read_content_weights=read_content_weights,
                prev_read_weights=prev_read_weights,
                link=link,
                read_mode=read_mode_normalized,
                training=True
            )
            # 使用更敏感的损失函数
            loss = tf.reduce_sum(read_weights * read_content_weights)

        gradients = tape.gradient(loss, [read_content_weights, prev_read_weights, link, read_mode])

        # 确保梯度不为 None 并具有合理的值
        for grad in gradients:
            self.assertIsNotNone(grad)
            grad_norm = tf.norm(grad).numpy()
            self.assertGreater(grad_norm, 1e-6)  # 可根据需要调整阈值

    def test_write_weights_normalization(self):
        """
        测试 write_weights 是否按预期进行归一化。
        """
        batch_size = 3
        num_reads = 2
        num_writes = 2
        memory_size = 4

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes,
            epsilon=1e-6
        )

        # 创建 DefaultReadWeightCalculator 实例
        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        # 创建测试输入
        read_content_weights = tf.random.uniform(
            [batch_size, num_reads, memory_size], minval=0.0, maxval=1.0
        )
        # 设置 read_mode 为 [1.0,0.0,0.0]，即只使用 content_mode
        read_mode = tf.ones([batch_size, num_reads, 1 + 2 * num_writes], dtype=tf.float32)
        read_mode = tf.concat([
            tf.ones([batch_size, num_reads, 1], dtype=tf.float32),
            tf.zeros([batch_size, num_reads, 2 * num_writes], dtype=tf.float32)
        ], axis=-1)  # [batch_size, num_reads, 5]

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            prev_read_weights=tf.zeros([batch_size, num_reads, memory_size], dtype=tf.float32),
            link=tf.zeros([batch_size, num_writes, memory_size, memory_size], dtype=tf.float32),
            read_mode=read_mode,
            training=False
        )

        # 因为 allocation_gate 和 write_gate 都为0，所以 read_weights 应等于 content_mode * read_content_weights
        # content_mode =1.0, read_weights = read_content_weights
        # 需要确保 read_weights 的每个读头在 memory_size 维度上和为1
        # 但由于 read_content_weights 是随机生成的且未归一化，因此调整测试
        # 改为在 read_content_weights 上应用 softmax，使其和为1

        read_content_weights_softmax = tf.nn.softmax(read_content_weights, axis=-1)
        read_weights_softmax = read_weight_calculator.compute(
            read_content_weights=read_content_weights_softmax,
            prev_read_weights=tf.zeros([batch_size, num_reads, memory_size], dtype=tf.float32),
            link=tf.zeros([batch_size, num_writes, memory_size, memory_size], dtype=tf.float32),
            read_mode=read_mode,
            training=False
        )

        # 检查 write_weights 和 read_content_weights_softmax 是否相等
        self.assertAllClose(read_weights_softmax.numpy(), read_content_weights_softmax.numpy(), atol=1e-6)

        # 检查权重在 memory_size 维度上的和是否为1
        sum_weights = tf.reduce_sum(read_weights_softmax, axis=-1)  # [batch_size, num_reads]
        self.assertAllClose(sum_weights.numpy(), np.ones_like(sum_weights.numpy()), atol=1e-6)

    def test_compute_read_weights_specific_modes_forward(self):
        """
        测试 DefaultReadWeightCalculator 在特定 read_mode 下的行为（前向模式）。
        """
        batch_size = 1
        num_reads = 1
        num_writes = 1
        memory_size = 2

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes,
            epsilon=1e-6
        )

        # 创建 DefaultReadWeightCalculator 实例
        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        # 设置具体的 link 和 prev_read_weights
        link = tf.constant([[[[0.0, 1.0],
                             [0.0, 0.0]]]], dtype=tf.float32)  # [1, 1, 2, 2]
        prev_read_weights = tf.constant([[[1.0, 0.0]]], dtype=tf.float32)  # [1, 1, 2]
        read_content_weights = tf.constant([[[0.5, 0.5]]], dtype=tf.float32)  # [1, 1, 2]

        # 设置 read_mode: content_mode=1, forward_mode=0, backward_mode=0
        read_mode = tf.constant([[[1.0, 0.0, 0.0]]], dtype=tf.float32)  # [1,1,3]

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,
            training=False
        )

        # 预期读取权重 = content_mode * read_content_weights = [0.5, 0.5]
        expected_read_weights = np.array([[[0.5, 0.5]]], dtype=np.float32)
        self.assertAllClose(read_weights.numpy(), expected_read_weights, atol=1e-6)

    def test_compute_read_weights_specific_modes_forward_nonzero(self):
        """
        测试 DefaultReadWeightCalculator 在特定 read_mode 下的行为（前向模式激活）。
        """
        batch_size = 1
        num_reads = 1
        num_writes = 1
        memory_size = 2

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes,
            epsilon=1e-6
        )

        # 创建 DefaultReadWeightCalculator 实例
        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        # 设置具体的 link 和 prev_read_weights
        link = tf.constant([[[[0.0, 1.0],
                             [0.0, 0.0]]]], dtype=tf.float32)  # [1, 1, 2, 2]
        prev_read_weights = tf.constant([[[1.0, 0.0]]], dtype=tf.float32)  # [1, 1, 2]
        read_content_weights = tf.constant([[[0.5, 0.5]]], dtype=tf.float32)  # [1, 1, 2]

        # 设置 read_mode: content_mode=0.5, forward_mode=0.5, backward_mode=0.0
        read_mode = tf.constant([[[0.5, 0.5, 0.0]]], dtype=tf.float32)  # [1,1,3]

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,
            training=False
        )

        # 根据 link 和 prev_read_weights:
        # forward_weights = tf.einsum('bwmn,brm->brwn', link, prev_read_weights) = [[[0.0,1.0]]]
        # forward_component = sum(forward_mode * forward_weights, axis=2) = 0.5 * [0.0,1.0] = [0.0,0.5]
        # read_weights = 0.5 * [0.5,0.5] + [0.0,0.5] = [0.25,0.75]
        expected_read_weights = np.array([[[0.25, 0.75]]], dtype=np.float32)
        self.assertAllClose(read_weights.numpy(), expected_read_weights, atol=1e-6)

    def test_compute_read_weights_specific_modes_backward(self):
        """
        测试 DefaultReadWeightCalculator 在特定 read_mode 下的行为（后向模式激活）。
        """
        batch_size = 1
        num_reads = 1
        num_writes = 1
        memory_size = 2

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes,
            epsilon=1e-6
        )

        # 创建 DefaultReadWeightCalculator 实例
        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        # 设置具体的 link 和 prev_read_weights
        link = tf.constant([[[[0.0, 0.0],
                             [1.0, 0.0]]]], dtype=tf.float32)  # [1, 1, 2, 2]
        prev_read_weights = tf.constant([[[0.0, 1.0]]], dtype=tf.float32)  # [1, 1, 2]
        read_content_weights = tf.constant([[[0.5, 0.5]]], dtype=tf.float32)  # [1, 1, 2]

        # 设置 read_mode: content_mode=0.5, forward_mode=0.0, backward_mode=0.5
        read_mode = tf.constant([[[0.5, 0.0, 0.5]]], dtype=tf.float32)  # [1,1,3]

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,
            training=False
        )

        # 根据 link 和 prev_read_weights:
        # backward_weights = tf.einsum('bwmn,brm->brwn', tf.transpose(link, perm=[0,1,3,2]), prev_read_weights)
        # = [[[0.0,1.0]]]
        # backward_component = sum(backward_mode * backward_weights, axis=2) = 0.5 * [0.0,1.0] = [0.0,0.5]
        # read_weights = 0.5 * [0.5,0.5] + [0.0,0.5] = [0.25,0.75]
        expected_read_weights = np.array([[[0.25, 0.75]]], dtype=np.float32)
        self.assertAllClose(read_weights.numpy(), expected_read_weights, atol=1e-6)

if __name__ == '__main__':
    tf.test.main()