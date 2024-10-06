import numpy as np
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import DefaultReadWeightCalculator, \
    DefaultTemporalLinkageUpdater

tf.random.set_seed(42)
np.random.seed(42)

class DefaultReadWeightCalculatorTest(tf.test.TestCase):
    def test_compute_read_weights_basic(self):
        batch_size = 2
        num_reads = 3
        num_writes = 2
        memory_size = 4

        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes
        )

        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        read_content_weights = tf.random.uniform([batch_size, num_reads, memory_size], minval=0.0, maxval=1.0)
        prev_read_weights = tf.nn.softmax(tf.random.uniform([batch_size, num_reads, memory_size]), axis=-1)
        link = tf.random.uniform([batch_size, num_writes, memory_size, memory_size], minval=0.0, maxval=1.0)
        read_mode = tf.random.uniform([batch_size, num_reads, 3], minval=0.0, maxval=1.0)

        # 对 read_content_weights 在 memory_size 维度上进行归一化
        read_content_weights_normalized = tf.nn.softmax(read_content_weights, axis=-1)

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights_normalized,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,
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

    def test_compute_read_weights_mixed_modes(self):
        """
        测试 DefaultReadWeightCalculator 在混合 read_mode 下的行为。
        """
        batch_size = 1
        num_reads = 1
        num_writes = 1
        memory_size = 2

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes
        )

        # 创建 DefaultReadWeightCalculator 实例
        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        # 设置具体的 link 和 prev_read_weights
        link = tf.constant([[[[0.0, 1.0],
                              [1.0, 0.0]]]], dtype=tf.float32)  # [1, 1, 2, 2]
        prev_read_weights = tf.constant([[[0.6, 0.4]]], dtype=tf.float32)  # [1, 1, 2]
        read_content_weights = tf.constant([[[1.0, 2.0]]], dtype=tf.float32)  # [1, 1, 2]

        # 对 read_content_weights 进行归一化
        read_content_weights_normalized = tf.nn.softmax(read_content_weights, axis=-1)

        # 设置 read_mode，未归一化，让 compute 方法处理
        read_mode = tf.constant([[[0.5, 0.3, 0.2]]], dtype=tf.float32)  # [1,1,3]

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights_normalized,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,  # 在 compute 方法中进行 softmax 归一化
            training=False
        )

        # 手动计算归一化后的 read_mode
        read_mode_normalized = tf.nn.softmax(read_mode, axis=-1).numpy()[0, 0]

        # 计算各模式的权重
        content_mode_weight = read_mode_normalized[0]
        forward_mode_weight = read_mode_normalized[1]
        backward_mode_weight = read_mode_normalized[2]

        # 计算内容权重
        content_weights = content_mode_weight * read_content_weights_normalized.numpy()[0, 0, :]

        # 计算前向权重
        forward_weights = forward_mode_weight * np.matmul(
            prev_read_weights.numpy()[0, 0, :], link.numpy()[0, 0, :, :]
        )

        # 计算后向权重
        backward_weights = backward_mode_weight * np.matmul(
            prev_read_weights.numpy()[0, 0, :], link.numpy()[0, 0, :, :].T
        )

        # 组合并归一化
        expected_read_weights = content_weights + forward_weights + backward_weights
        expected_read_weights = expected_read_weights / (np.sum(expected_read_weights) + 1e-8)
        expected_read_weights = expected_read_weights.reshape(1, 1, -1)

        self.assertAllClose(
            read_weights.numpy(),
            expected_read_weights,
            atol=1e-6)

    def test_compute_read_weights_specific_modes(self):
        """
        测试 DefaultReadWeightCalculator 在特定 read_mode 下的行为（仅使用内容模式）。
        """
        batch_size = 1
        num_reads = 1
        num_writes = 1
        memory_size = 2

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes
        )

        # 创建 DefaultReadWeightCalculator 实例
        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        # 设置具体的 link 和 prev_read_weights
        link = tf.zeros([batch_size, num_writes, memory_size, memory_size], dtype=tf.float32)
        prev_read_weights = tf.zeros([batch_size, num_reads, memory_size], dtype=tf.float32)
        read_content_weights = tf.constant([[[2.0, 1.0]]], dtype=tf.float32)  # [1, 1, 2]

        # 对 read_content_weights 进行归一化
        read_content_weights_normalized = tf.nn.softmax(read_content_weights, axis=-1)

        # 设置 read_mode，使内容模式权重接近 1
        read_mode = tf.constant([[[10.0, -10.0, -10.0]]], dtype=tf.float32)  # [1,1,3]

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights_normalized,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,  # 在 compute 方法中进行 softmax 归一化
            training=False
        )

        # 预期读取权重应该等于归一化的 read_content_weights
        expected_read_weights = read_content_weights_normalized.numpy()
        self.assertAllClose(read_weights.numpy(), expected_read_weights, atol=1e-6)

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
            num_writes=num_writes
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
        read_content_weights = tf.zeros([batch_size, num_reads, memory_size], dtype=tf.float32)

        # 对 read_content_weights 进行归一化
        read_content_weights_normalized = tf.constant([[[0.5, 0.5]]], dtype=tf.float32)

        # 设置 read_mode，使前向模式权重接近 1
        read_mode = tf.constant([[[-10.0, 10.0, -10.0]]], dtype=tf.float32)  # [1,1,3]

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights_normalized,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,  # 在 compute 方法中进行 softmax 归一化
            training=False
        )

        # 预期读取权重应该等于前向权重
        expected_read_weights = np.array([[[0.0, 1.0]]], dtype=np.float32)
        self.assertAllClose(read_weights.numpy(), expected_read_weights, atol=1e-6)

    def test_compute_read_weights_specific_modes_backward(self):
        """
        测试 DefaultReadWeightCalculator 在特定 read_mode 下的行为（后向模式）。
        """
        batch_size = 1
        num_reads = 1
        num_writes = 1
        memory_size = 2

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes
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
        prev_read_weights = tf.constant([[[0.0, 1.0]]], dtype=tf.float32)  # [1, 1, 2]
        read_content_weights = tf.zeros([batch_size, num_reads, memory_size], dtype=tf.float32)

        # 对 read_content_weights 进行归一化
        read_content_weights_normalized = tf.constant([[[0.5, 0.5]]], dtype=tf.float32)

        # 设置 read_mode，使后向模式权重接近 1
        read_mode = tf.constant([[[-10.0, -10.0, 10.0]]], dtype=tf.float32)  # [1,1,3]

        # 计算读取权重
        read_weights = read_weight_calculator.compute(
            read_content_weights=read_content_weights_normalized,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,
            training=False
        )

        # 预期读取权重应该等于后向权重
        expected_read_weights = np.array([[[1.0, 0.0]]], dtype=np.float32)
        self.assertAllClose(read_weights.numpy(), expected_read_weights, atol=1e-6)

    def test_gradients_exist(self):
        """
        测试在训练过程中是否可以正确计算梯度。
        """
        batch_size = 2
        num_reads = 2
        num_writes = 2
        memory_size = 4

        # 创建 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=memory_size,
            num_writes=num_writes
        )

        # 创建 DefaultReadWeightCalculator 实例
        read_weight_calculator = DefaultReadWeightCalculator(
            temporal_linkage=temporal_linkage,
            num_reads=num_reads,
            num_writes=num_writes
        )

        # 创建随机的输入变量
        read_content_weights = tf.Variable(
            tf.nn.softmax(tf.random.uniform([batch_size, num_reads, memory_size]), axis=-1)
        )
        prev_read_weights = tf.Variable(
            tf.nn.softmax(tf.random.uniform([batch_size, num_reads, memory_size]), axis=-1)
        )
        link = tf.Variable(
            tf.random.uniform([batch_size, num_writes, memory_size, memory_size], minval=0.0, maxval=1.0)
        )
        # 不对 link 进行归一化，以确保梯度计算
        read_mode = tf.Variable(
            tf.random.uniform([batch_size, num_reads, 3], minval=-1.0, maxval=1.0)
        )
        # 不对 read_mode 进行归一化，在 compute 方法中处理

        with tf.GradientTape() as tape:
            read_weights = read_weight_calculator.compute(
                read_content_weights=read_content_weights,
                prev_read_weights=prev_read_weights,
                link=link,
                read_mode=read_mode,
                training=True
            )
            # 使用与输入相关的损失函数，确保梯度非零
            loss = tf.reduce_sum(read_weights * read_content_weights)

        gradients = tape.gradient(
            loss, [read_content_weights, prev_read_weights, link, read_mode])

        # 确保梯度不是 None，并且具有合理的值
        for idx, grad in enumerate(gradients):
            self.assertIsNotNone(
                grad, f"Gradient for variable index {idx} is None.")
            grad_norm = tf.norm(grad).numpy()
            self.assertGreater(
                grad_norm, 1e-6,
                f"Gradient norm for variable index {idx} is too small: {grad_norm}")


if __name__ == '__main__':
    tf.test.main()
