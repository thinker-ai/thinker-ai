# test_default_write_weight_calculator.py

import tensorflow as tf
import numpy as np

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import DefaultWriteWeightCalculator


class DefaultWriteWeightCalculatorTest(tf.test.TestCase):
    def setUp(self):
        super(DefaultWriteWeightCalculatorTest, self).setUp()
        self.memory_size = 4
        self.num_writes = 2
        self.epsilon = 1e-6

        # Instantiate DefaultWriteWeightCalculator
        self.write_weight_calculator = DefaultWriteWeightCalculator(
            memory_size=self.memory_size,
            num_writes=self.num_writes
        )

    def test_compute_shape(self):
        """
        测试 compute 方法返回的 write_weights 是否具有正确的形状。
        """
        batch_size = 3

        # 创建随机测试输入
        write_content_weights = tf.random.uniform(
            [batch_size, self.num_writes, self.memory_size], minval=0.0, maxval=1.0
        )
        allocation_gate = tf.random.uniform(
            [batch_size, self.num_writes], minval=0.0, maxval=1.0
        )
        write_gate = tf.random.uniform(
            [batch_size, self.num_writes], minval=0.0, maxval=1.0
        )
        prev_usage = tf.random.uniform(
            [batch_size, self.memory_size], minval=0.0, maxval=1.0 - self.epsilon
        )

        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        expected_shape = [batch_size, self.num_writes, self.memory_size]
        self.assertAllEqual(write_weights.shape, expected_shape)

    def test_write_weights_range(self):
        """
        测试 write_weights 的值是否在 [0, 1] 范围内。
        """
        batch_size = 3

        # 创建测试输入
        write_content_weights = tf.random.uniform(
            [batch_size, self.num_writes, self.memory_size], minval=0.0, maxval=1.0
        )
        allocation_gate = tf.random.uniform(
            [batch_size, self.num_writes], minval=0.0, maxval=1.0
        )
        write_gate = tf.random.uniform(
            [batch_size, self.num_writes], minval=0.0, maxval=1.0
        )
        prev_usage = tf.random.uniform(
            [batch_size, self.memory_size], minval=0.0, maxval=1.0 - self.epsilon
        )

        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 检查 write_weights 的值是否在 [0, 1] 范围内
        self.assertGreaterEqual(tf.reduce_min(write_weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(write_weights).numpy(), 1.0)

    def test_write_weights_sum(self):
        """
        测试 write_weights 在 memory_size 维度上的和是否接近于 write_gate。
        """
        batch_size = 3

        # 创建测试输入，确保 write_content_weights 的值不为零，并进行归一化
        write_content_weights = tf.random.uniform(
            [batch_size, self.num_writes, self.memory_size], minval=0.1, maxval=1.0
        )
        write_content_weights_sum = tf.reduce_sum(write_content_weights, axis=-1, keepdims=True) + self.epsilon
        write_content_weights_normalized = write_content_weights / write_content_weights_sum

        allocation_gate = tf.random.uniform(
            [batch_size, self.num_writes], minval=0.0, maxval=1.0 - self.epsilon
        )
        write_gate = tf.random.uniform(
            [batch_size, self.num_writes], minval=0.0, maxval=1.0 - self.epsilon
        )
        prev_usage = tf.random.uniform(
            [batch_size, self.memory_size], minval=0.0, maxval=0.5
        )

        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights_normalized,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 计算 write_weights 在 memory_size 维度上的和
        write_weights_sum = tf.reduce_sum(write_weights, axis=-1)  # [batch_size, num_writes]

        # 检查和是否接近于 write_gate
        self.assertAllClose(write_weights_sum.numpy(), write_gate.numpy(), atol=1e-5)

    def test_allocation_weights_correctness(self):
        """
        测试 allocation_weights 的计算是否正确。
        """
        batch_size = 1
        memory_size = 4

        # 设置特定的 prev_usage，便于手动计算
        prev_usage = tf.constant([[0.2, 0.1, 0.4, 0.3]], dtype=tf.float32)  # [1, 4]

        # 手动计算 allocation_weights
        usage = tf.clip_by_value(prev_usage, 0.0, 1.0 - self.epsilon)
        sorted_usage, sorted_indices = tf.nn.top_k(-usage, k=memory_size)
        sorted_usage = -sorted_usage  # [1, 4]
        sorted_nonusage = 1 - sorted_usage  # [1, 4]

        cumprod = tf.math.cumprod(sorted_nonusage + self.epsilon, axis=1, exclusive=True)  # [1, 4]
        allocation_weights_sorted = sorted_nonusage * cumprod  # [1, 4]

        # 将 allocation_weights_sorted 映射回原来的顺序
        inverted_indices = tf.argsort(sorted_indices, axis=1)
        allocation_weights = tf.gather(allocation_weights_sorted, inverted_indices, batch_dims=1)

        # 对 allocation_weights 进行归一化
        allocation_weights_sum = tf.reduce_sum(allocation_weights, axis=1, keepdims=True) + self.epsilon
        allocation_weights_normalized = allocation_weights / allocation_weights_sum  # [1, 4]

        # 由于 num_writes = 2，扩展维度
        allocation_weights_normalized = tf.expand_dims(allocation_weights_normalized, axis=1)  # [1, 1, 4]
        allocation_weights_normalized = tf.tile(allocation_weights_normalized, [1, self.num_writes, 1])  # [1, 2, 4]

        # 通过 compute_allocation_weights 计算
        calculated_allocation_weights = self.write_weight_calculator.compute_allocation_weights(prev_usage)

        # 比较手动计算和函数计算的结果
        self.assertAllClose(calculated_allocation_weights.numpy(), allocation_weights_normalized.numpy(), atol=1e-6)

    def test_compute_with_known_values(self):
        """
        测试 compute 方法使用已知输入时，输出是否符合预期。
        """
        batch_size = 1
        num_writes = self.num_writes
        memory_size = self.memory_size

        # 定义已知输入
        write_content_weights = tf.constant([[[0.1, 0.2, 0.3, 0.4],
                                              [0.4, 0.3, 0.2, 0.1]]], dtype=tf.float32)  # [1, 2, 4]
        allocation_gate = tf.constant([[0.5, 0.5]], dtype=tf.float32)  # [1, 2]
        write_gate = tf.constant([[0.8, 0.6]], dtype=tf.float32)  # [1, 2]
        prev_usage = tf.constant([[0.0, 0.5, 0.5, 0.0]], dtype=tf.float32)  # [1, 4]

        # 手动计算 allocation_weights
        allocation_weights = self.write_weight_calculator.compute_allocation_weights(prev_usage).numpy()  # [1, 2, 4]

        # 归一化 write_content_weights
        wcw = write_content_weights.numpy()
        wcw_sum = np.sum(wcw, axis=-1, keepdims=True) + self.epsilon
        write_content_weights_normalized = wcw / wcw_sum  # [1, 2, 4]

        # 计算 expected write_weights
        ag = allocation_gate.numpy()[:, :, np.newaxis]  # [1, 2, 1]
        wg = write_gate.numpy()[:, :, np.newaxis]  # [1, 2, 1]

        expected_write_weights = wg * (
                ag * allocation_weights +
                (1 - ag) * write_content_weights_normalized
        )  # [1, 2, 4]

        # 使用 compute 方法计算 write_weights
        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        ).numpy()

        # 比较 expected_write_weights 和 write_weights
        self.assertAllClose(write_weights, expected_write_weights, atol=1e-6)

    def test_allocation_weights_sum(self):
        """
        测试 allocation_weights 在 memory_size 维度上的和是否为 1。
        """
        batch_size = 3

        # 创建随机 prev_usage
        prev_usage = tf.random.uniform(
            [batch_size, self.memory_size], minval=0.0, maxval=1.0 - self.epsilon
        )

        allocation_weights = self.write_weight_calculator.compute_allocation_weights(prev_usage)

        # 计算 allocation_weights 在 memory_size 维度上的和
        allocation_weights_sum = tf.reduce_sum(allocation_weights, axis=-1)  # [batch_size, num_writes]

        # 检查和是否为 1
        expected_sum = np.ones((batch_size, self.num_writes))
        self.assertAllClose(allocation_weights_sum.numpy(), expected_sum, atol=1e-6)

    def test_allocation_weights_with_full_usage(self):
        """
        测试 prev_usage 为全 1 时，allocation_weights 是否为零。
        """
        batch_size = 2
        prev_usage = tf.ones([batch_size, self.memory_size], dtype=tf.float32) * (1.0 - self.epsilon)

        allocation_weights = self.write_weight_calculator.compute_allocation_weights(prev_usage)

        # allocation_weights 应该为零
        self.assertAllClose(allocation_weights.numpy(), np.zeros_like(allocation_weights.numpy()), atol=1e-6)

    def test_gradients(self):
        """
        测试 compute 方法是否可以正确地计算梯度。
        """
        batch_size = 2

        # 创建可训练变量
        write_content_weights = tf.Variable(
            tf.random.uniform([batch_size, self.num_writes, self.memory_size], minval=0.1, maxval=1.0)
        )
        allocation_gate = tf.Variable(
            tf.random.uniform([batch_size, self.num_writes], minval=0.0, maxval=1.0)
        )
        write_gate = tf.Variable(
            tf.random.uniform([batch_size, self.num_writes], minval=0.0, maxval=1.0)
        )
        prev_usage = tf.Variable(
            tf.random.uniform([batch_size, self.memory_size], minval=0.0, maxval=1.0 - self.epsilon)
        )

        with tf.GradientTape() as tape:
            write_weights = self.write_weight_calculator.compute(
                write_content_weights=write_content_weights,
                allocation_gate=allocation_gate,
                write_gate=write_gate,
                prev_usage=prev_usage,
                training=True
            )
            loss = tf.reduce_sum(write_weights)

        gradients = tape.gradient(loss, [write_content_weights, allocation_gate, write_gate, prev_usage])

        # 确保梯度不是 None
        for grad in gradients:
            self.assertIsNotNone(grad)

    def test_compute_allocation_weights_zero_usage(self):
        """
        测试 prev_usage 为零时，allocation_weights 的和是否为 1。
        """
        batch_size = 2
        prev_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)

        allocation_weights = self.write_weight_calculator.compute_allocation_weights(prev_usage)

        # 计算 allocation_weights_sum
        allocation_weights_sum = tf.reduce_sum(allocation_weights[:, 0, :], axis=-1)  # [batch_size]

        # 检查 allocation_weights_sum 是否等于 1
        expected_sum = np.ones(batch_size)
        self.assertAllClose(allocation_weights_sum.numpy(), expected_sum, atol=1e-6)

    def test_compute_allocation_weights_with_ties(self):
        """
        测试 prev_usage 存在相同值（ties）时，allocation_weights 的计算是否合理。
        """
        batch_size = 1
        prev_usage = tf.constant([[0.5, 0.5, 0.5, 0.5]], dtype=tf.float32)  # [1, 4]

        allocation_weights = self.write_weight_calculator.compute_allocation_weights(prev_usage)

        # 计算 allocation_weights 在 memory_size 维度上的和
        allocation_weights_sum = tf.reduce_sum(allocation_weights, axis=-1)  # [batch_size, num_writes]

        # 检查和是否为 1
        expected_sum = np.ones((batch_size, self.num_writes))
        self.assertAllClose(allocation_weights_sum.numpy(), expected_sum, atol=1e-6)

    def test_invalid_memory_size(self):
        """
        测试 memory_size 非正数时是否抛出异常。
        """
        with self.assertRaises(ValueError):
            DefaultWriteWeightCalculator(memory_size=0, num_writes=1)

    def test_invalid_num_writes(self):
        """
        测试 num_writes 非正数时是否抛出异常。
        """
        with self.assertRaises(ValueError):
            DefaultWriteWeightCalculator(memory_size=4, num_writes=0)

    def test_prev_usage_out_of_bounds(self):
        """
        测试 prev_usage 超出 [0, 1] 范围时，是否被正确处理。
        """
        batch_size = 1
        prev_usage = tf.constant([[-0.1, 1.1, 0.5, 0.5]], dtype=tf.float32)

        # 期望的 prev_usage 被裁剪到 [0, 1 - epsilon] 范围内
        expected_prev_usage = tf.clip_by_value(prev_usage, 0.0, 1.0 - self.epsilon)

        # 计算 allocation_weights
        allocation_weights = self.write_weight_calculator.compute_allocation_weights(prev_usage)
        allocation_weights_expected = self.write_weight_calculator.compute_allocation_weights(expected_prev_usage)

        # 验证 allocation_weights 是否与使用裁剪后的 prev_usage 计算的 allocation_weights_expected 一致
        self.assertAllClose(allocation_weights.numpy(), allocation_weights_expected.numpy(), atol=1e-6)

    def test_write_content_weights_normalization(self):
        """
        测试 write_content_weights 的归一化是否正确。
        """
        batch_size = 1
        write_content_weights = tf.constant([[[0.0, 0.0, 0.0, 0.0],
                                              [0.0, 0.0, 0.0, 0.0]]], dtype=tf.float32)

        # 手动计算归一化后的 write_content_weights
        expected_normalized = np.full((batch_size, self.num_writes, self.memory_size),
                                      1.0 / self.memory_size)

        # 调用 compute 方法
        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=tf.ones([batch_size, self.num_writes], dtype=tf.float32),
            write_gate=tf.ones([batch_size, self.num_writes], dtype=tf.float32),
            prev_usage=tf.zeros([batch_size, self.memory_size], dtype=tf.float32),
            training=False
        )

        # 手动计算 expected write_weights
        allocation_weights = self.write_weight_calculator.compute_allocation_weights(
            tf.zeros([batch_size, self.memory_size], dtype=tf.float32)
        ).numpy()

        allocation_gate = np.ones([batch_size, self.num_writes, 1])
        write_gate = np.ones([batch_size, self.num_writes, 1])

        expected_write_weights = write_gate * (
                allocation_gate * allocation_weights +
                (1 - allocation_gate) * expected_normalized
        )

        # 验证 write_weights 是否与期望值一致
        self.assertAllClose(write_weights.numpy(), expected_write_weights, atol=1e-6)

    def test_write_weights_when_write_gate_zero(self):
        """
        测试 write_gate 为零时，write_weights 是否为零。
        """
        batch_size = 1

        write_weights = self.write_weight_calculator.compute(
            write_content_weights=tf.random.uniform([batch_size, self.num_writes, self.memory_size]),
            allocation_gate=tf.random.uniform([batch_size, self.num_writes]),
            write_gate=tf.zeros([batch_size, self.num_writes]),
            prev_usage=tf.random.uniform([batch_size, self.memory_size]),
            training=False
        )

        # 验证 write_weights 是否为零
        self.assertAllClose(write_weights.numpy(), np.zeros_like(write_weights.numpy()), atol=1e-6)

    def test_large_memory_size(self):
        """
        测试在较大的 memory_size 下，compute 方法是否正常工作。
        """
        batch_size = 2
        memory_size = 1024
        num_writes = 4

        write_weight_calculator = DefaultWriteWeightCalculator(
            memory_size=memory_size,
            num_writes=num_writes
        )

        write_weights = write_weight_calculator.compute(
            write_content_weights=tf.random.uniform([batch_size, num_writes, memory_size]),
            allocation_gate=tf.random.uniform([batch_size, num_writes]),
            write_gate=tf.random.uniform([batch_size, num_writes]),
            prev_usage=tf.random.uniform([batch_size, memory_size], maxval=1.0 - self.epsilon),
            training=False
        )

        # 验证输出的形状
        self.assertEqual(write_weights.shape, (batch_size, num_writes, memory_size))

    def test_allocation_gate_out_of_range(self):
        """
        测试 allocation_gate 超出 [0, 1] 范围的情况。
        """
        batch_size = 2
        memory_size = 4
        num_writes = 2

        write_weight_calculator = DefaultWriteWeightCalculator(
            memory_size=memory_size,
            num_writes=num_writes
        )

        write_content_weights = tf.random.uniform(
            [batch_size, num_writes, memory_size], minval=0.0, maxval=1.0
        )
        allocation_gate = tf.constant([
            [1.2, -0.1],
            [0.5, 1.5]
        ], dtype=tf.float32)  # 超出范围
        write_gate = tf.random.uniform(
            [batch_size, num_writes], minval=0.0, maxval=1.0
        )
        prev_usage = tf.random.uniform(
            [batch_size, memory_size], minval=0.0, maxval=1.0 - self.epsilon
        )

        # 运行 compute 方法，观察是否处理超出范围的 gate
        write_weights = write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 检查 write_weights 的范围是否合理
        self.assertGreaterEqual(tf.reduce_min(write_weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(write_weights).numpy(), 1.0)

    def test_prev_usage_out_of_range(self):
        """
        测试 prev_usage 超出 [0, 1] 范围的情况。
        """
        batch_size = 2
        memory_size = 4
        num_writes = 2

        write_weight_calculator = DefaultWriteWeightCalculator(
            memory_size=memory_size,
            num_writes=num_writes
        )

        write_content_weights = tf.random.uniform(
            [batch_size, num_writes, memory_size], minval=0.0, maxval=1.0
        )
        allocation_gate = tf.random.uniform(
            [batch_size, num_writes], minval=0.0, maxval=1.0
        )
        write_gate = tf.random.uniform(
            [batch_size, num_writes], minval=0.0, maxval=1.0
        )
        prev_usage = tf.constant([
            [1.2, -0.1, 0.5, 0.3],
            [0.7, 1.5, 0.2, -0.2]
        ], dtype=tf.float32)  # 超出范围

        # 运行 compute 方法，观察是否处理超出范围的 prev_usage
        write_weights = write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 检查 write_weights 的范围是否合理
        self.assertGreaterEqual(tf.reduce_min(write_weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(write_weights).numpy(), 1.0)


if __name__ == '__main__':
    tf.test.main()
