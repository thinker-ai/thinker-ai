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
            num_writes=self.num_writes,
            epsilon=self.epsilon
        )

    def test_compute_shape(self):
        """
        Test that the compute method returns write_weights of the correct shape.
        """
        batch_size = 3

        # Create random test inputs
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
            [batch_size, self.memory_size], minval=0.0, maxval=1.0
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
        Test that the write_weights values are within the range [0, 1].
        """
        batch_size = 3

        # Create test inputs
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
            [batch_size, self.memory_size], minval=0.0, maxval=1.0
        )

        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # Check the range of write_weights
        self.assertGreaterEqual(tf.reduce_min(write_weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(write_weights).numpy(), 1.0)

    def test_write_weights_computation(self):
        """
        测试 write_weights 的计算。
        """
        batch_size = 1

        # 设置特定测试输入，调整 num_writes 为 2
        write_content_weights = tf.constant([
            [[0.1, 0.2, 0.3, 0.4],
             [0.5, 0.6, 0.7, 0.8]]
        ], dtype=tf.float32)  # [1, 2, 4]
        allocation_gate = tf.constant([[0.5, 0.5]], dtype=tf.float32)  # [1, 2]
        write_gate = tf.constant([[0.8, 0.8]], dtype=tf.float32)  # [1, 2]
        prev_usage = tf.constant([[0.4, 0.1, 0.2, 0.3]], dtype=tf.float32)  # [1, 4]

        # 手动计算 expected_allocation_weights
        allocation_scores = -prev_usage  # [1, 4]
        allocation_weights = tf.nn.softmax(allocation_scores, axis=-1).numpy()  # [1, 4]
        expected_allocation_weights = allocation_weights.reshape(batch_size, 1, self.memory_size)  # [1, 1, 4]
        expected_allocation_weights = np.tile(expected_allocation_weights, [1, self.num_writes, 1])  # [1, 2, 4]

        # 扩展 gates
        allocation_gate_expanded = allocation_gate[..., tf.newaxis]  # [1, 2, 1]
        write_gate_expanded = write_gate[..., tf.newaxis]  # [1, 2, 1]

        # 计算 expected_write_weights
        expected_write_weights = write_gate_expanded * (
                allocation_gate_expanded * expected_allocation_weights +
                (1 - allocation_gate_expanded) * write_content_weights
        )  # [1, 2, 4]

        # 计算实际的 write_weights
        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 比较预期和实际的 write_weights
        self.assertAllClose(write_weights.numpy(), expected_write_weights, atol=1e-6)

    def test_write_weights_with_zero_prev_usage(self):
        """
        当 prev_usage 为零时，测试 write_weights 的计算。
        """
        batch_size = 1

        # 设置特定测试输入，调整 num_writes 为 2
        write_content_weights = tf.constant([
            [[0.2, 0.4, 0.6, 0.8],
             [0.1, 0.3, 0.5, 0.7]]
        ], dtype=tf.float32)  # [1, 2, 4]
        allocation_gate = tf.constant([[1.0, 1.0]], dtype=tf.float32)  # [1, 2]
        write_gate = tf.constant([[1.0, 1.0]], dtype=tf.float32)  # [1, 2]
        prev_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [1, 4]

        # 预期的 allocation_weights 应该是均匀分布
        expected_allocation_weights = np.full((batch_size, 1, self.memory_size), 1.0 / self.memory_size,
                                              dtype=np.float32)
        expected_allocation_weights = np.tile(expected_allocation_weights, [1, self.num_writes, 1])  # [1, 2, 4]

        # 扩展 gates
        allocation_gate_expanded = allocation_gate[..., tf.newaxis]  # [1, 2, 1]
        write_gate_expanded = write_gate[..., tf.newaxis]  # [1, 2, 1]

        # 计算 expected_write_weights
        expected_write_weights = write_gate_expanded * (
                allocation_gate_expanded * expected_allocation_weights +
                (1 - allocation_gate_expanded) * write_content_weights
        )  # [1, 2, 4]

        # 计算实际的 write_weights
        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 比较预期和实际的 write_weights
        self.assertAllClose(write_weights.numpy(), expected_write_weights, atol=1e-6)

    def test_write_weights_with_full_prev_usage(self):
        """
        当 prev_usage 为全1时，测试 write_weights 的计算。
        """
        batch_size = 1

        # 设置特定测试输入，调整 num_writes 为 2
        write_content_weights = tf.constant([
            [[0.2, 0.4, 0.6, 0.8],
             [0.1, 0.3, 0.5, 0.7]]
        ], dtype=tf.float32)  # [1, 2, 4]
        allocation_gate = tf.constant([[1.0, 1.0]], dtype=tf.float32)  # [1, 2]
        write_gate = tf.constant([[1.0, 1.0]], dtype=tf.float32)  # [1, 2]
        prev_usage = tf.ones([batch_size, self.memory_size], dtype=tf.float32)  # [1, 4]

        # 预期的 allocation_weights
        allocation_scores = -prev_usage.numpy()  # [1, 4]
        allocation_weights = tf.nn.softmax(allocation_scores, axis=-1).numpy()  # [1, 4]
        expected_allocation_weights = allocation_weights.reshape(batch_size, 1, self.memory_size)  # [1, 1, 4]
        expected_allocation_weights = np.tile(expected_allocation_weights, [1, self.num_writes, 1])  # [1, 2, 4]

        # 扩展 gates
        allocation_gate_expanded = allocation_gate[..., tf.newaxis]  # [1, 2, 1]
        write_gate_expanded = write_gate[..., tf.newaxis]  # [1, 2, 1]

        # 计算 expected_write_weights
        expected_write_weights = write_gate_expanded * (
                allocation_gate_expanded * expected_allocation_weights +
                (1 - allocation_gate_expanded) * write_content_weights
        )  # [1, 2, 4]

        # 计算实际的 write_weights
        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 比较预期和实际的 write_weights
        self.assertAllClose(write_weights.numpy(), expected_write_weights, atol=1e-6)

    def test_write_weights_gradients(self):
        """
        测试是否可以计算相对于输入的梯度。
        """
        batch_size = 2

        # 创建输入的变量，使用较大的范围
        write_content_weights = tf.Variable(
            tf.random.uniform([batch_size, self.num_writes, self.memory_size], minval=0.5, maxval=1.5)
        )
        allocation_gate = tf.Variable(
            tf.random.uniform([batch_size, self.num_writes], minval=0.5, maxval=1.5)
        )
        write_gate = tf.Variable(
            tf.random.uniform([batch_size, self.num_writes], minval=0.5, maxval=1.5)
        )
        prev_usage = tf.Variable(
            tf.random.uniform([batch_size, self.memory_size], minval=0.0, maxval=1.0)  # 调整范围
        )

        with tf.GradientTape() as tape:
            write_weights = self.write_weight_calculator.compute(
                write_content_weights=write_content_weights,
                allocation_gate=allocation_gate,
                write_gate=write_gate,
                prev_usage=prev_usage,
                training=True
            )
            # 修改损失函数，使其对输入更敏感
            loss = tf.reduce_sum(write_weights * write_content_weights)

        gradients = tape.gradient(loss, [write_content_weights, allocation_gate, write_gate, prev_usage])

        # 确保梯度不为 None 并具有合理的值
        for grad in gradients:
            self.assertIsNotNone(grad)
            grad_norm = tf.norm(grad).numpy()
            print("Gradient norm:", grad_norm)
            self.assertGreater(grad_norm, 1e-8)  # 调整阈值
    def test_write_weights_normalization(self):
        """
        Test that write_weights sum up appropriately.
        """
        batch_size = 3

        # Create test inputs
        write_content_weights = tf.random.uniform(
            [batch_size, self.num_writes, self.memory_size], minval=0.0, maxval=1.0
        )
        allocation_gate = tf.ones([batch_size, self.num_writes], dtype=tf.float32)
        write_gate = tf.ones([batch_size, self.num_writes], dtype=tf.float32)
        prev_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)

        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # Since allocation_gate and write_gate are 1, write_weights should be equal to allocation_weights
        allocation_weights = self.write_weight_calculator.compute_allocation_weights(prev_usage)
        allocation_weights = tf.expand_dims(allocation_weights, axis=1)  # [batch_size, 1, memory_size]
        allocation_weights = tf.tile(allocation_weights, [1, self.num_writes, 1])

        # Compare write_weights and allocation_weights
        self.assertAllClose(write_weights.numpy(), allocation_weights.numpy(), atol=1e-6)

        # Check that write_weights sum to 1 over the memory_size dimension
        sum_write_weights = tf.reduce_sum(write_weights, axis=-1)  # [batch_size, num_writes]
        self.assertAllClose(sum_write_weights.numpy(), np.ones_like(sum_write_weights.numpy()), atol=1e-6)


if __name__ == '__main__':
    tf.test.main()