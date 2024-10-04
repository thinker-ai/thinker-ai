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
        usage = prev_usage + self.epsilon  # 防止除零
        usage_sorted, indices = tf.nn.top_k(-usage, k=self.memory_size)
        usage_sorted = -usage_sorted  # [1, 4]
        one_minus_u = 1 - usage_sorted  # [1, 4]
        cumprod = tf.math.cumprod(one_minus_u, axis=1, exclusive=False)  # [1, 4]
        shifted_cumprod = tf.concat([
            tf.ones([tf.shape(cumprod)[0], 1], dtype=cumprod.dtype),
            cumprod[:, :-1]
        ], axis=1)  # [1, 4]
        allocation_weights_sorted = one_minus_u * shifted_cumprod  # [1, 4]

        # 归一化 allocation_weights_sorted
        sum_allocation = tf.reduce_sum(allocation_weights_sorted, axis=1, keepdims=True)  # [1, 1]
        allocation_weights_normalized = tf.where(
            sum_allocation > 1e-6,
            allocation_weights_sorted / sum_allocation,
            tf.ones_like(allocation_weights_sorted) / self.memory_size
        )  # [1, 4]

        # 手动执行 scatter
        batch_size_dynamic = tf.shape(indices)[0]
        memory_size_dynamic = self.memory_size

        batch_indices = tf.range(batch_size_dynamic)[:, tf.newaxis]  # [B,1]
        batch_indices = tf.tile(batch_indices, [1, memory_size_dynamic])  # [B,M]
        scatter_indices = tf.stack([batch_indices, indices], axis=2)  # [B,M,2]
        scatter_indices_flat = tf.reshape(scatter_indices, [-1, 2])  # [B*M,2]
        allocation_weights_flat = tf.reshape(allocation_weights_normalized, [-1])  # [B*M]

        allocation_weights = tf.scatter_nd(scatter_indices_flat, allocation_weights_flat,
                                           [batch_size_dynamic, memory_size_dynamic])  # [B,M]

        allocation_weights = allocation_weights.numpy()  # [1,4]

        # 扩展和复制
        allocation_weights_expanded = allocation_weights.reshape(batch_size, 1, self.memory_size)  # [1,1,4]
        allocation_weights_expanded = np.tile(allocation_weights_expanded, [1, self.num_writes, 1])  # [1,2,4]

        # 计算未归一化的 expected_write_weights
        allocation_gate_expanded = allocation_gate.numpy()[..., np.newaxis]  # [1,2,1]
        write_gate_expanded = write_gate.numpy()[..., np.newaxis]  # [1,2,1]

        expected_write_weights_unnormalized = write_gate_expanded * (
                allocation_gate_expanded * allocation_weights_expanded +
                (1 - allocation_gate_expanded) * write_content_weights.numpy()
        )  # [1,2,4]

        # 归一化 expected_write_weights
        sum_expected = np.sum(expected_write_weights_unnormalized, axis=-1, keepdims=True) + self.epsilon
        expected_write_weights = expected_write_weights_unnormalized / sum_expected  # [1,2,4]

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

        # 手动计算 expected_allocation_weights
        usage = prev_usage + self.epsilon  # 防止除零
        usage_sorted, indices = tf.nn.top_k(-usage, k=self.memory_size)
        usage_sorted = -usage_sorted  # [1, 4]
        one_minus_u = 1 - usage_sorted  # [1, 4]
        cumprod = tf.math.cumprod(one_minus_u, axis=1, exclusive=False)  # [1, 4]
        shifted_cumprod = tf.concat([
            tf.ones([tf.shape(cumprod)[0], 1], dtype=cumprod.dtype),
            cumprod[:, :-1]
        ], axis=1)  # [1, 4]
        allocation_weights_sorted = one_minus_u * shifted_cumprod  # [1, 4]

        # 归一化 allocation_weights_sorted
        sum_allocation = tf.reduce_sum(allocation_weights_sorted, axis=1, keepdims=True)  # [1, 1]
        allocation_weights_normalized = tf.where(
            sum_allocation > 1e-6,
            allocation_weights_sorted / sum_allocation,
            tf.ones_like(allocation_weights_sorted) / self.memory_size
        )  # [1, 4]

        # 手动执行 scatter
        batch_size_dynamic = tf.shape(indices)[0]
        memory_size_dynamic = self.memory_size

        batch_indices = tf.range(batch_size_dynamic)[:, tf.newaxis]  # [B,1]
        batch_indices = tf.tile(batch_indices, [1, memory_size_dynamic])  # [B,M]
        scatter_indices = tf.stack([batch_indices, indices], axis=2)  # [B,M,2]
        scatter_indices_flat = tf.reshape(scatter_indices, [-1, 2])  # [B*M,2]
        allocation_weights_flat = tf.reshape(allocation_weights_normalized, [-1])  # [B*M]

        allocation_weights = tf.scatter_nd(scatter_indices_flat, allocation_weights_flat, [batch_size_dynamic, memory_size_dynamic])  # [B,M]

        allocation_weights = allocation_weights.numpy()  # [1,4]

        # 扩展和复制
        allocation_weights_expanded = allocation_weights.reshape(batch_size, 1, self.memory_size)  # [1,1,4]
        allocation_weights_expanded = np.tile(allocation_weights_expanded, [1, self.num_writes, 1])  # [1,2,4]

        # 计算 expected_write_weights
        allocation_gate_expanded = allocation_gate.numpy()[..., np.newaxis]  # [1,2,1]
        write_gate_expanded = write_gate.numpy()[..., np.newaxis]  # [1,2,1]

        expected_write_weights = write_gate_expanded * (
                allocation_gate_expanded * allocation_weights_expanded +
                (1 - allocation_gate_expanded) * write_content_weights.numpy()
        )  # [1,2,4]

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

        # Check that write_weights sum to 1 over the memory_size dimension
        sum_write_weights = tf.reduce_sum(write_weights, axis=-1)  # [batch_size, num_writes]
        self.assertAllClose(sum_write_weights.numpy(), np.ones_like(sum_write_weights.numpy()), atol=1e-6)

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

        # 手动计算 expected_allocation_weights
        usage = prev_usage + self.epsilon  # 防止除零
        usage_sorted, indices = tf.nn.top_k(-usage, k=self.memory_size)
        usage_sorted = -usage_sorted  # [1, 4]
        one_minus_u = 1 - usage_sorted  # [1, 4]
        one_minus_u = tf.maximum(one_minus_u, 0)  # 确保非负
        cumprod = tf.math.cumprod(one_minus_u, axis=1, exclusive=False)  # [1, 4]
        shifted_cumprod = tf.concat([
            tf.ones([tf.shape(cumprod)[0], 1], dtype=cumprod.dtype),
            cumprod[:, :-1]
        ], axis=1)  # [1, 4]
        allocation_weights_sorted = one_minus_u * shifted_cumprod  # [1, 4]

        # 归一化 allocation_weights_sorted
        sum_allocation = tf.reduce_sum(allocation_weights_sorted, axis=1, keepdims=True)  # [1, 1]
        allocation_weights_normalized = tf.where(
            sum_allocation > 1e-6,
            allocation_weights_sorted / sum_allocation,
            tf.ones_like(allocation_weights_sorted) / self.memory_size
        )  # [1, 4]

        # 手动执行 scatter
        batch_size_dynamic = tf.shape(indices)[0]
        memory_size_dynamic = self.memory_size

        batch_indices = tf.range(batch_size_dynamic)[:, tf.newaxis]  # [B,1]
        batch_indices = tf.tile(batch_indices, [1, memory_size_dynamic])  # [B,M]
        scatter_indices = tf.stack([batch_indices, indices], axis=2)  # [B,M,2]
        scatter_indices_flat = tf.reshape(scatter_indices, [-1, 2])  # [B*M,2]
        allocation_weights_flat = tf.reshape(allocation_weights_normalized, [-1])  # [B*M]

        allocation_weights = tf.scatter_nd(scatter_indices_flat, allocation_weights_flat, [batch_size_dynamic, memory_size_dynamic])  # [B,M]

        allocation_weights = allocation_weights.numpy()  # [1,4]

        # 扩展和复制
        allocation_weights_expanded = allocation_weights.reshape(batch_size, 1, self.memory_size)  # [1,1,4]
        allocation_weights_expanded = np.tile(allocation_weights_expanded, [1, self.num_writes, 1])  # [1,2,4]

        # 计算 expected_write_weights
        allocation_gate_expanded = allocation_gate.numpy()[..., np.newaxis]  # [1,2,1]
        write_gate_expanded = write_gate.numpy()[..., np.newaxis]  # [1,2,1]

        expected_write_weights = write_gate_expanded * (
                allocation_gate_expanded * allocation_weights_expanded +
                (1 - allocation_gate_expanded) * write_content_weights.numpy()
        )  # [1,2,4]

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

    def test_memory_size_one(self):
        """
        测试 memory_size 为1的情况。
        """
        batch_size = 2
        memory_size = 1
        num_writes = 1

        # 实例化写入权重计算器
        write_weight_calculator = DefaultWriteWeightCalculator(
            memory_size=memory_size,
            num_writes=num_writes
        )

        # 设置特定测试输入
        write_content_weights = tf.constant([
            [[0.5]],
            [[0.8]]
        ], dtype=tf.float32)  # [2, 1, 1]
        allocation_gate = tf.constant([[1.0], [0.0]], dtype=tf.float32)  # [2, 1]
        write_gate = tf.constant([[1.0], [1.0]], dtype=tf.float32)  # [2, 1]
        prev_usage = tf.constant([
            [0.0],
            [1.0]
        ], dtype=tf.float32)  # [2, 1]

        # 计算写入权重
        write_weights = write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 期望的写入权重（归一化后）
        expected_write_weights = tf.constant([
            [[1.0]],  # allocation_gate=1, allocation_weights=1
            [[1.0]]  # allocation_gate=0, write_weights_normalized=1.0
        ], dtype=tf.float32)  # [2, 1, 1]

        # 比较预期和实际的写入权重
        self.assertAllClose(write_weights.numpy(), expected_write_weights.numpy(), atol=1e-6)
    def test_gates_zero(self):
        """
        测试 allocation_gate 和 write_gate 为0的情况。
        """
        batch_size = 1
        memory_size = 4
        num_writes = 2

        # 实例化写入权重计算器
        write_weight_calculator = DefaultWriteWeightCalculator(
            memory_size=memory_size,
            num_writes=num_writes
        )

        # 设置特定测试输入
        write_content_weights = tf.constant([
            [[0.1, 0.2, 0.3, 0.4],
             [0.5, 0.6, 0.7, 0.8]]
        ], dtype=tf.float32)  # [1, 2, 4]
        allocation_gate = tf.constant([[0.0, 0.0]], dtype=tf.float32)  # [1, 2]
        write_gate = tf.constant([[0.0, 0.0]], dtype=tf.float32)  # [1, 2]
        prev_usage = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)  # [1, 4]

        # 计算写入权重
        write_weights = write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 期望的写入权重为0
        expected_write_weights = tf.zeros([batch_size, num_writes, memory_size], dtype=tf.float32)

        # 比较预期和实际的写入权重
        self.assertAllClose(write_weights.numpy(), expected_write_weights.numpy(), atol=1e-6)

    def test_write_content_weights_zero(self):
        """
        测试 write_content_weights 全为0的情况。
        """
        batch_size = 1
        memory_size = 4
        num_writes = 2

        # 实例化写入权重计算器
        write_weight_calculator = DefaultWriteWeightCalculator(
            memory_size=memory_size,
            num_writes=num_writes
        )

        # 设置特定测试输入
        write_content_weights = tf.zeros([batch_size, num_writes, memory_size], dtype=tf.float32)  # [1, 2, 4]
        allocation_gate = tf.constant([[1.0, 1.0]], dtype=tf.float32)  # [1, 2]
        write_gate = tf.constant([[1.0, 1.0]], dtype=tf.float32)  # [1, 2]
        prev_usage = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)  # [1, 4]

        # 计算写入权重
        write_weights = write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 计算期望的分配权重
        allocation_weights = write_weight_calculator.compute_allocation_weights(prev_usage).numpy()  # [1, 4]
        allocation_weights_expanded = np.tile(allocation_weights[:, np.newaxis, :], [1, num_writes, 1])  # [1, 2, 4]

        # 期望的写入权重
        expected_write_weights = allocation_weights_expanded  # [1, 2, 4]

        # 比较预期和实际的写入权重
        self.assertAllClose(write_weights.numpy(), expected_write_weights, atol=1e-6)

    def test_prev_usage_nan_inf(self):
        """
        测试 prev_usage 包含 NaN 或 Inf 的情况。
        """
        batch_size = 2
        memory_size = 4
        num_writes = 2

        # 实例化写入权重计算器
        write_weight_calculator = DefaultWriteWeightCalculator(
            memory_size=memory_size,
            num_writes=num_writes
        )

        # 设置特定测试输入，包含 NaN 和 Inf
        write_content_weights = tf.random.uniform([batch_size, num_writes, memory_size], minval=0.0, maxval=1.0)
        allocation_gate = tf.random.uniform([batch_size, num_writes], minval=0.0, maxval=1.0)
        write_gate = tf.random.uniform([batch_size, num_writes], minval=0.0, maxval=1.0)
        prev_usage = tf.constant([
            [0.1, float('nan'), 0.3, 0.4],
            [0.5, float('inf'), 0.7, 0.8]
        ], dtype=tf.float32)  # [2, 4]

        # 计算写入权重，确保不会抛出异常
        try:
            write_weights = write_weight_calculator.compute(
                write_content_weights=write_content_weights,
                allocation_gate=allocation_gate,
                write_gate=write_gate,
                prev_usage=prev_usage,
                training=False
            )
            # 如果需要，可以添加额外的断言来检查输出的有效性
            self.assertTrue(True)  # 如果没有异常，测试通过
        except Exception as e:
            self.fail(f"compute 方法在 prev_usage 包含 NaN 或 Inf 时抛出了异常: {e}")

    def test_large_batch_and_memory_size(self):
        """
        测试大规模 memory_size 和 num_writes 的情况。
        """
        batch_size = 10
        memory_size = 1024
        num_writes = 16

        write_weight_calculator = DefaultWriteWeightCalculator(
            memory_size=memory_size,
            num_writes=num_writes
        )

        # 设置随机测试输入
        write_content_weights = tf.random.uniform(
            [batch_size, num_writes, memory_size], minval=0.0, maxval=1.0
        )
        allocation_gate = tf.random.uniform(
            [batch_size, num_writes], minval=0.0, maxval=1.0
        )
        write_gate = tf.random.uniform(
            [batch_size, num_writes], minval=0.0, maxval=1.0
        )
        prev_usage = tf.random.uniform(
            [batch_size, memory_size], minval=0.0, maxval=1.0
        )

        # 运行 compute 方法
        write_weights = write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=False
        )

        # 检查 write_weights 的形状和范围
        self.assertAllEqual(write_weights.shape, [batch_size, num_writes, memory_size])
        self.assertGreaterEqual(tf.reduce_min(write_weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(write_weights).numpy(), 1.0)

        # 检查归一化
        sum_write_weights = tf.reduce_sum(write_weights, axis=-1)  # [batch_size, num_writes]
        self.assertAllClose(sum_write_weights.numpy(), np.ones_like(sum_write_weights.numpy()), atol=1e-6)
    def test_write_content_weights_non_normalized(self):
        """
        测试 write_content_weights 非标准化的情况。
        """
        batch_size = 2
        memory_size = 4
        num_writes = 2

        write_weight_calculator = DefaultWriteWeightCalculator(
            memory_size=memory_size,
            num_writes=num_writes
        )

        # 创建非标准化的 write_content_weights（可能超过1或小于0）
        write_content_weights = tf.constant([
            [[1.5, -0.5, 2.0, 0.0],
             [0.0, 1.0, -1.0, 3.0]],
            [[-1.0, 2.0, 0.5, 1.5],
             [2.0, -0.5, 1.0, 0.5]]
        ], dtype=tf.float32)  # [2, 2, 4]
        allocation_gate = tf.constant([[0.5, 1.2], [1.5, 0.3]], dtype=tf.float32)  # [2, 2]
        write_gate = tf.constant([[0.8, -0.2], [1.0, 1.0]], dtype=tf.float32)  # [2, 2]
        prev_usage = tf.constant([
            [0.3, 0.7, 0.2, 0.5],
            [0.6, 0.1, 0.4, 0.9]
        ], dtype=tf.float32)  # [2, 4]

        # 运行 compute 方法
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
    def test_memory_size_zero(self):
        """
        测试 memory_size 为0的情况，应抛出异常。
        """
        with self.assertRaises(ValueError):
            write_weight_calculator = DefaultWriteWeightCalculator(
                memory_size=0,
                num_writes=1
            )

    def test_num_writes_zero(self):
        """
        测试 num_writes 为0的情况，应抛出异常。
        """
        with self.assertRaises(ValueError):
            write_weight_calculator = DefaultWriteWeightCalculator(
                memory_size=4,
                num_writes=0
            )

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
            [batch_size, memory_size], minval=0.0, maxval=1.0
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
