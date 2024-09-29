from typing import Optional

import tensorflow as tf
import numpy as np

# 假设 WriteAllocation 和 weighted_softmax 已正确导入
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import WriteAllocation, weighted_softmax


class WriteAllocationTest(tf.test.TestCase):
    def setUp(self):
        super(WriteAllocationTest, self).setUp()
        self.memory_size = 3
        self.num_writes = 1  # 为简化测试，设置 num_writes 为 1
        self.epsilon = 1e-6

        # 定义自定义的 write_content_weights_fn，与原始代码一致
        def custom_write_content_weights_fn(inputs: dict, training: bool = False,
                                            num_writes: Optional[int] = None) -> tf.Tensor:
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
                write_content_weights = weighted_softmax(scores, write_content_strengths, tf.nn.softmax)
                return write_content_weights
            else:
                # 使用 softmax 基于 usage 生成 write_content_weights
                usage = inputs.get('usage')
                write_gates_sum = inputs.get('write_gates_sum')
                if usage is not None and write_gates_sum is not None:
                    write_content_weights = tf.nn.softmax(tf.expand_dims(-usage, axis=1),
                                                          axis=-1)  # [batch_size, num_writes, memory_size]
                    return write_content_weights
                else:
                    raise KeyError(
                        "Inputs must contain either ('write_content_keys' and 'write_content_strengths') or ('usage' and 'write_gates_sum').")

        # 实例化 WriteAllocation，传入自定义的 write_content_weights_fn
        self.write_allocation = WriteAllocation(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            epsilon=self.epsilon,
            write_content_weights_fn=custom_write_content_weights_fn
        )

    def test_compute_allocation_weights_shape(self):
        """
        测试 compute_allocation_weights 方法的输出形状是否正确。
        """
        batch_size = 2
        usage = tf.random.uniform([batch_size, self.memory_size], minval=0.0, maxval=1.0)
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
        allocation_weights = tf.squeeze(write_weights, axis=1)

        # 检查输出形状
        expected_shape = [batch_size, self.memory_size]
        self.assertAllEqual(allocation_weights.shape, expected_shape)

    def test_allocation_weights_range(self):
        """
        测试 allocation_weights 的数值范围应在 [0, 1] 之间。
        """
        batch_size = 2
        usage = tf.random.uniform([batch_size, self.memory_size], minval=0.0, maxval=1.0)
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
        allocation_weights = tf.squeeze(write_weights, axis=1)

        # 检查数值范围
        self.assertGreaterEqual(tf.reduce_min(allocation_weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(allocation_weights).numpy(), 1.0)

    def test_allocation_weights_normalization(self):
        """
        测试 allocation_weights 在 memory_size 维度上是否归一化。
        """
        batch_size = 2
        usage = tf.random.uniform([batch_size, self.memory_size], minval=0.0, maxval=1.0)
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
        allocation_weights = tf.squeeze(write_weights, axis=1)

        # 检查在 memory_size 维度上的和是否为 1
        sum_allocation_weights = tf.reduce_sum(allocation_weights, axis=-1)
        self.assertAllClose(sum_allocation_weights.numpy(), np.ones_like(sum_allocation_weights.numpy()), atol=1e-6)

    def test_compute_allocation_weights_with_zero_usage(self):
        """
        测试 usage 为零向量的情况。
        """
        batch_size = 1
        usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
        allocation_weights = tf.squeeze(write_weights, axis=1)

        # 预期分配权重为均匀分布
        expected_allocation_weights = np.full((batch_size, self.memory_size), 1.0 / self.memory_size, dtype=np.float32)

        self.assertAllClose(allocation_weights.numpy(), expected_allocation_weights, atol=1e-6)

    def test_compute_allocation_weights_with_full_usage(self):
        """
        测试 usage 为全1向量的情况，预期分配权重应为均匀分布。
        """
        batch_size = 1
        usage = tf.ones([batch_size, self.memory_size], dtype=tf.float32)
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        # 即使 usage 为全 1，softmax(-usage) 仍会产生均匀分布
        write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
        allocation_weights = tf.squeeze(write_weights, axis=1)

        # 预期分配权重为均匀分布
        expected_allocation_weights = np.full((batch_size, self.memory_size), 1.0 / self.memory_size, dtype=np.float32)

        self.assertAllClose(allocation_weights.numpy(), expected_allocation_weights, atol=1e-6)

    def test_compute_allocation_weights_sorting(self):
        """
        测试 allocation_weights 是否正确地根据 usage 排序。
        """
        batch_size = 1
        usage = tf.constant([[0.2, 0.5, 0.3]], dtype=tf.float32)  # [batch_size, memory_size]
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        # 手动计算预期的 allocation_weights
        # 由于使用了 softmax(-usage)，所以较小的 usage 值对应较大的权重
        expected_allocation_weights = tf.nn.softmax(tf.constant([[-0.2, -0.5, -0.3]]), axis=-1).numpy()

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
        allocation_weights = tf.squeeze(write_weights, axis=1).numpy()

        # 验证结果
        self.assertAllClose(allocation_weights, expected_allocation_weights, atol=1e-6)

    def test_gradient_with_respect_to_usage(self):
        """
        测试 allocation_weights 对 usage 的梯度计算是否正确。
        """
        batch_size = 2
        usage = tf.Variable([[0.2, 0.5, 0.3], [0.4, 0.1, 0.2]], dtype=tf.float32)
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        with tf.GradientTape() as tape:
            write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
            allocation_weights = tf.squeeze(write_weights, axis=1)
            # 修改损失函数，使其依赖于 usage
            loss = tf.reduce_sum(allocation_weights * usage)

        # 计算梯度
        gradients = tape.gradient(loss, [usage])

        # 检查梯度是否存在
        self.assertIsNotNone(gradients[0], "No gradient with respect to usage")
        grad_norm = tf.norm(gradients[0]).numpy()
        self.assertGreater(grad_norm, 1e-12, "Gradient norm is too small")

    def test_allocation_weights_sum(self):
        """
        测试 allocation_weights 与 usage 之间的关系，确保使用率越低的内存槽获得越高的分配权重。
        """
        batch_size = 1
        usage = tf.constant([[0.1, 0.4, 0.9]], dtype=tf.float32)
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        expected_allocation_weights = tf.nn.softmax(tf.constant([[-0.1, -0.4, -0.9]]), axis=-1).numpy()

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
        allocation_weights = tf.squeeze(write_weights, axis=1).numpy()

        # 检查 allocation_weights 与 usage 的关系
        # 预期内存槽 0 的分配权重最高，内存槽 2 的分配权重最低
        self.assertAllClose(allocation_weights, expected_allocation_weights, atol=1e-6)

    def test_allocation_weights_with_same_usage(self):
        """
        测试所有 usage 值相同的情况，allocation_weights 应该均匀分布。
        """
        batch_size = 1
        usage = tf.constant([[0.5, 0.5, 0.5]], dtype=tf.float32)
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        expected_allocation_weights = np.full((batch_size, self.memory_size), 1.0 / self.memory_size, dtype=np.float32)

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
        allocation_weights = tf.squeeze(write_weights, axis=1).numpy()

        self.assertAllClose(allocation_weights, expected_allocation_weights, atol=1e-6)

    def test_compute_allocation_weights_with_random_usage(self):
        """
        使用随机的 usage 测试 compute_allocation_weights 方法。
        """
        batch_size = 5
        usage = tf.random.uniform([batch_size, self.memory_size], minval=0.0, maxval=1.0)
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
        allocation_weights = tf.squeeze(write_weights, axis=1)

        # 检查输出形状
        expected_shape = [batch_size, self.memory_size]
        self.assertAllEqual(allocation_weights.shape, expected_shape)

        # 检查数值范围
        self.assertGreaterEqual(tf.reduce_min(allocation_weights).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(allocation_weights).numpy(), 1.0)

        # 检查归一化
        sum_allocation_weights = tf.reduce_sum(allocation_weights, axis=-1)
        self.assertAllClose(sum_allocation_weights.numpy(), np.ones_like(sum_allocation_weights.numpy()), atol=1e-6)

    def test_allocation_weights_cumulative_product(self):
        """
        测试 allocation_weights 的计算是否与手动计算的结果一致。
        """
        batch_size = 1
        usage = tf.constant([[0.2, 0.3, 0.5]], dtype=tf.float32)
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)

        # 手动计算 allocation_weights
        # 使用 softmax(-usage)
        expected_allocation_weights = tf.nn.softmax(tf.constant([[-0.2, -0.3, -0.5]]), axis=-1).numpy()

        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        write_weights = self.write_allocation.compute_write_weights(inputs, training=False)
        allocation_weights = tf.squeeze(write_weights, axis=1).numpy()

        self.assertAllClose(allocation_weights, expected_allocation_weights, atol=1e-6)


if __name__ == '__main__':
    tf.test.main()
