#thinker_ai_tests/agent/memory/humanoid_memory/dnc/addressing_test.py
"""Tests for memory addressing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc_new import util, addressing


class WeightedSoftmaxTest(tf.test.TestCase):

    def testValues(self):
        batch_size = 5
        num_heads = 3
        memory_size = 7

        # 使用 NumPy 生成输入数据
        activations_data = np.random.randn(batch_size, num_heads, memory_size).astype(np.float32)
        weights_data = np.ones((batch_size, num_heads), dtype=np.float32)

        # 定义输入张量而不是占位符
        activations = tf.convert_to_tensor(activations_data)
        weights = tf.convert_to_tensor(weights_data)

        # 调用加权 Softmax 函数
        observed = addressing.weighted_softmax(activations, weights, tf.identity)

        # TensorFlow 自带的 softmax 函数用于对比
        expected = tf.nn.softmax(activations, axis=-1)

        # 使用 eager execution 模式直接运行
        observed_val = observed.numpy()  # 通过 .numpy() 获得结果
        expected_val = expected.numpy()

        # 验证 observed 和 expected 是否接近
        self.assertAllClose(observed_val, expected_val)


class CosineWeightsTest(tf.test.TestCase):

    def testShape(self):
        batch_size = 5
        num_heads = 3
        memory_size = 7
        word_size = 2

        module = addressing.CosineWeights(num_heads, word_size)

        # 创建输入张量
        mem = tf.random.normal([batch_size, memory_size, word_size])
        keys = tf.random.normal([batch_size, num_heads, word_size])
        strengths = tf.random.normal([batch_size, num_heads])

        # 调用模块
        weights = module({'memory': mem, 'keys': keys, 'strengths': strengths})

        # 检查输出形状
        self.assertTrue(weights.shape.is_compatible_with([batch_size, num_heads, memory_size]))

    def testValues(self):
        batch_size = 5
        num_heads = 4
        memory_size = 10
        word_size = 2

        mem_data = np.random.randn(batch_size, memory_size, word_size).astype(np.float32)
        np.copyto(mem_data[0, 0], [1, 2])
        np.copyto(mem_data[0, 1], [3, 4])
        np.copyto(mem_data[0, 2], [5, 6])

        keys_data = np.random.randn(batch_size, num_heads, word_size).astype(np.float32)
        np.copyto(keys_data[0, 0], [5, 6])
        np.copyto(keys_data[0, 1], [1, 2])
        np.copyto(keys_data[0, 2], [5, 6])
        np.copyto(keys_data[0, 3], [3, 4])
        strengths_data = np.random.randn(batch_size, num_heads).astype(np.float32)

        module = addressing.CosineWeights(num_heads, word_size)

        # 转换为张量
        mem = tf.convert_to_tensor(mem_data)
        keys = tf.convert_to_tensor(keys_data)
        strengths = tf.convert_to_tensor(strengths_data)

        # 调用模块
        weights = module({'memory': mem, 'keys': keys, 'strengths': strengths})

        # 使用 eager execution 直接运行
        weights_val = weights.numpy()

        # 手动验证结果
        strengths_softplus = np.log(1 + np.exp(strengths_data))
        similarity = np.zeros((memory_size))

        for b in range(batch_size):
            for h in range(num_heads):
                key = keys_data[b, h]
                key_norm = np.linalg.norm(key)

                for m in range(memory_size):
                    row = mem_data[b, m]
                    similarity[m] = np.dot(key, row) / (key_norm * np.linalg.norm(row))

                similarity = np.exp(similarity * strengths_softplus[b, h])
                similarity /= similarity.sum()
                self.assertAllClose(weights_val[b, h], similarity, atol=1e-4, rtol=1e-4)

    def testDivideByZero(self):
        batch_size = 5
        num_heads = 4
        memory_size = 4  # Ensure that memory_size matches num_heads to avoid shape incompatibility
        word_size = 2

        module = addressing.CosineWeights(num_heads, word_size)

        # 构建随机张量
        keys = tf.random.normal([batch_size, num_heads, word_size])
        strengths = tf.random.normal([batch_size, num_heads])

        # 第一行设置为1，其他行设置为0，调整 memory_size 和 num_heads 一致
        first_row_ones = tf.ones([batch_size, 1, word_size], dtype=tf.float32)
        remaining_zeros = tf.zeros([batch_size, memory_size - 1, word_size], dtype=tf.float32)
        mem = tf.concat([first_row_ones, remaining_zeros], axis=1)

        # 调用模块
        output = module({'memory': mem, 'keys': keys, 'strengths': strengths})

        # 计算梯度
        with tf.GradientTape() as tape:
            tape.watch([mem, keys, strengths])
            output = module({'memory': mem, 'keys': keys, 'strengths': strengths})

        gradients = tape.gradient(output, [mem, keys, strengths])

        # 检查输出和梯度是否有NaN
        output_val = output.numpy()
        gradients_val = [g.numpy() for g in gradients]

        self.assertFalse(np.any(np.isnan(output_val)))
        self.assertFalse(np.any(np.isnan(gradients_val[0])))
        self.assertFalse(np.any(np.isnan(gradients_val[1])))
        self.assertFalse(np.any(np.isnan(gradients_val[2])))


class TemporalLinkageTest(tf.test.TestCase):
    class TemporalLinkageTest(tf.test.TestCase):

        def testModule(self):
            batch_size = 7
            memory_size = 4
            num_reads = 11
            num_writes = 5
            module = addressing.TemporalLinkage(memory_size=memory_size, num_writes=num_writes)

            # Initialize state
            prev_link_in = tf.zeros((batch_size, num_writes, memory_size, memory_size), dtype=tf.float32)
            prev_precedence_weights_in = tf.zeros((batch_size, num_writes, memory_size), dtype=tf.float32)
            write_weights_in = tf.random.uniform((batch_size, num_writes, memory_size))

            calc_state = module({
                'write_weights': write_weights_in,
                'prev_linkage': addressing.TemporalLinkageState(
                    link=prev_link_in,
                    precedence_weights=prev_precedence_weights_in
                )
            })

            num_steps = 5
            for i in range(num_steps):
                write_weights = np.random.rand(batch_size, num_writes, memory_size).astype(np.float32)
                write_weights /= write_weights.sum(2, keepdims=True) + 1

                # Simulate specific writing patterns in final steps
                if i == num_steps - 2:
                    write_weights[0, 0, :] = util.one_hot(memory_size, 0)
                    write_weights[0, 1, :] = util.one_hot(memory_size, 3)
                elif i == num_steps - 1:
                    write_weights[0, 0, :] = util.one_hot(memory_size, 1)
                    write_weights[0, 1, :] = util.one_hot(memory_size, 2)

                # Calculate state
                calc_state = module({
                    'write_weights': tf.constant(write_weights, dtype=tf.float32),
                    'prev_linkage': addressing.TemporalLinkageState(
                        link=prev_link_in,
                        precedence_weights=prev_precedence_weights_in
                    )
                })

            # link should be bounded in range [0, 1]
            self.assertGreaterEqual(tf.reduce_min(calc_state.link).numpy(), 0)
            self.assertLessEqual(tf.reduce_max(calc_state.link).numpy(), 1)

            # 使用 tf.meshgrid 获取对角线元素的索引
            diag_indices = tf.range(memory_size)

            # 构建批量和写头的索引
            batch_indices, write_indices = tf.meshgrid(
                tf.range(batch_size), tf.range(num_writes), indexing='ij'
            )

            # 使用 tf.meshgrid 构建完整索引
            diag_indices_i, diag_indices_j = tf.meshgrid(diag_indices, diag_indices, indexing='ij')

            # 构建用于 gather 的索引
            indices = tf.stack([batch_indices, write_indices, diag_indices_i, diag_indices_j], axis=-1)

            # 使用 tf.gather_nd 获取链路矩阵中的对角线元素
            diag_elements = tf.gather_nd(calc_state.link, indices)

            # 对角线元素应该为 0，进行断言
            self.assertAllEqual(diag_elements, np.zeros([batch_size, num_writes, memory_size]))

            # link rows and columns should sum to at most 1
            self.assertLessEqual(tf.reduce_max(tf.reduce_sum(calc_state.link, axis=2)).numpy(), 1)
            self.assertLessEqual(tf.reduce_max(tf.reduce_sum(calc_state.link, axis=3)).numpy(), 1)

            # Test transitions in batch 0: head 0 (0->1), and head 1 (3->2)
            self.assertAllEqual(calc_state.link[0, 0, :, 0], util.one_hot(memory_size, 1))
            self.assertAllEqual(calc_state.link[0, 1, :, 3], util.one_hot(memory_size, 2))

            # Now test calculation of forward and backward read weights
            prev_read_weights = np.random.rand(batch_size, num_reads, memory_size).astype(np.float32)
            prev_read_weights[0, 5, :] = util.one_hot(memory_size, 0)  # read head 5 at position 0
            prev_read_weights[0, 6, :] = util.one_hot(memory_size, 2)  # read head 6 at position 2

            forward_read_weights = module.directional_read_weights(
                link=tf.constant(calc_state.link),
                prev_read_weights=tf.constant(prev_read_weights),
                forward=True
            )
            backward_read_weights = module.directional_read_weights(
                link=tf.constant(calc_state.link),
                prev_read_weights=tf.constant(prev_read_weights),
                forward=False
            )

            # forward_read_weights and backward_read_weights values
            forward_read_weights_val = forward_read_weights.numpy()
            backward_read_weights_val = backward_read_weights.numpy()

            # Check directional weights calculated correctly
            self.assertAllEqual(forward_read_weights_val[0, 5, 0, :], util.one_hot(memory_size, 1))  # read=5, write=0
            self.assertAllEqual(backward_read_weights_val[0, 6, 1, :], util.one_hot(memory_size, 3))  # read=6, write=1
    def testPrecedenceWeights(self):
        batch_size = 7
        memory_size = 3
        num_writes = 5
        module = addressing.TemporalLinkage(memory_size=memory_size, num_writes=num_writes)

        prev_precedence_weights = np.random.rand(batch_size, num_writes, memory_size).astype(np.float32)
        write_weights = np.random.rand(batch_size, num_writes, memory_size).astype(np.float32)

        # Normalize write weights and precedence weights
        write_weights /= write_weights.sum(2, keepdims=True) + 1
        prev_precedence_weights /= prev_precedence_weights.sum(2, keepdims=True) + 1

        # No writing in batch 0 head 1, full writing in batch 1 head 2
        write_weights[0, 1, :] = 0
        write_weights[1, 2, :] /= write_weights[1, 2, :].sum()

        precedence_weights = module._precedence_weights(
            prev_precedence_weights=tf.constant(prev_precedence_weights),
            write_weights=tf.constant(write_weights)
        )

        # Check that precedence weights are in the range [0, 1]
        self.assertGreaterEqual(tf.reduce_min(precedence_weights).numpy(), 0)
        self.assertLessEqual(tf.reduce_max(precedence_weights).numpy(), 1)

        # No writing in batch 0, head 1
        self.assertAllClose(precedence_weights[0, 1, :], prev_precedence_weights[0, 1, :])

        # Full writing in batch 1, head 2
        self.assertAllClose(precedence_weights[1, 2, :], write_weights[1, 2, :])


class FreenessTest(tf.test.TestCase):

    def testModule(self):
        batch_size = 5
        memory_size = 11
        num_reads = 3
        num_writes = 7
        module = addressing.Freeness(memory_size)

        free_gate = np.random.rand(batch_size, num_reads).astype(np.float32)

        # Produce read weights that sum to 1 for each batch and head.
        prev_read_weights = np.random.rand(batch_size, num_reads, memory_size).astype(np.float32)
        prev_read_weights[1, :, 3] = 0  # no read at batch 1, position 3; see below
        prev_read_weights /= prev_read_weights.sum(2, keepdims=True)
        prev_write_weights = np.random.rand(batch_size, num_writes, memory_size).astype(np.float32)
        prev_write_weights /= prev_write_weights.sum(2, keepdims=True)
        prev_usage = np.random.rand(batch_size, memory_size).astype(np.float32)

        # Add some special values that allow us to test the behaviour:
        prev_write_weights[1, 2, 3] = 1  # full write in batch 1, head 2, position 3
        prev_read_weights[2, 0, 4] = 1  # full read at batch 2, head 0, position 4
        free_gate[2, 0] = 1  # can free up all locations for batch 2, read head 0

        usage = module(
            {
                'write_weights': tf.constant(prev_write_weights),
                'free_gate': tf.constant(free_gate),
                'read_weights': tf.constant(prev_read_weights),
                'prev_usage': tf.constant(prev_usage)
            }
        )

        usage = usage.numpy()

        # Check all usages are between 0 and 1.
        self.assertGreaterEqual(np.min(usage), 0)
        self.assertLessEqual(np.max(usage), 1)

        # Check that the full write at batch 1, position 3 makes it fully used.
        self.assertEqual(usage[1][3], 1)

        # Check that the full free at batch 2, position 4 makes it fully free.
        self.assertEqual(usage[2][4], 0)

    def testWriteAllocationWeights(self):
        batch_size = 7
        memory_size = 23
        num_writes = 5
        module = addressing.Freeness(memory_size)

        usage = np.random.rand(batch_size, memory_size).astype(np.float32)
        write_gates = np.random.rand(batch_size, num_writes).astype(np.float32)

        # Turn off gates for heads 1 and 3 in batch 0. This doesn't scale down the
        # weighting, but it means that the usage doesn't change, so we should get
        # the same allocation weightings for: (1, 2) and (3, 4) (but all others
        # being different).
        write_gates[0, 1] = 0
        write_gates[0, 3] = 0
        # Turn heads 0 and 2 on for full effect.
        write_gates[0, 0] = 1
        write_gates[0, 2] = 1

        # In batch 1, make one of the usages 0 and another almost 0, so that these
        # entries get most of the allocation weights for the first and second heads.
        usage[1] = usage[1] * 0.9 + 0.1  # make sure all entries are in [0.1, 1]
        usage[1][4] = 0  # write head 0 should get allocated to position 4
        usage[1][3] = 1e-4  # write head 1 should get allocated to position 3
        write_gates[1, 0] = 1  # write head 0 fully on
        write_gates[1, 1] = 1  # write head 1 fully on

        weights = module.write_allocation_weights(
            usage=tf.constant(usage),
            write_gates=tf.constant(write_gates),
            num_writes=num_writes
        )

        weights = weights.numpy()

        # Check that all weights are between 0 and 1
        self.assertGreaterEqual(np.min(weights), 0)
        self.assertLessEqual(np.max(weights), 1)

        # Check that weights sum to close to 1
        self.assertAllClose(np.sum(weights, axis=2), np.ones([batch_size, num_writes]), atol=1e-3)

        # Check the same / different allocation weight pairs as described above.
        self.assertGreater(np.abs(weights[0, 0, :] - weights[0, 1, :]).max(), 0.1)
        self.assertAllEqual(weights[0, 1, :], weights[0, 2, :])
        self.assertGreater(np.abs(weights[0, 2, :] - weights[0, 3, :]).max(), 0.1)
        self.assertAllEqual(weights[0, 3, :], weights[0, 4, :])

        self.assertAllClose(weights[1][0], util.one_hot(memory_size, 4), atol=1e-3)
        self.assertAllClose(weights[1][1], util.one_hot(memory_size, 3), atol=1e-3)

    def testAllocation(self):
        batch_size = 7
        memory_size = 13
        usage = np.random.rand(batch_size, memory_size).astype(np.float32)
        module = addressing.Freeness(memory_size)
        allocation = module._allocation(tf.constant(usage))

        allocation = allocation.numpy()

        # 1. 确保 allocation 的形状正确
        print("allocation shape:", allocation.shape)

        # 2. Test that max allocation goes to min usage, and vice versa.
        self.assertAllEqual(np.argmin(usage, axis=1), np.argmax(allocation, axis=1))


if __name__ == '__main__':
    tf.test.main()
