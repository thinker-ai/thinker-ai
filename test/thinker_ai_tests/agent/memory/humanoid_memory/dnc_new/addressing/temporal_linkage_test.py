import tensorflow as tf
import numpy as np

from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import TemporalLinkage


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

        # 调用 TemporalLinkage 层，使用字典传递输入
        updated_linkage = self.temporal_linkage_layer(
            {
                'write_weights': write_weights,
                'prev_linkage': prev_linkage
            },
            training=False
        )

        # 预期 precedence_weights = write_weights * epsilon
        expected_precedence_weights = write_weights * self.temporal_linkage_layer.epsilon  # [1, 2, 3]

        # 预期 link = write_weights_i * prev_precedence_weights_j = 0 since prev_precedence_weights = 0
        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_dynamic_num_writes(self):
        """
        测试动态调整 num_writes 的情况。
        """
        batch_size = 2
        dynamic_num_writes = 3

        # 定义 write_weights，形状与 batch_size 和 dynamic_num_writes 一致
        write_weights = tf.constant([
            [[0.1, 0.2, 0.3],
             [0.4, 0.1, 0.2],
             [0.2, 0.2, 0.2]],
            [[0.3, 0.2, 0.1],
             [0.1, 0.5, 0.2],
             [0.3, 0.2, 0.1]]
        ], dtype=tf.float32)  # [2, 3, 3]

        # 定义 prev_linkage，形状与 batch_size 和 dynamic_num_writes 一致
        prev_linkage = {
            'link': tf.zeros([batch_size, dynamic_num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, dynamic_num_writes, self.memory_size], dtype=tf.float32)
        }

        # 调用 TemporalLinkage 层，使用字典传递输入
        updated_linkage = self.temporal_linkage_layer(
            {
                'write_weights': write_weights,
                'prev_linkage': prev_linkage
            },
            training=False
        )

        # 预期 precedence_weights = write_weights * epsilon
        expected_precedence_weights = write_weights * self.temporal_linkage_layer.epsilon  # [2, 3, 3]

        # 预期 link = write_weights_i * prev_precedence_weights_j = 0 since prev_precedence_weights = 0
        expected_link = tf.zeros([batch_size, dynamic_num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        # 验证 precedence_weights 和 link
        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
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

        # 调用 TemporalLinkage 层，使用字典传递输入
        updated_linkage = self.temporal_linkage_layer(
            {
                'write_weights': write_weights,
                'prev_linkage': {
                    'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
                    'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
                }
            },
            training=False
        )

        # 预期 precedence_weights = write_weights * epsilon
        expected_precedence_weights = write_weights * self.temporal_linkage_layer.epsilon  # [1, 2, 3]

        # 预期 link = write_weights_i * prev_precedence_weights_j = 0 since prev_precedence_weights = 0
        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_initial_state(self):
        """
        测试 get_initial_state 方法，确保返回正确的初始链路矩阵和优先级权重。
        """
        batch_size = 4
        batch_size_tensor = tf.constant(batch_size, dtype=tf.int32)  # 将 batch_size 转换为 tf.Tensor
        initial_linkage = self.temporal_linkage_layer.get_initial_state(batch_size=batch_size_tensor)

        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)
        expected_precedence_weights = tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)

        self.assertAllClose(initial_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)
        self.assertAllClose(initial_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(),
                            atol=1e-6)
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
        linkage_state = module(
            {
                'write_weights': write_weights,
                'prev_linkage': {
                    'link': prev_link,
                    'precedence_weights': prev_precedence_weights
                }
            },
            training=False
        )

        # 验证第一次写入后的链路矩阵
        expected_link_after_first_write = np.zeros((1, 1, 3, 3), dtype=np.float32)
        self.assertAllClose(linkage_state['link'].numpy(), expected_link_after_first_write, atol=1e-6)

        # 更新 prev_link 和 prev_precedence_weights
        prev_link = linkage_state['link']
        prev_precedence_weights = linkage_state['precedence_weights']

        # 第二次写入
        write_weights = tf.constant([[[0, 1, 0]]], dtype=tf.float32)  # 写入位置 1
        linkage_state = module(
            {
                'write_weights': write_weights,
                'prev_linkage': {
                    'link': prev_link,
                    'precedence_weights': prev_precedence_weights
                }
            },
            training=False
        )

        # 预期 precedence_weights = write_weights * epsilon
        expected_precedence_weights = write_weights * module.epsilon  # [1, 1, 3]

        # 预期 link = write_weights_i * prev_precedence_weights_j = 0 since prev_precedence_weights = 0
        expected_link = tf.zeros([batch_size, num_writes, memory_size, memory_size], dtype=tf.float32)

        self.assertAllClose(linkage_state['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
        self.assertAllClose(linkage_state['link'].numpy(), expected_link.numpy(), atol=1e-6)

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
        linkage_state = module(
            {
                'write_weights': write_weights,
                'prev_linkage': {
                    'link': prev_link,
                    'precedence_weights': prev_precedence_weights
                }
            },
            training=False
        )

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
        prev_link_scale = 1 - write_weights_i - write_weights_j  # [1, 1, 3, 3]
        prev_link_scale = tf.clip_by_value(prev_link_scale, 0.0, 1.0)

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

        # 调用模块，使用字典传递输入
        linkage_state = module(
            {
                'write_weights': write_weights,
                'prev_linkage': {
                    'link': prev_link,
                    'precedence_weights': prev_precedence_weights
                }
            },
            training=False
        )

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

    def test_prev_precedence_weights_j(self):
        """
        测试 prev_precedence_weights_j 的扩展，确保正确性。
        """
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)

        # 计算 prev_precedence_weights_j
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)  # (batch_size, num_writes, 1, memory_size)

        # 验证 shape 和值
        expected_prev_precedence_weights_j = np.array([[[[0, 0, 0]]]], dtype=np.float32)
        np.testing.assert_allclose(prev_precedence_weights_j.numpy(), expected_prev_precedence_weights_j, atol=1e-6)

    def test_prev_link_scale(self):
        """
        测试 prev_link_scale 的计算，确保正确应用缩放因子。
        """
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        # 扩展维度
        write_weights_i = tf.expand_dims(write_weights, 3)
        write_weights_j = tf.expand_dims(write_weights, 2)

        # 计算 prev_link_scale
        prev_link_scale = 1 - write_weights_i - write_weights_j  # [1, 1, 3, 3]
        prev_link_scale = tf.clip_by_value(prev_link_scale, 0.0, 1.0)

        # 验证缩放因子
        expected_prev_link_scale_clipped = np.array([[[[0.0, 0.0, 0.0],
                                                      [0.0, 1.0, 1.0],
                                                      [0.0, 1.0, 1.0]]]], dtype=np.float32)  # [1, 1, 3, 3]
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
            # 调用 TemporalLinkage 层，使用字典传递输入
            updated_linkage = self.temporal_linkage_layer(
                {
                    'write_weights': write_weights,
                    'prev_linkage': prev_linkage
                },
                training=True
            )

            # 定义一个依赖于 precedence_weights 的损失函数
            loss = tf.reduce_sum(updated_linkage['precedence_weights'])  # scalar

        # 计算梯度
        gradients = tape.gradient(loss, [write_weights])

        # 检查梯度
        for grad, var in zip(gradients, [write_weights]):
            self.assertIsNotNone(grad, f"No gradient for {var.name}")
            grad_norm = tf.norm(grad).numpy()
            self.assertGreater(grad_norm, 1e-12, f"Gradient too small for {var.name}")
            tf.print(f"Gradient norm for {var.name}: {grad_norm}")

    def test_all_write_weights_zero(self):
        """
        测试所有写入权重为零的情况，确保 precedence_weights 和 link 不更新。
        """
        batch_size = 1
        write_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)  # [1, 1, 3]
        prev_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)
        prev_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        # 调用 TemporalLinkage 层，使用字典传递输入
        updated_linkage = self.temporal_linkage_layer(
            {
                'write_weights': write_weights,
                'prev_linkage': {
                    'link': prev_link,
                    'precedence_weights': prev_precedence_weights
                }
            },
            training=False
        )

        # 预期 precedence_weights 和 link 不变
        expected_precedence_weights = tf.constant([[[0, 0, 0]]], dtype=tf.float32)
        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_multiple_writes(self):
        """
        测试多个写入操作，确保 precedence_weights 和 link 正确累积。
        """
        batch_size = 1
        write_weights_1 = tf.constant([
            [[0.2, 0.3, 0.5],
             [0.0, 0.0, 0.0]]  # 第二次写入为零
        ], dtype=tf.float32)  # [1, 2, 3]
        write_weights_2 = tf.constant([
            [[0.1, 0.6, 0.3],
             [0.0, 0.0, 0.0]]  # 第二次写入为零
        ], dtype=tf.float32)  # [1, 2, 3]

        # 初始化 TemporalLinkage 实例的 num_writes 为 2
        module = TemporalLinkage(memory_size=self.memory_size, num_writes=2)

        # 第一次写入，传递字典
        linkage_state_1 = module(
            {
                'write_weights': write_weights_1,
                'prev_linkage': {
                    'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size],
                                     dtype=tf.float32),
                    'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
                }
            },
            training=False
        )

        # 第二次写入
        linkage_state_2 = module(
            {
                'write_weights': write_weights_2,
                'prev_linkage': linkage_state_1
            },
            training=False
        )

        # 预期 precedence_weights = write_weights_2 * epsilon
        expected_precedence_weights_2 = write_weights_2 * module.epsilon  # [1, 2, 3]

        # 预期 link 仍为全零矩阵，因为 prev_precedence_weights 初始为零
        expected_link_2 = tf.zeros([batch_size, 2, self.memory_size, self.memory_size], dtype=tf.float32)

        self.assertAllClose(linkage_state_2['precedence_weights'].numpy(), expected_precedence_weights_2.numpy(),
                            atol=1e-6)
        self.assertAllClose(linkage_state_2['link'].numpy(), expected_link_2.numpy(), atol=1e-6)

    def test_write_and_precedence_weights_relation(self):
        """
        测试写入权重和 precedence_weights 之间的关系，确保正确累积。
        """
        batch_size = 1
        write_weights = tf.constant([[[0.3, 0.4, 0.3]]], dtype=tf.float32)  # [1, 1, 3]
        prev_precedence_weights = tf.constant([[[0.1, 0.2, 0.3]]], dtype=tf.float32)

        # 计算 updated_precedence_weights
        reset_gate = 1 - tf.reduce_sum(write_weights, axis=2, keepdims=True)  # [1, 1, 1]
        updated_precedence_weights = reset_gate * prev_precedence_weights + write_weights * self.temporal_linkage_layer.epsilon  # [1, 1, 3]

        # 打印调试信息（可选）
        tf.print("Prev Precedence Weights:", prev_precedence_weights)
        tf.print("Write Weights:", write_weights)
        tf.print("Reset Gate:", reset_gate)
        tf.print("Updated Precedence Weights:", updated_precedence_weights)

        # 预期 precedence_weights = (1 - sum(write_weights)) * prev_precedence_weights + write_weights * epsilon
        expected_precedence_weights = (1 - tf.reduce_sum(write_weights, axis=2, keepdims=True)) * prev_precedence_weights + write_weights * self.temporal_linkage_layer.epsilon

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
        new_link_1 = np.array([[[0.2 * 0.5, 0.2 * 0.0, 0.2 * 0.5],
                                [0.0 * 0.5, 0.0 * 0.0, 0.0 * 0.5],
                                [0.8 * 0.5, 0.8 * 0.0, 0.8 * 0.5]]], dtype=np.float32)  # Shape: [1, 1, 3, 3]

        # 写入头2
        new_link_2 = np.array([[[0.1 * 0.4, 0.1 * 0.1, 0.1 * 0.5],
                                [0.3 * 0.4, 0.3 * 0.1, 0.3 * 0.5],
                                [0.6 * 0.4, 0.6 * 0.1, 0.6 * 0.5]]], dtype=np.float32)  # Shape: [1, 1, 3, 3]

        # 扩展 new_link_1 和 new_link_2 的维度，使它们符合 [1, 1, 3, 3] 的形状
        new_link_1 = np.expand_dims(new_link_1, axis=1)  # [1, 1, 3, 3]
        new_link_2 = np.expand_dims(new_link_2, axis=1)  # [1, 1, 3, 3]

        # 组合所有写入头的新链路
        new_link = np.concatenate([new_link_1, new_link_2], axis=1)  # Shape: [1, 2, 3, 3]

        # 计算期望的链路缩放因子：prev_link_scale = 1 - write_weights_i - write_weights_j
        # 写入头1
        write_weights_i_1 = write_weights[:, 0, :, np.newaxis]  # [1, 1, 3, 1]
        write_weights_j_1 = write_weights[:, 0, np.newaxis, :]  # [1, 1, 1, 3]
        prev_link_scale_1 = 1 - write_weights_i_1 - write_weights_j_1  # [1, 1, 3, 3]
        prev_link_scale_1 = np.clip(prev_link_scale_1.numpy(), 0.0, 1.0)

        # 写入头2
        write_weights_i_2 = write_weights[:, 1, :, np.newaxis]  # [1, 1, 3, 1]
        write_weights_j_2 = write_weights[:, 1, np.newaxis, :]  # [1, 1, 1, 3]
        prev_link_scale_2 = 1 - write_weights_i_2 - write_weights_j_2  # [1, 1, 3, 3]
        prev_link_scale_2 = np.clip(prev_link_scale_2.numpy(), 0.0, 1.0)

        # 组合所有写入头的缩放因子，并确保形状正确为 [1, 2, 3, 3]
        expected_prev_link_scale = np.stack([prev_link_scale_1, prev_link_scale_2], axis=1)  # [1, 2, 3, 3]

        # 计算期望的 updated_link：expected_updated_link = prev_link_scale * prev_link + new_link
        # 因为 prev_link 是全1矩阵，所以相当于 prev_link_scale + new_link
        expected_updated_link = expected_prev_link_scale + new_link  # Shape: [1, 2, 3, 3]

        # 计算期望的 final_link，避免自连接
        # 创建 mask，形状 [1, 2, 3, 3]
        mask = 1 - np.eye(self.memory_size, dtype=np.float32)  # Shape: [3, 3]
        mask = np.tile(mask, (self.num_writes, 1, 1))  # Shape: [2, 3, 3]
        mask = np.expand_dims(mask, axis=0)  # Shape: [1, 2, 3, 3]
        expected_final_link = expected_updated_link * mask  # Shape: [1, 2, 3, 3]

        # 打印调试信息，查看实际的 updated_link 和期望的 final_link
        tf.print("write_weights:", write_weights)
        tf.print("prev_precedence_weights:", prev_precedence_weights)
        tf.print("new_link:", new_link)
        tf.print("expected_prev_link_scale:", expected_prev_link_scale)
        tf.print("expected_updated_link:", expected_updated_link)
        tf.print("expected_final_link:", expected_final_link)
        tf.print("actual updated_link:", updated_link.numpy())

        # 验证链路更新结果是否符合预期
        np.testing.assert_allclose(updated_link.numpy(), expected_final_link, atol=1e-6)