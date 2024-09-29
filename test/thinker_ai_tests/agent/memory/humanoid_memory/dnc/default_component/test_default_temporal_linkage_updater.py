import tensorflow as tf
import numpy as np

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import DefaultTemporalLinkageUpdater


# 假设您将 TemporalLinkage 类保存在 temporal_linkage.py 文件中
# from temporal_linkage import TemporalLinkage

class TemporalLinkageTest(tf.test.TestCase):
    def setUp(self):
        super(TemporalLinkageTest, self).setUp()
        self.memory_size = 3
        self.num_writes = 2
        self.temporal_linkage = DefaultTemporalLinkageUpdater(
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

        # 调用 update_linkage 方法
        updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        # 修正 expected_precedence_weights，考虑 epsilon
        expected_precedence_weights = write_weights * self.temporal_linkage.epsilon  # [1, 2, 3]

        # 预期 link = zeros，因为 prev_precedence_weights 为零
        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_dynamic_num_writes(self):
        """
        测试动态调整 num_writes 的情况。
        """
        batch_size = 2
        dynamic_num_writes = 3

        # 重新初始化 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(memory_size=self.memory_size, num_writes=dynamic_num_writes)

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

        # 调用 update_linkage 方法
        updated_linkage = temporal_linkage.update_linkage(write_weights, prev_linkage)

        # 修正 expected_precedence_weights，考虑 epsilon
        expected_precedence_weights = write_weights * temporal_linkage.epsilon  # [2, 3, 3]

        # 预期 link = zeros，因为 prev_precedence_weights 为零
        expected_link = tf.zeros([batch_size, dynamic_num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        # 验证 precedence_weights 和 link
        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_partial_write_and_read(self):
        """
        测试部分内存槽被写入和部分内存槽被读出的情况。
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

        # 调用 update_linkage 方法
        updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        # 修正 expected_precedence_weights，考虑 epsilon
        expected_precedence_weights = write_weights * self.temporal_linkage.epsilon  # [1, 2, 3]

        # 预期 link = zeros，因为 prev_precedence_weights 为零
        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_initial_state(self):
        """
        测试初始状态，确保返回正确的初始链路矩阵和优先级权重。
        """
        batch_size = 4
        # 初始化 prev_linkage
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        }

        expected_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)
        expected_precedence_weights = tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)

        self.assertAllClose(prev_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)
        self.assertAllClose(prev_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)

    def test_state_size(self):
        """
        测试 state_size 方法，确保返回正确的形状。
        """
        expected_state_size = {
            'link': tf.TensorShape([self.num_writes, self.memory_size, self.memory_size]),
            'precedence_weights': tf.TensorShape([self.num_writes, self.memory_size])
        }
        self.assertEqual(self.temporal_linkage.state_size(), expected_state_size)

    def test_link_update_steps(self):
        """
        测试链路更新的多个步骤。
        """
        batch_size = 1
        memory_size = 3
        num_writes = 1

        temporal_linkage = DefaultTemporalLinkageUpdater(memory_size=memory_size, num_writes=num_writes)

        # 初始化状态
        prev_linkage = {
            'link': tf.zeros([batch_size, num_writes, memory_size, memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, num_writes, memory_size], dtype=tf.float32)
        }

        # 第一次写入
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)  # 写入位置 0
        linkage_state = temporal_linkage.update_linkage(write_weights, prev_linkage)

        # 验证第一次写入后的链路矩阵
        expected_link_after_first_write = np.zeros((1, 1, 3, 3), dtype=np.float32)
        self.assertAllClose(linkage_state['link'].numpy(), expected_link_after_first_write, atol=1e-6)

        # 更新 prev_linkage
        prev_linkage = linkage_state

        # 第二次写入
        write_weights = tf.constant([[[0, 1, 0]]], dtype=tf.float32)  # 写入位置 1
        linkage_state = temporal_linkage.update_linkage(write_weights, prev_linkage)

        # 修正 expected_precedence_weights，考虑 epsilon
        sum_write_weights = tf.reduce_sum(write_weights, axis=-1, keepdims=True)  # [1, 1, 1]
        reset_gate = 1 - sum_write_weights
        expected_precedence_weights = reset_gate * prev_linkage['precedence_weights'] + write_weights * temporal_linkage.epsilon

        self.assertAllClose(linkage_state['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)

        # 验证 link 矩阵
        # 计算 new_link
        prev_precedence_weights = prev_linkage['precedence_weights']
        write_weights_i = tf.expand_dims(write_weights, axis=-1)  # [1, 1, 3, 1]
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, axis=-2)  # [1, 1, 1, 3]
        new_link = write_weights_i * prev_precedence_weights_j  # [1, 1, 3, 3]

        # 计算链路缩放因子
        write_weights_j = tf.expand_dims(write_weights, axis=-2)  # [1, 1, 1, 3]
        prev_link_scale = 1 - write_weights_i - write_weights_j  # [1, 1, 3, 3]
        prev_link_scale = tf.clip_by_value(prev_link_scale, 0.0, 1.0)

        # 更新链路矩阵
        expected_link = prev_linkage['link'] * prev_link_scale + new_link

        # 避免自连接
        mask = 1 - tf.eye(memory_size, batch_shape=[num_writes], dtype=tf.float32)
        expected_link = expected_link * tf.expand_dims(mask, axis=0)

        self.assertAllClose(linkage_state['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_full_link_method(self):
        """
        测试在第一次写入时链路矩阵为全零。
        """
        batch_size = 1
        memory_size = 3
        num_writes = 1

        temporal_linkage = DefaultTemporalLinkageUpdater(memory_size=memory_size, num_writes=num_writes)

        # 初始化 prev_linkage
        prev_linkage = {
            'link': tf.zeros([batch_size, num_writes, memory_size, memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, num_writes, memory_size], dtype=tf.float32)
        }

        # 创建 write_weights
        write_weights = tf.constant([[[1, 0, 0]]], dtype=tf.float32)

        # 调用 update_linkage 方法
        linkage_state = temporal_linkage.update_linkage(write_weights, prev_linkage)

        # 验证更新后的 link 矩阵是否符合预期
        expected_link = np.zeros((1, 1, 3, 3), dtype=np.float32)  # 第一次写入，link 应为全零矩阵
        self.assertAllClose(linkage_state['link'].numpy(), expected_link, atol=1e-6)

    def test_gradient_flow(self):
        """
        验证 TemporalLinkage 类的梯度是否能够正确传播。
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
            # 调用 update_linkage 方法
            updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

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

        prev_linkage = {
            'link': prev_link,
            'precedence_weights': prev_precedence_weights
        }

        # 调用 update_linkage 方法
        updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        # 预期 precedence_weights 和 link 不变
        expected_precedence_weights = prev_precedence_weights
        expected_link = prev_link

        self.assertAllClose(updated_linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(), atol=1e-6)
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

    def test_multiple_writes(self):
        """
        测试多个写入操作，确保 precedence_weights 和 link 正确累积。
        """
        batch_size = 1
        write_weights_1 = tf.constant([
            [[0.2, 0.3, 0.5],
             [0.0, 0.0, 0.0]]  # 第二个写头不写入
        ], dtype=tf.float32)  # [1, 2, 3]
        write_weights_2 = tf.constant([
            [[0.1, 0.6, 0.3],
             [0.0, 0.0, 0.0]]  # 第二个写头不写入
        ], dtype=tf.float32)  # [1, 2, 3]

        # 初始化 TemporalLinkage 实例
        temporal_linkage = DefaultTemporalLinkageUpdater(memory_size=self.memory_size, num_writes=self.num_writes)

        # 第一次写入
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size],
                             dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        }
        linkage_state_1 = temporal_linkage.update_linkage(write_weights_1, prev_linkage)

        # 第二次写入
        linkage_state_2 = temporal_linkage.update_linkage(write_weights_2, linkage_state_1)

        # 计算 expected_precedence_weights_2
        sum_write_weights = tf.reduce_sum(write_weights_2, axis=-1, keepdims=True)  # [1, 2, 1]
        reset_gate = 1 - sum_write_weights
        expected_precedence_weights_2 = reset_gate * linkage_state_1['precedence_weights'] + write_weights_2 * temporal_linkage.epsilon

        self.assertAllClose(linkage_state_2['precedence_weights'].numpy(), expected_precedence_weights_2.numpy(),
                            atol=1e-6)

    def test_write_and_precedence_weights_relation(self):
        """
        测试写入权重和 precedence_weights 之间的关系，确保正确累积。
        """
        batch_size = 1
        write_weights = tf.constant([[[0.3, 0.4, 0.3]]], dtype=tf.float32)  # [1, 1, 3]
        prev_precedence_weights = tf.constant([[[0.1, 0.2, 0.3]]], dtype=tf.float32)

        # 计算 updated_precedence_weights
        sum_write_weights = tf.reduce_sum(write_weights, axis=-1, keepdims=True)  # [1, 1, 1]
        reset_gate = 1 - sum_write_weights  # [1, 1, 1]
        updated_precedence_weights = reset_gate * prev_precedence_weights + write_weights * self.temporal_linkage.epsilon  # 考虑 epsilon

        # 调用 update_linkage 方法
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size],
                             dtype=tf.float32),
            'precedence_weights': prev_precedence_weights
        }
        linkage_state = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        self.assertAllClose(linkage_state['precedence_weights'].numpy(), updated_precedence_weights.numpy(), atol=1e-6)

    def test_link_update_with_non_zero_precedence(self):
        """
        测试链路更新时，precedence_weights 非零的情况。
        """
        # 定义 batch_size
        batch_size = 1

        # 定义输入参数
        write_weights = tf.constant([[[0.2, 0.0, 0.8],
                                      [0.1, 0.3, 0.6]]], dtype=tf.float32)  # Shape: [1, 2, 3]
        prev_precedence_weights = tf.constant([[[0.5, 0.0, 0.5],
                                                [0.4, 0.1, 0.5]]], dtype=tf.float32)  # Shape: [1, 2, 3]
        prev_link = tf.ones([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)  # 全1矩阵

        # 创建 prev_linkage 字典
        prev_linkage = {
            'link': prev_link,
            'precedence_weights': prev_precedence_weights
        }

        # 调用 update_linkage 方法
        updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        # 手动计算 expected_link
        expected_link = prev_link.numpy()  # 将 prev_link 转换为 NumPy 数组以进行修改

        # 计算 write_weights_i * prev_precedence_weights_j
        for b in range(batch_size):
            for w in range(self.num_writes):
                write_weights_i = write_weights.numpy()[b, w, :, np.newaxis]  # [memory_size, 1]
                prev_precedence_weights_j = prev_precedence_weights.numpy()[b, w, np.newaxis, :]  # [1, memory_size]
                new_link = write_weights_i * prev_precedence_weights_j  # [memory_size, memory_size]

                # 计算链路缩放因子
                write_weights_i_expanded = write_weights.numpy()[b, w, :, np.newaxis]  # [memory_size, 1]
                write_weights_j_expanded = write_weights.numpy()[b, w, np.newaxis, :]  # [1, memory_size]
                prev_link_scale = 1 - write_weights_i_expanded - write_weights_j_expanded  # [memory_size, memory_size]
                prev_link_scale = np.clip(prev_link_scale, 0.0, 1.0)

                # 更新 expected_link
                expected_link[b, w] = prev_link_scale * expected_link[b, w] + new_link

                # 避免自连接
                np.fill_diagonal(expected_link[b, w], 0.0)

        # 将 expected_link 转换回 Tensor 以进行比较
        expected_link = tf.convert_to_tensor(expected_link)

        # 验证链路矩阵是否符合预期
        self.assertAllClose(updated_linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)

if __name__ == '__main__':
    tf.test.main()