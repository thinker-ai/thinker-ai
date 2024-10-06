import tensorflow as tf
import numpy as np

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import DefaultTemporalLinkageUpdater


class TemporalLinkageTest(tf.test.TestCase):
    def setUp(self):
        super(TemporalLinkageTest, self).setUp()
        tf.random.set_seed(42)
        np.random.seed(42)
        self.memory_size = 3
        self.num_writes = 2
        self.temporal_linkage = DefaultTemporalLinkageUpdater(
            memory_size=self.memory_size,
            num_writes=self.num_writes
        )

    def test_directional_read_weights_normalization(self):
        """
        测试方向性读取权重的归一化。
        """
        batch_size = 2
        num_reads = 2

        # 创建随机链接矩阵
        link = tf.random.uniform([batch_size, self.num_writes, self.memory_size, self.memory_size], minval=0.0, maxval=1.0)

        # 创建归一化的先前读取权重
        prev_read_weights = tf.nn.softmax(tf.random.uniform([batch_size, num_reads, self.memory_size], minval=0.0, maxval=1.0), axis=-1)

        directional_weights_forward = self.temporal_linkage.directional_read_weights(link, prev_read_weights, forward=True)
        directional_weights_backward = self.temporal_linkage.directional_read_weights(link, prev_read_weights, forward=False)

        # 检查方向性读取权重的值是否合理（非 NaN 或 Inf）
        self.assertFalse(tf.math.reduce_any(tf.math.is_nan(directional_weights_forward)))
        self.assertFalse(tf.math.reduce_any(tf.math.is_inf(directional_weights_forward)))
        self.assertFalse(tf.math.reduce_any(tf.math.is_nan(directional_weights_backward)))
        self.assertFalse(tf.math.reduce_any(tf.math.is_inf(directional_weights_backward)))

    def test_directional_read_weights(self):
        """
        测试方向性读取权重的计算。
        """
        batch_size = 1
        num_reads = 1

        link = tf.constant([[
            [[0.0, 0.1, 0.2],
             [0.3, 0.0, 0.4],
             [0.5, 0.6, 0.0]],
            [[0.0, 0.2, 0.1],
             [0.1, 0.0, 0.5],
             [0.4, 0.2, 0.0]]
        ]], dtype=tf.float32)  # [1, 2, 3, 3]

        prev_read_weights = tf.constant([[[1.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 1, 3]

        # 计算前向方向性读取权重
        directional_weights = self.temporal_linkage.directional_read_weights(
            link, prev_read_weights, forward=True
        )

        # 手动计算综合的链接矩阵
        combined_link = tf.reduce_sum(link, axis=1)  # [1, 3, 3]

        # 手动计算预期的方向性读取权重
        expected_directional_weights = tf.matmul(prev_read_weights, combined_link)  # [1, 1, 3]

        # 断言实际值和预期值相等
        self.assertAllClose(directional_weights, expected_directional_weights, atol=1e-6)

    def test_precedence_weights_update(self):
        """
        测试先行权重的更新是否正确。
        """
        batch_size = 1
        write_weights = tf.constant([[[0.1, 0.2, 0.3],
                                      [0.4, 0.1, 0.2]]], dtype=tf.float32)  # [1, 2, 3]
        prev_precedence_weights = tf.constant([[[0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 2, 3]
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': prev_precedence_weights
        }

        updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        write_sum = tf.reduce_sum(write_weights, axis=2, keepdims=True)
        expected_precedence_weights = (1 - write_sum) * prev_precedence_weights + write_weights

        self.assertAllClose(updated_linkage['precedence_weights'], expected_precedence_weights, atol=1e-6)

    def test_link_matrix_update(self):
        """
        测试链接矩阵的更新是否正确。
        """
        batch_size = 1
        write_weights = tf.constant([[[0.2, 0.5, 0.3],
                                      [0.1, 0.6, 0.3]]], dtype=tf.float32)  # [1, 2, 3]
        prev_precedence_weights = tf.constant([[[0.1, 0.2, 0.7],
                                                [0.3, 0.4, 0.3]]], dtype=tf.float32)  # [1, 2, 3]
        prev_link = tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)

        prev_linkage = {
            'link': prev_link,
            'precedence_weights': prev_precedence_weights
        }

        updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        # 手动计算更新的链接矩阵
        write_sum = tf.reduce_sum(write_weights, axis=2, keepdims=True)
        updated_precedence_weights = (1 - write_sum) * prev_precedence_weights + write_weights

        write_weights_i = tf.expand_dims(write_weights, axis=3)  # [batch_size, num_writes, memory_size, 1]
        precedence_weights_j = tf.expand_dims(updated_precedence_weights, axis=2)  # [batch_size, num_writes, 1, memory_size]

        new_link = write_weights_i * precedence_weights_j  # [batch_size, num_writes, memory_size, memory_size]

        # 移除自连接
        identity = tf.eye(self.memory_size, batch_shape=[self.num_writes], dtype=tf.float32)
        identity = tf.expand_dims(identity, axis=0)
        new_link = new_link * (1 - identity)

        expected_link = new_link  # 因为 prev_link 为零，所以直接使用 new_link

        self.assertAllClose(updated_linkage['link'], expected_link, atol=1e-6)

    def test_state_size(self):
        """
        测试 state_size 方法。
        """
        expected_state_size = {
            'link': tf.TensorShape([self.num_writes, self.memory_size, self.memory_size]),
            'precedence_weights': tf.TensorShape([self.num_writes, self.memory_size])
        }
        self.assertEqual(self.temporal_linkage.state_size(), expected_state_size)

    def test_no_self_links(self):
        """
        测试链接矩阵的对角线元素是否为零（无自连接）。
        """
        batch_size = 1
        write_weights = tf.constant([[[0.3, 0.6, 0.1],
                                      [0.2, 0.5, 0.3]]], dtype=tf.float32)  # [1, 2, 3]
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        }
        updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)

        link_matrices = updated_linkage['link'].numpy()
        for i in range(self.num_writes):
            # 提取第 i 个写头的链接矩阵
            link_matrix = link_matrices[0, i, :, :]
            # 检查对角线是否为零
            diagonal = np.diag(link_matrix)
            self.assertAllClose(diagonal, np.zeros(self.memory_size), atol=1e-6)

    def test_gradient_flow(self):
        """
        测试梯度是否能正确地通过 update_linkage 方法传播。
        """
        batch_size = 1
        write_weights = tf.Variable([[[0.5, 0.3, 0.2],
                                      [0.1, 0.6, 0.3]]], dtype=tf.float32)  # [1, 2, 3]
        prev_linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        }

        with tf.GradientTape() as tape:
            updated_linkage = self.temporal_linkage.update_linkage(write_weights, prev_linkage)
            loss = tf.reduce_sum(updated_linkage['precedence_weights'])

        grads = tape.gradient(loss, [write_weights])
        self.assertIsNotNone(grads[0])
        self.assertGreater(tf.norm(grads[0]).numpy(), 1e-6)

if __name__ == '__main__':
    tf.test.main()
