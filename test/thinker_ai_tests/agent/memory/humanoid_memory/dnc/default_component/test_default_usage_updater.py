import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import DefaultUsageUpdater


class UsageUpdateTest(tf.test.TestCase):
    def setUp(self):
        super(UsageUpdateTest, self).setUp()
        self.memory_size = 3  # 定义 memory_size
        self.num_writes = 2  # 定义 num_writes
        self.num_reads = 2  # 定义 num_reads
        self.epsilon = 1e-6  # 定义 epsilon

        # 初始化 UsageUpdate 实例
        self.usage_update = DefaultUsageUpdater(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            num_reads=self.num_reads
        )

    def test_basic_usage_update(self):
        """
        基本测试：验证写操作和读操作对使用率的影响。
        """
        batch_size = 2

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [2, 3]

        # 定义 write_weights
        write_weights = tf.constant([
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]],
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]]
        ], dtype=tf.float32)  # [2, 2, 3]

        # 定义 free_gate 和 read_weights
        free_gate = tf.constant([
            [1.0, 0.0],
            [1.0, 1.0]
        ], dtype=tf.float32)  # [2, 2]

        read_weights = tf.constant([
            [[0.5, 0.5, 0.0],
             [0.0, 0.5, 0.5]],
            [[1.0, 1.0, 1.0],
             [0.0, 1.0, 0.0]]
        ], dtype=tf.float32)  # [2, 2, 3]

        # 调用 update_usage 方法
        updated_usage = self.usage_update.update_usage(
            write_weights=write_weights,
            free_gate=free_gate,
            read_weights=read_weights,
            prev_usage=initial_usage
        )  # [2, 3]

        # 计算预期使用率
        sum_w_write = tf.reduce_sum(write_weights, axis=1)  # [2, 3]
        sum_w_read = tf.reduce_sum(free_gate[:, :, tf.newaxis] * read_weights, axis=1)  # [2, 3]
        usage = initial_usage + sum_w_write - initial_usage * sum_w_write - sum_w_read  # [2, 3]
        expected_usage = tf.clip_by_value(usage, 0.0, 1.0)  # [2, 3]

        # 断言
        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_all_full_usage(self):
        """
        测试所有内存槽已满的情况，确保进一步写操作释放使用率。
        """
        batch_size = 1

        # 创建初始使用率为全1
        initial_usage = tf.ones([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义 write_weights
        write_weights = tf.constant([
            [[1 / 3, 1 / 3, 1 / 3]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 定义 free_gate 和 read_weights（全读）
        free_gate = tf.constant([
            [1.0]
        ], dtype=tf.float32)  # [1, 1]

        read_weights = tf.constant([
            [[1.0, 1.0, 1.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 调用 update_usage 方法
        updated_usage = self.usage_update.update_usage(
            write_weights=write_weights,
            free_gate=free_gate,
            read_weights=read_weights,
            prev_usage=initial_usage
        )  # [1, 3]

        # 计算预期使用率
        sum_w_write = tf.reduce_sum(write_weights, axis=1)  # [1, 3]
        sum_w_read = tf.reduce_sum(free_gate[:, :, tf.newaxis] * read_weights, axis=1)  # [1, 3]
        usage = initial_usage + sum_w_write - initial_usage * sum_w_write - sum_w_read  # [1, 3]
        expected_usage = tf.clip_by_value(usage, 0.0, 1.0)  # [1, 3]

        # 断言
        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_all_zero_read_weights(self):
        """
        测试所有读权重为零的情况，确保读操作不影响使用率。
        """
        batch_size = 1

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [1, 3]

        # 定义 write_weights
        write_weights = tf.constant([
            [[1 / 3, 1 / 3, 1 / 3]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 定义 free_gate 和 read_weights（全零）
        free_gate = tf.constant([
            [0.0]
        ], dtype=tf.float32)  # [1, 1]

        read_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 调用 update_usage 方法
        updated_usage = self.usage_update.update_usage(
            write_weights=write_weights,
            free_gate=free_gate,
            read_weights=read_weights,
            prev_usage=initial_usage
        )  # [1, 3]

        # 计算预期使用率
        sum_w_write = tf.reduce_sum(write_weights, axis=1)  # [1, 3]
        sum_w_read = tf.reduce_sum(free_gate[:, :, tf.newaxis] * read_weights, axis=1)  # [1, 3]
        usage = initial_usage + sum_w_write - initial_usage * sum_w_write - sum_w_read  # [1, 3]
        expected_usage = tf.clip_by_value(usage, 0.0, 1.0)  # [1, 3]

        # 断言
        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

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

        # 调用 update_usage 方法
        updated_usage = self.usage_update.update_usage(
            write_weights=write_weights,
            free_gate=free_gate,
            read_weights=read_weights,
            prev_usage=initial_usage
        )  # [1, 3]

        # 计算预期使用率
        sum_w_write = tf.reduce_sum(write_weights, axis=1)  # [1, 3]
        sum_w_read = tf.reduce_sum(free_gate[:, :, tf.newaxis] * read_weights, axis=1)  # [1, 3]
        usage = initial_usage + sum_w_write - initial_usage * sum_w_write - sum_w_read  # [1, 3]
        expected_usage = tf.clip_by_value(usage, 0.0, 1.0)  # [1, 3]

        # 打印调试信息（可选）
        # tf.print("Initial Usage:", initial_usage)
        # tf.print("Write Weights:", write_weights)
        # tf.print("Sum Write Weights:", sum_w_write)
        # tf.print("Sum Read Weights:", sum_w_read)
        # tf.print("Usage:", usage)
        # tf.print("Expected Usage:", expected_usage)
        # tf.print("Updated Usage:", updated_usage)

        # 断言
        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    # 如果新的 UsageUpdate 类没有 get_initial_state 和 state_size 方法，可以删除以下测试
    # def test_initial_state(self):
    #     """
    #     测试 get_initial_state 方法，确保返回正确的初始使用率。
    #     """
    #     batch_size = 4
    #     initial_usage = self.usage_update.get_initial_state(batch_size=batch_size)  # [4, 3]
    #     expected_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [4, 3]
    #     self.assertAllClose(initial_usage.numpy(), expected_usage.numpy(), atol=1e-6)
    #
    # def test_state_size(self):
    #     """
    #     测试 state_size 属性，确保返回正确的形状。
    #     """
    #     self.assertEqual(self.usage_update.state_size(), tf.TensorShape([self.memory_size]))


if __name__ == '__main__':
    tf.test.main()