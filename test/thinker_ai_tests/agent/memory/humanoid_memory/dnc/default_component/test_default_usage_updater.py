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
        self.usage_update = DefaultUsageUpdater()

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

        # 定义 free_gates
        free_gates = tf.constant([
            [0.1, 0.2]
        ], dtype=tf.float32)  # [1, 2]

        # 定义 prev_read_weights
        prev_read_weights = tf.constant([
            [[0.3, 0.4, 0.3],
             [0.2, 0.5, 0.3]]
        ], dtype=tf.float32)  # [1, 2, 3]

        # 定义 training
        training = False

        # 调用 update_usage 方法
        updated_usage = self.usage_update.update_usage(
            write_weights=write_weights,
            free_gates=free_gates,
            prev_read_weights=prev_read_weights,
            prev_usage=initial_usage,
            training=training
        )  # [1, 3]

        # 计算预期使用率基于新的 update_usage 实现
        write_weights_sum = tf.reduce_sum(write_weights, axis=1)  # [1, 3]

        retention = tf.reduce_prod(1 - tf.expand_dims(free_gates, axis=-1) * prev_read_weights, axis=1)  # [1, 3]

        usage = (initial_usage + write_weights_sum - initial_usage * write_weights_sum) * retention  # [1, 3]

        expected_usage = tf.clip_by_value(usage, 0.0, 1.0)  # [1, 3]

        # 断言
        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)

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

        # 定义 free_gates
        free_gates = tf.constant([
            [0.5, 0.5],
            [0.5, 0.5]
        ], dtype=tf.float32)  # [2, 2]

        # 定义 prev_read_weights
        prev_read_weights = tf.constant([
            [[0.2, 0.3, 0.5],
             [0.1, 0.7, 0.2]],
            [[0.4, 0.4, 0.2],
             [0.3, 0.6, 0.1]]
        ], dtype=tf.float32)  # [2, 2, 3]

        # 定义 training
        training = False

        # 调用 update_usage 方法
        updated_usage = self.usage_update.update_usage(
            write_weights=write_weights,
            free_gates=free_gates,
            prev_read_weights=prev_read_weights,
            prev_usage=initial_usage,
            training=training
        )  # [2, 3]

        # 计算预期使用率基于新的 update_usage 实现
        write_weights_sum = tf.reduce_sum(write_weights, axis=1)  # [2, 3]

        retention = tf.reduce_prod(1 - tf.expand_dims(free_gates, axis=-1) * prev_read_weights, axis=1)  # [2, 3]

        usage = (initial_usage + write_weights_sum - initial_usage * write_weights_sum) * retention  # [2, 3]

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

        # 定义 free_gates
        free_gates = tf.constant([
            [0.3]
        ], dtype=tf.float32)  # [1, 1]

        # 定义 prev_read_weights
        prev_read_weights = tf.constant([
            [[0.4, 0.4, 0.2]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 定义 training
        training = False

        # 调用 update_usage 方法
        updated_usage = self.usage_update.update_usage(
            write_weights=write_weights,
            free_gates=free_gates,
            prev_read_weights=prev_read_weights,
            prev_usage=initial_usage,
            training=training
        )  # [1, 3]

        # 计算预期使用率基于新的 update_usage 实现
        write_weights_sum = tf.reduce_sum(write_weights, axis=1)  # [1, 3]

        retention = tf.reduce_prod(1 - tf.expand_dims(free_gates, axis=-1) * prev_read_weights, axis=1)  # [1, 3]

        usage = (initial_usage + write_weights_sum - initial_usage * write_weights_sum) * retention  # [1, 3]

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

        # 定义 free_gates
        free_gates = tf.constant([
            [0.0]
        ], dtype=tf.float32)  # [1, 1]

        # 定义 prev_read_weights
        prev_read_weights = tf.constant([
            [[0.0, 0.0, 0.0]]
        ], dtype=tf.float32)  # [1, 1, 3]

        # 定义 training
        training = False

        # 调用 update_usage 方法
        updated_usage = self.usage_update.update_usage(
            write_weights=write_weights,
            free_gates=free_gates,
            prev_read_weights=prev_read_weights,
            prev_usage=initial_usage,
            training=training
        )  # [1, 3]

        # 计算预期使用率基于新的 update_usage 实现
        write_weights_sum = tf.reduce_sum(write_weights, axis=1)  # [1, 3]

        retention = tf.reduce_prod(1 - tf.expand_dims(free_gates, axis=-1) * prev_read_weights, axis=1)  # [1, 3]

        usage = (initial_usage + write_weights_sum - initial_usage * write_weights_sum) * retention  # [1, 3]

        expected_usage = tf.clip_by_value(usage, 0.0, 1.0)  # [1, 3]

        # 断言
        self.assertAllClose(updated_usage.numpy(), expected_usage.numpy(), atol=1e-6)


if __name__ == '__main__':
    tf.test.main()
