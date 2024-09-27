import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import weighted_softmax, WriteAllocation, UsageUpdate


class UsageUpdateTest(tf.test.TestCase):
    def setUp(self):
        super(UsageUpdateTest, self).setUp()
        self.memory_size = 3  # 定义 memory_size
        self.num_writes = 2  # 定义 num_writes
        self.num_reads = 2  # 定义 num_reads
        self.epsilon = 1e-6  # 定义 epsilon

        # 初始化 UsageUpdate 层
        self.usage_update_layer = UsageUpdate(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            num_reads=self.num_reads,
            epsilon=self.epsilon
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

        # 调用 UsageUpdate 层
        updated_usage = self.usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [2, 3]

        # 计算预期使用率
        # 1. 计算 write_allocation = 1 - prod(1 - write_weights, axis=1)
        write_weights_cumprod = tf.reduce_prod(1 - write_weights, axis=1)  # [2, 3]
        write_allocation = 1 - write_weights_cumprod  # [2, 3]

        # 2. 使用 usage_after_write = initial_usage + (1 - initial_usage) * write_allocation
        usage_after_write = initial_usage + (1 - initial_usage) * write_allocation  # [2, 3]

        # 3. 计算自由读权重
        free_gate_expanded = tf.expand_dims(free_gate, axis=-1)  # [2, 2, 1]
        free_read_weights = free_gate_expanded * read_weights  # [2, 2, 3]

        # 4. 计算 total_free_read_weights = sum(free_read_weights, axis=1)
        total_free_read_weights = tf.reduce_sum(free_read_weights, axis=1)  # [2, 3]

        # 5. 使用 usage_after_read = usage_after_write - total_free_read_weights
        usage_after_read = usage_after_write - total_free_read_weights  # [2, 3]

        # 6. 裁剪使用率到 [0, 1]
        expected_usage = tf.clip_by_value(usage_after_read, 0.0, 1.0)  # [2, 3]

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

        # 调用 UsageUpdate 层
        updated_usage = self.usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 计算预期使用率
        # 1. 计算 write_allocation = 1 - prod(1 - write_weights, axis=1)
        write_weights_cumprod = tf.reduce_prod(1 - write_weights, axis=1)  # [1, 3]
        write_allocation = 1 - write_weights_cumprod  # [1, 3]

        # 2. 使用 usage_after_write = initial_usage + (1 - initial_usage) * write_allocation
        usage_after_write = initial_usage + (1 - initial_usage) * write_allocation  # [1, 3]
        # 由于 initial_usage =1, usage_after_write =1 +0 * write_allocation =1

        # 3. 计算自由读权重
        free_gate_expanded = tf.expand_dims(free_gate, axis=-1)  # [1, 1, 1]
        free_read_weights = free_gate_expanded * read_weights  # [1, 1, 3]

        # 4. 计算 total_free_read_weights = sum(free_read_weights, axis=1)
        total_free_read_weights = tf.reduce_sum(free_read_weights, axis=1)  # [1, 3]

        # 5. 使用 usage_after_read = usage_after_write - total_free_read_weights
        usage_after_read = usage_after_write - total_free_read_weights  # [1, 3]
        # 1 -1 =0, 1 -1 =0, 1 -1 =0

        # 6. 裁剪使用率到 [0, 1]
        expected_usage = tf.clip_by_value(usage_after_read, 0.0, 1.0)  # [1, 3]

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

        # 调用 UsageUpdate 层
        updated_usage = self.usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 计算预期使用率
        # 1. 计算 write_allocation = 1 - prod(1 - write_weights, axis=1)
        write_weights_cumprod = tf.reduce_prod(1 - write_weights, axis=1)  # [1, 3]
        write_allocation = 1 - write_weights_cumprod  # [1, 3]

        # 2. 使用 usage_after_write = initial_usage + (1 - initial_usage) * write_allocation
        usage_after_write = initial_usage + (1 - initial_usage) * write_allocation  # [1, 3]
        # =0 +1 * write_allocation =write_allocation

        # 3. 计算自由读权重
        free_gate_expanded = tf.expand_dims(free_gate, axis=-1)  # [1, 1, 1]
        free_read_weights = free_gate_expanded * read_weights  # [1, 1, 3]
        # =0 * read_weights =0

        # 4. 计算 total_free_read_weights = sum(free_read_weights, axis=1)
        total_free_read_weights = tf.reduce_sum(free_read_weights, axis=1)  # [1, 3]
        # =0

        # 5. 使用 usage_after_read = usage_after_write - total_free_read_weights
        usage_after_read = usage_after_write - total_free_read_weights  # [1, 3]
        # = write_allocation -0 = write_allocation

        # 6. 裁剪使用率到 [0, 1]
        expected_usage = tf.clip_by_value(usage_after_read, 0.0, 1.0)  # [1, 3]

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

        # 调用 UsageUpdate 层
        updated_usage = self.usage_update_layer({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': initial_usage
        }, training=False)  # [1, 3]

        # 计算预期使用率
        # 步骤1: 计算 write_allocation = 1 - prod(1 - write_weights, axis=1)
        write_weights_cumprod = tf.reduce_prod(1 - write_weights, axis=1)  # [1, 3]
        write_allocation = 1 - write_weights_cumprod  # [1, 3]

        # 步骤2: 计算 usage_after_write = initial_usage + (1 - initial_usage) * write_allocation
        usage_after_write = initial_usage + (1 - initial_usage) * write_allocation  # [1, 3]

        # 步骤3: 计算 total_free_read_weights = sum(free_gate * read_weights, axis=1)
        free_gate_expanded = tf.expand_dims(free_gate, axis=-1)  # [1, 2, 1]
        free_read_weights = free_gate_expanded * read_weights  # [1, 2, 3]
        total_free_read_weights = tf.reduce_sum(free_read_weights, axis=1)  # [1, 3]

        # 步骤4: 计算 usage_after_read = usage_after_write - total_free_read_weights
        usage_after_read = usage_after_write - total_free_read_weights  # [1, 3]
        usage_after_read = tf.maximum(usage_after_read, 0.0)  # 确保不低于0

        # 步骤5: 裁剪使用率到 [0, 1]
        expected_usage = tf.clip_by_value(usage_after_read, 0.0, 1.0)  # [1, 3]

        # 转换为 numpy 进行比较
        expected_usage_np = expected_usage.numpy()
        updated_usage_np = updated_usage.numpy()

        # 打印调试信息（可选）
        tf.print("Initial Usage:", initial_usage)
        tf.print("Write Weights:", write_weights)
        tf.print("Write Allocation:", write_allocation)
        tf.print("Usage After Write:", usage_after_write)
        tf.print("Free Gate:", free_gate)
        tf.print("Read Weights:", read_weights)
        tf.print("Free Read Weights:", free_read_weights)
        tf.print("Total Free Read Weights:", total_free_read_weights)
        tf.print("Usage After Read:", usage_after_read)
        tf.print("Expected Usage:", expected_usage)
        tf.print("Updated Usage:", updated_usage)

        # 断言更新后的使用率与预期值接近
        self.assertAllClose(updated_usage_np, expected_usage_np, atol=1e-6)

    def test_initial_state(self):
        """
        测试 get_initial_state 方法，确保返回正确的初始使用率。
        """
        batch_size = 4
        initial_usage = self.usage_update_layer.get_initial_state([batch_size])  # [4,3]

        expected_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [4,3]

        self.assertAllClose(initial_usage.numpy(), expected_usage.numpy(), atol=1e-6)

    def test_state_size(self):
        """
        测试 state_size 属性，确保返回正确的形状。
        """
        self.assertEqual(self.usage_update_layer.state_size, tf.TensorShape([self.memory_size]))


