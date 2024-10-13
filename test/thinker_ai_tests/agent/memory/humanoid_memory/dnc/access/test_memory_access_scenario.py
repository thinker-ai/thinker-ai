from unittest import mock

import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess
from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import get_default_config


class MemoryAccessUserScenarioTest(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessUserScenarioTest, self).setUp()
        self.memory_size = 32
        self.word_size = 16
        self.num_reads = 2
        self.num_writes = 1
        self.batch_size = 2  # 两个用户
        self.controller_output_size = 64  # 固定的控制器输出尺寸

        self.cache_manager_mock = mock.Mock()
        self.cache_manager_mock.read_from_cache.return_value = None  # Mock to always return None

        # Initialize MemoryAccess with the mocked CacheManager
        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size,
            cache_manager=self.cache_manager_mock  # Inject the mocked cache manager
        )
        self.initial_state = self.memory_access.get_initial_state(self.batch_size)

    def test_user_memory_isolation(self):
        # 模拟两个用户的不同输入
        controller_output_user1 = tf.random.uniform([1, self.controller_output_size], dtype=tf.float32)
        controller_output_user2 = tf.random.uniform([1, self.controller_output_size], dtype=tf.float32)

        controller_output = tf.concat([controller_output_user1, controller_output_user2], axis=0)

        inputs = {
            'inputs': controller_output,
            'prev_state': self.initial_state
        }

        output = self.memory_access(inputs, training=False)

        read_words = output['read_words']
        final_state = output['final_state']

        # 验证每个用户的记忆是隔离的
        memory_user1 = final_state.memory[0]
        memory_user2 = final_state.memory[1]

        # 使用 assertNotAllClose 并设置容差
        self.assertNotAllClose(memory_user1, memory_user2, atol=1e-6)

        # 验证张量形状一致性
        self.assertShapeEqual(memory_user1.numpy(), memory_user2.numpy())

    def test_memory_shape(self):
        """验证 memory 的形状是否正确"""
        expected_shape = [self.batch_size, self.memory_size, self.word_size]
        actual_shape = self.initial_state.memory.shape
        self.assertEqual(expected_shape, actual_shape)

    # 新增测试用例

    def test_content_read_write(self):
        """
        测试内容的读写操作是否正确。
        """
        # 定义写入向量
        write_vectors = tf.constant([
            [[1.0] * self.word_size]
        ], dtype=tf.float32)  # [1, 1, word_size]

        # 定义写入权重，写入到第一个内存槽
        write_weights = tf.constant([
            [[1.0] + [0.0] * (self.memory_size - 1)]
        ], dtype=tf.float32)  # [1, 1, memory_size]

        # 执行写入操作
        updated_memory, updated_usage = self.memory_access.write(
            write_weights=write_weights,
            write_vectors=write_vectors,
            prev_memory=self.initial_state.memory[:1],  # [1, memory_size, word_size]
            prev_usage=self.initial_state.usage[:1],    # [1, memory_size]
            training=False
        )  # [1, memory_size, word_size], [1, memory_size]

        # 定义读取权重（读取第一个内存槽）
        read_weights = tf.constant([
            [[1.0] + [0.0] * (self.memory_size - 1)]
        ], dtype=tf.float32)  # [1, num_reads, memory_size]

        # 执行读取操作
        read_vectors = self.memory_access.read(
            read_weights=read_weights,
            memory=updated_memory
        )  # [1, num_reads, word_size]

        # 预期读取的向量应与写入的向量相同
        expected_read_vectors = write_vectors  # [1, 1, word_size]

        # 断言
        self.assertAllClose(read_vectors, expected_read_vectors, atol=1e-6)

    def test_history_query_related_to_current_input(self):
        """
        测试与当前输入主题相关性的历史记录查询。
        """
        # 定义多个写入向量
        write_vectors = tf.constant([
            [[1.0, 0.0, 0.0, 0.0] * (self.word_size // 4)],
            [[0.0, 1.0, 0.0, 0.0] * (self.word_size // 4)],
            [[0.0, 0.0, 1.0, 0.0] * (self.word_size // 4)]
        ], dtype=tf.float32)  # [3, 1, word_size]

        # 定义写入权重，依次写入不同的内存槽
        write_weights = tf.constant([
            [[1.0] + [0.0] * (self.memory_size - 1)],
            [[0.0] + [1.0] + [0.0] * (self.memory_size - 2)],
            [[0.0] * 2 + [1.0] + [0.0] * (self.memory_size - 3)]
        ], dtype=tf.float32)  # [3, 1, memory_size]

        # 执行多次写入操作
        memory = self.initial_state.memory[:1]  # [1, memory_size, word_size]
        usage = self.initial_state.usage[:1]    # [1, memory_size]

        for i in range(3):
            memory, usage = self.memory_access.write(
                write_weights=write_weights[i:i+1, :, :],
                write_vectors=write_vectors[i:i+1, :, :],
                prev_memory=memory,
                prev_usage=usage,
                training=False
            )

        # 定义当前输入主题向量，类似于第一个写入向量
        current_input = tf.constant([
            [1.0] + [0.0] * (self.word_size - 1)
        ], dtype=tf.float32)  # [1, word_size]

        # 执行历史记录查询，检索与当前输入相关的 top_k 记录
        related_records = self.memory_access.query_history(
            query_vector=current_input,
            top_k=2
        )  # [1, 2, word_size]

        # 预期相关记录应为最相似的两个向量（第一个和第二个写入向量）
        expected_related_records = tf.constant([
            [
                [1.0, 0.0, 0.0, 0.0] * (self.word_size // 4),
                [0.0, 1.0, 0.0, 0.0] * (self.word_size // 4)
            ]
        ], dtype=tf.float32)  # [1, 2, word_size]

        # 断言
        self.assertAllClose(related_records, expected_related_records, atol=1e-6)

    def test_history_query_temporal_order(self):
        """
        测试相关历史记录查询结果按时序排列。
        """
        # 定义多个写入向量，后写入的向量更相关
        write_vectors = tf.constant([
            [[0.1, 0.2, 0.3, 0.4] * (self.word_size // 4)],
            [[0.4, 0.3, 0.2, 0.1] * (self.word_size // 4)],
            [[0.5, 0.5, 0.5, 0.5] * (self.word_size // 4)]
        ], dtype=tf.float32)  # [3, 1, word_size]

        # 定义写入权重，依次写入不同的内存槽
        write_weights = tf.constant([
            [[1.0] + [0.0] * (self.memory_size - 1)],
            [[0.0] + [1.0] + [0.0] * (self.memory_size - 2)],
            [[0.0] * 2 + [1.0] + [0.0] * (self.memory_size - 3)]
        ], dtype=tf.float32)  # [3, 1, memory_size]

        # 执行多次写入操作
        memory = self.initial_state.memory[:1]  # [1, memory_size, word_size]
        usage = self.initial_state.usage[:1]  # [1, memory_size]

        for i in range(3):
            memory, usage = self.memory_access.write(
                write_weights=write_weights[i:i + 1, :, :],
                write_vectors=write_vectors[i:i + 1, :, :],
                prev_memory=memory,
                prev_usage=usage,
                training=False
            )

        # 定义当前输入主题向量，最相关于最后一个写入
        current_input = tf.constant([
            [0.5] * self.word_size  # [1, word_size]
        ], dtype=tf.float32)

        # 执行历史记录查询，检索与当前输入相关的 top_k 记录
        related_records = self.memory_access.query_history(
            query_vector=current_input,
            top_k=3
        )  # [1, 3, word_size]

        # 预期相关记录应按照写入顺序排列，最相关的在前
        expected_related_records = tf.constant([
            [
                [0.5, 0.5, 0.5, 0.5] * (self.word_size // 4),
                [0.4, 0.3, 0.2, 0.1] * (self.word_size // 4),
                [0.1, 0.2, 0.3, 0.4] * (self.word_size // 4)
            ]
        ], dtype=tf.float32)  # [1, 3, word_size]

        # 断言
        self.assertAllClose(related_records, expected_related_records, atol=1e-6)

