import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import get_default_config
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess
from unittest import mock


class MemoryAccessInitializationTest(tf.test.TestCase):
    def test_initialization(self):
        memory_size = 128
        word_size = 64
        num_reads = 4
        num_writes = 1
        controller_output_size = 128
        # Mock CacheManager to avoid interference between tests
        cache_manager_mock = mock.Mock()
        cache_manager_mock.read_from_cache.return_value = None  # Mock to always return None

        # Initialize MemoryAccess with the mocked CacheManager
        self.memory_access = MemoryAccess(
            memory_size=memory_size,
            word_size=word_size,
            num_reads=num_reads,
            num_writes=num_writes,
            controller_output_size=controller_output_size,
            cache_manager=cache_manager_mock  # Inject the mocked cache manager
        )

        self.assertEqual(self.memory_access.memory_size, memory_size)
        self.assertEqual(self.memory_access.word_size, word_size)
        self.assertEqual(self.memory_access.num_reads, num_reads)
        self.assertEqual(self.memory_access.num_writes, num_writes)
        self.assertIsNotNone(self.memory_access.content_weight_calculator)
        self.assertIsNotNone(self.memory_access.write_weight_calculator)
        self.assertIsNotNone(self.memory_access.temporal_linkage_updater)
        self.assertIsNotNone(self.memory_access.read_weight_calculator)
        self.assertIsNotNone(self.memory_access.usage_updater)
        self.assertIsNotNone(self.memory_access.memory_updater)


class MemoryAccessSingleStepTest(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessSingleStepTest, self).setUp()
        self.memory_size = 16
        self.word_size = 8
        self.num_reads = 1
        self.num_writes = 1
        self.batch_size = 1
        self.controller_output_size = 64
        # Mock CacheManager to avoid interference between tests
        cache_manager_mock = mock.Mock()
        cache_manager_mock.read_from_cache.return_value = None  # Mock to always return None

        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size,
            cache_manager=cache_manager_mock  # Inject the mocked cache manager
        )

        self.initial_state = self.memory_access.get_initial_state(self.batch_size)

    def test_single_step_read_write(self):
        input_size = 32
        controller = tf.keras.layers.Dense(self.controller_output_size)
        controller_input = tf.random.uniform([self.batch_size, input_size], dtype=tf.float32)
        controller_output = controller(controller_input)

        inputs = {
            'inputs': controller_output,
            'prev_state': self.initial_state
        }

        output = self.memory_access(inputs, training=False)
        read_words = output['read_words']
        final_state = output['final_state']

        self.assertEqual(read_words.shape, (self.batch_size, self.num_reads, self.word_size))
        self.assertEqual(final_state.memory.shape, (self.batch_size, self.memory_size, self.word_size))


class MemoryAccessLayerTest(tf.test.TestCase):
    def setUp(self):
        super(MemoryAccessLayerTest, self).setUp()
        # 定义 MemoryAccess 的初始化参数
        self.memory_size = 10
        self.word_size = 16
        self.num_reads = 2
        self.num_writes = 2
        self.controller_output_size = 32  # 示例值
        self.cache_manager = None  # 根据需要定义或Mock
        self.name = 'memory_access_test'

        # 使用 get_default_config 方法生成配置
        self.config = get_default_config(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            num_reads=self.num_reads,
            word_size=self.word_size
        )

        # 初始化 MemoryAccess 实例
        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size,
            cache_manager=self.cache_manager,
            name=self.name,
            config=self.config
        )

    def test_calculate_interface_size(self):
        """测试 _calculate_interface_size() 方法的正确性"""
        # 使用 getattr 访问私有方法
        interface_size = getattr(self.memory_access, '_calculate_interface_size')()

        # 根据 MemoryAccess 的实现，计算预期的 interface_size
        expected_interface_size = (self.num_reads + self.num_writes) * self.word_size * 2.25  # 2+2)*16*2.25=144

        self.assertEqual(interface_size, expected_interface_size)

    def test_interface_layer_behavior(self):
        """测试 interface_layer 是否正确映射控制器输出到接口向量"""
        # 获取接口向量的大小
        interface_size = self.memory_access.interface_size

        # 创建已知的控制器输出
        # 假设控制器输出的形状为 [batch_size, controller_output_size]
        batch_size = 2
        controller_output = tf.constant([
            [1.0] * self.controller_output_size,
            [0.5] * self.controller_output_size
        ], dtype=tf.float32)  # [2, 32]

        # 传递一个“dummy”输入以构建 layer
        dummy_input = tf.zeros((1, self.controller_output_size), dtype=tf.float32)
        _ = self.memory_access.interface_layer(dummy_input)

        # 为了使输出可预测，设置固定的权重和偏置
        # 例如，将权重初始化为全1，偏置设置为全0
        fixed_weights = tf.ones((self.controller_output_size, interface_size), dtype=tf.float32)
        fixed_bias = tf.zeros((interface_size,), dtype=tf.float32)
        self.memory_access.interface_layer.set_weights([fixed_weights, fixed_bias])

        # 获取 interface_layer 的输出
        interface_output = self.memory_access.interface_layer(controller_output)  # [2, interface_size]

        # 计算预期输出
        # 每个输出单元 = sum(controller_output) + bias = sum(controller_output)
        sum_controller = tf.reduce_sum(controller_output, axis=1, keepdims=True)  # [2, 1]
        expected_interface_output = tf.tile(sum_controller, [1, interface_size])  # [2, interface_size]

        # 断言接口输出是否与预期相符
        self.assertAllClose(interface_output, expected_interface_output, atol=1e-6)

    def test_interface_layer_integration(self):
        """测试 interface_layer 在 MemoryAccess 中的集成行为"""
        # 定义写入权重和向量
        write_weights = tf.constant([
            [1.0] + [0.0] * (self.memory_size - 1),
            [1.0] + [0.0] * (self.memory_size - 1)
        ], dtype=tf.float32)  # [batch_size, memory_size]

        write_vectors = tf.constant([
            [1.0] * self.word_size,
            [0.5] * self.word_size
        ], dtype=tf.float32)  # [batch_size, word_size]

        # 创建控制器输出，以触发写入操作
        controller_output = tf.constant([
            [1.0] * self.controller_output_size,
            [0.5] * self.controller_output_size
        ], dtype=tf.float32)  # [2, 32]

        # 传递一个“dummy”输入以构建 layer
        dummy_input = tf.zeros((1, self.controller_output_size), dtype=tf.float32)
        _ = self.memory_access.interface_layer(dummy_input)

        # 设置固定权重和偏置以确保可预测的接口层输出
        interface_size = self.memory_access.interface_size
        fixed_weights = tf.ones((self.controller_output_size, interface_size), dtype=tf.float32)
        fixed_bias = tf.zeros((interface_size,), dtype=tf.float32)
        self.memory_access.interface_layer.set_weights([fixed_weights, fixed_bias])

        # 获取初始状态
        initial_state = self.memory_access.get_initial_state(batch_size=2)

        # 构建输入字典，模拟控制器输出和先前状态
        inputs = {
            'inputs': controller_output,
            'prev_state': initial_state
        }

        # 执行写入操作
        outputs = self.memory_access(inputs, training=False)
        read_words = outputs['read_words']
        final_state = outputs['final_state']

        # 执行查询操作
        query_vector = tf.constant([
            [1.0] * self.word_size,
            [0.5] * self.word_size
        ], dtype=tf.float32)  # [2, 16]

        related_records = self.memory_access.query_history(query_vector, top_k=1)  # [2, 1, 16]

        # 定义预期的相关记录
        expected_related_records = tf.constant([
            [[6.4] * self.word_size],
            [[3.2] * self.word_size]
        ], dtype=tf.float32)  # [2, 1, 16]

        # 断言查询结果是否与预期一致
        self.assertAllClose(related_records, expected_related_records, atol=1e-6)


if __name__ == '__main__':
    tf.test.main()
