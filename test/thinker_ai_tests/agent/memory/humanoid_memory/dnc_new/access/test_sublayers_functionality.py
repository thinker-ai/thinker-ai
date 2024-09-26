import tensorflow as tf
import numpy as np
import os

from thinker_ai.agent.memory.humanoid_memory.dnc_new import access
from thinker_ai.agent.memory.humanoid_memory.dnc_new.access import MemoryAccess, AccessState

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 定义测试常量
BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4
SEQUENCE_LENGTH = TIME_STEPS  # 保持一致性
INPUT_SIZE = 12  # 输入大小
EPSILON = 1e-6


class MemoryAccessSublayersFunctionalityTests(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessSublayersFunctionalityTests, self).setUp()
        # 初始化 MemoryAccess 模块
        self.module = MemoryAccess(
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_reads=NUM_READS,
            num_writes=NUM_WRITES,
            epsilon=EPSILON
        )
        # 构建模块以初始化权重
        # 通过调用一次模块，Keras会自动构建子层
        dummy_input = {
            'inputs': tf.zeros([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32),
            'prev_state': self.module.get_initial_state(batch_shape=[BATCH_SIZE])
        }
        _ = self.module(dummy_input, training=False)
        self.initial_state = self.module.get_initial_state(batch_shape=[BATCH_SIZE])

    def testValidReadMode(self):
        """测试读取模式的有效性，确保处理输入时各子层正确运作。"""
        inputs = {
            'inputs': tf.random.normal([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]),
            'prev_state': self.initial_state
        }

        # 假设有一个内部方法 _read_inputs 处理读取逻辑
        memory_tiled = tf.tile(inputs['prev_state'].memory, [SEQUENCE_LENGTH, 1, 1])
        processed_inputs = self.module._read_inputs(tf.reshape(inputs['inputs'], [-1, INPUT_SIZE]), memory_tiled)

        # 检查关键子层的输出是否存在
        self.assertIn('read_content_weights', processed_inputs)
        self.assertIn('write_content_weights', processed_inputs)
        self.assertIn('write_vectors', processed_inputs)
        self.assertIn('erase_vectors', processed_inputs)

    def testComputeWriteWeights(self):
        """测试写权重计算过程，确保输出形状正确。"""
        prev_state = self.initial_state
        controller_output = tf.random.normal([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE])
        reshaped_controller_output = tf.reshape(controller_output, [-1, INPUT_SIZE])
        memory_tiled = tf.tile(prev_state.memory, [SEQUENCE_LENGTH, 1, 1])
        processed_inputs = self.module._read_inputs(reshaped_controller_output, memory_tiled)

        # 提取计算 write_weights 所需的参数
        write_content_weights = processed_inputs['write_content_weights']  # [batch_size, num_writes, memory_size]
        allocation_gate = processed_inputs['allocation_gate']  # [batch_size, num_writes]
        write_gate = processed_inputs['write_gate']  # [batch_size, num_writes]

        # 直接调用 _compute_write_weights 方法
        write_weights = self.module._compute_write_weights(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_state.usage,
            training=False
        )

        self.assertEqual(write_weights.shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE])

    def testReadWeights(self):
        """测试读取权重的计算，确保输出形状正确。"""
        batch_size = BATCH_SIZE
        num_reads = NUM_READS
        memory_size = MEMORY_SIZE

        prev_state = self.initial_state

        controller_output = tf.random.normal([batch_size, SEQUENCE_LENGTH, INPUT_SIZE])
        reshaped_controller_output = tf.reshape(controller_output, [-1, INPUT_SIZE])

        memory_tiled = tf.tile(prev_state.memory, [SEQUENCE_LENGTH, 1, 1])
        processed_inputs = self.module._read_inputs(reshaped_controller_output, memory_tiled)

        next_state = self.module._step(processed_inputs, prev_state, training=False)

        # 获取最后一个时间步的 read_weights
        read_weights = next_state.read_weights[:, -1, :, :]  # 选择最后一个时间步，形状 [batch_size, num_reads, memory_size]

        # 打印用于调试
        tf.print("Final Read Weights Shape:", tf.shape(read_weights))

        # 验证形状是否符合预期
        self.assertEqual(read_weights.shape, [batch_size, num_reads, memory_size])

    def testSublayersRegistration(self):
        """测试所有子层是否正确注册在 MemoryAccess 模块中。"""
        expected_sublayers = [
            'write_content_weights', 'read_content_weights', 'temporal_linkage', 'freeness',
            'write_vectors', 'erase_vectors', 'free_gate', 'allocation_gate',
            'write_gate', 'read_mode', 'write_strengths', 'read_strengths', 'write_keys', 'read_keys'
        ]

        # 实例化 MemoryAccess 模块
        module = access.MemoryAccess(
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_reads=NUM_READS,
            num_writes=NUM_WRITES
        )

        # 初始化 prev_state 无 'read_words'
        initial_memory = tf.zeros([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE], dtype=tf.float32)
        initial_read_weights = tf.zeros([BATCH_SIZE, NUM_READS, MEMORY_SIZE], dtype=tf.float32)
        initial_write_weights = tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
        initial_linkage = module.temporal_linkage.get_initial_state(batch_size=BATCH_SIZE)
        initial_usage = module.freeness.get_initial_state([BATCH_SIZE])

        initial_state = AccessState(
            memory=initial_memory,
            read_weights=initial_read_weights,
            write_weights=initial_write_weights,
            linkage=initial_linkage,
            usage=initial_usage
        )

        # 创建单个时间步的随机输入
        dummy_input = tf.random.normal([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32)

        input_dict = {
            'inputs': dummy_input,
            'prev_state': initial_state
        }

        # 调用模块一次以构建它
        try:
            output, _ = module(input_dict, training=False)
        except Exception as e:
            self.fail(f"MemoryAccess module failed to build with input_dict: {e}")

        # 手动收集模块属性中的子层
        actual_sublayers = []
        for attr_name, attr_value in vars(module).items():
            if isinstance(attr_value, tf.keras.layers.Layer):
                actual_sublayers.append(attr_value.name)

        # 打印实际子层用于调试
        print("Expected sublayers:", expected_sublayers)
        print("Actual sublayers:", actual_sublayers)

        # 检查所有期望的子层是否在实际子层中
        for sublayer in expected_sublayers:
            self.assertIn(sublayer, actual_sublayers, f"Sublayer '{sublayer}' is not registered.")

    def testCosineWeightsFunctionality(self):
        """测试 CosineWeights 子层的功能，确保其计算内容权重正确。"""
        cosine_weights = access.CosineWeights(num_heads=NUM_WRITES, word_size=WORD_SIZE, name='test_cosine_weights')

        memory = tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])
        keys = tf.random.normal([BATCH_SIZE, NUM_WRITES, WORD_SIZE])
        strengths = tf.random.uniform([BATCH_SIZE, NUM_WRITES], minval=0.1, maxval=10.0)

        inputs = {
            'memory': memory,
            'keys': keys,
            'strengths': strengths
        }

        content_weights = cosine_weights(inputs)

        # 检查输出形状
        self.assertEqual(content_weights.shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE])

        # 检查权重和为1（softmax性质）
        sum_weights = tf.reduce_sum(content_weights, axis=2)
        self.assertAllClose(sum_weights, tf.ones_like(sum_weights), atol=1e-3, msg="Content weights do not sum to 1")

    def testTemporalLinkageFunctionality(self):
        """测试 TemporalLinkage 子层的功能，确保时序链路正确更新。"""
        temporal_linkage = access.TemporalLinkage(memory_size=MEMORY_SIZE, num_writes=NUM_WRITES,
                                                 name='test_temporal_linkage')

        write_weights = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], minval=0.0, maxval=1.0)
        write_weights /= tf.reduce_sum(write_weights, axis=2, keepdims=True) + 1e-6

        prev_linkage = access.TemporalLinkageState(
            link=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
            precedence_weights=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
        )

        inputs = {
            'write_weights': write_weights,
            'prev_linkage': prev_linkage
        }

        new_linkage = temporal_linkage(inputs)

        # 检查输出形状
        self.assertEqual(new_linkage.link.shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE])
        self.assertEqual(new_linkage.precedence_weights.shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE])

    def testFreenessFunctionality(self):
        """测试 Freeness 子层的功能，确保使用率和自由权重正确计算。"""
        freeness = access.Freeness(memory_size=MEMORY_SIZE, name='test_freeness')

        write_weights = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], minval=0.0, maxval=1.0)
        free_gate = tf.random.uniform([BATCH_SIZE, NUM_READS], minval=0.0, maxval=1.0)
        read_weights = tf.random.uniform([BATCH_SIZE, NUM_READS, MEMORY_SIZE], minval=0.0, maxval=1.0)
        prev_usage = tf.random.uniform([BATCH_SIZE, MEMORY_SIZE], minval=0.0, maxval=1.0)

        inputs = {
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': prev_usage
        }

        usage = freeness(inputs)

        # 检查输出形状
        self.assertEqual(usage.shape, [BATCH_SIZE, MEMORY_SIZE])

        # 检查使用率的合理范围
        self.assertTrue(tf.reduce_all(usage >= 0.0))
        self.assertTrue(tf.reduce_all(usage <= 1.0))


if __name__ == '__main__':
    tf.test.main()