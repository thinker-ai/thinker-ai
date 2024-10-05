# test_memory_access.py

import tensorflow as tf
import os

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import get_default_config
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 定义测试常量
BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4  # Simulated time steps
SEQUENCE_LENGTH = 1  # Initial input sequence length
INPUT_SIZE = 12  # Input size
EPSILON = 1e-6

# 获取默认配置
default_config = get_default_config(
    memory_size=MEMORY_SIZE,
    num_writes=NUM_WRITES,
    num_reads=NUM_READS,
    word_size=WORD_SIZE
)  # 使用动态配置


class MemoryAccessBasicFunctionalityTest(tf.test.TestCase):
    def setUp(self):
        super(MemoryAccessBasicFunctionalityTest, self).setUp()

        # Define test parameters and assign to self
        self.batch_size = BATCH_SIZE
        self.memory_size = MEMORY_SIZE
        self.word_size = WORD_SIZE
        self.num_reads = NUM_READS
        self.num_writes = NUM_WRITES
        self.initial_time_steps = 1

        # Initialize MemoryAccess
        self.module = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            epsilon=EPSILON,
            config=default_config
        )
        # 获取初始状态，不调用模块的 call 方法
        self.initial_state = self.module.get_initial_state(batch_size=self.batch_size, initial_time_steps=self.initial_time_steps)

    def testBuildAndTrain(self):
        inputs = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
        optimizer = tf.optimizers.SGD(learning_rate=1.0)

        with tf.GradientTape() as tape:
            output = self.module({
                'inputs': inputs,
                'prev_state': self.initial_state
            }, training=True)
            read_words = output['read_words']
            loss = tf.reduce_mean(read_words)

        gradients = tape.gradient(loss, self.module.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

        self.assertEqual(read_words.shape, (BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE))

    def testEdgeCaseInputs(self):
        inputs = tf.zeros([BATCH_SIZE, TIME_STEPS, INPUT_SIZE], dtype=tf.float32)
        targets = tf.zeros([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE], dtype=tf.float32)

        with tf.GradientTape() as tape:
            output = self.module({
                'inputs': inputs,
                'prev_state': self.initial_state
            }, training=True)
            read_words = output['read_words']
            loss = tf.reduce_mean(tf.square(read_words - targets))

        gradients = tape.gradient(loss, self.module.trainable_variables)

        # 检查是否至少有一个梯度大于零
        grads_greater_than_zero = [tf.reduce_sum(tf.abs(grad)).numpy() > 0.0 for grad in gradients if grad is not None]
        self.assertTrue(any(grads_greater_than_zero), "No gradients are greater than zero.")

    def testNonEdgeCaseInputs(self):
        inputs = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
        targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

        optimizer = tf.optimizers.SGD(learning_rate=1.0)

        with tf.GradientTape() as tape:
            output = self.module({
                'inputs': inputs,
                'prev_state': self.initial_state
            }, training=True)
            read_words = output['read_words']
            loss = tf.reduce_mean(tf.square(read_words - targets))

        gradients = tape.gradient(loss, self.module.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

        self.assertEqual(read_words.shape, (BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE))
        self.assertGreater(tf.norm(gradients[0]), 0.0)

    def test_generate_write_parameters(self):
        """
        测试 _generate_write_parameters 方法。
        """
        controller_output_t = tf.zeros([self.batch_size, INPUT_SIZE], dtype=tf.float32)
        write_vectors, erase_vectors, write_gate, allocation_gate, free_gate = self.module._generate_write_parameters(controller_output_t)

        # Check shapes
        self.assertAllEqual(write_vectors.shape, [self.batch_size, self.num_writes, self.word_size], msg="write_vectors shape mismatch.")
        self.assertAllEqual(erase_vectors.shape, [self.batch_size, self.num_writes, self.word_size], msg="erase_vectors shape mismatch.")
        self.assertAllEqual(write_gate.shape, [self.batch_size, self.num_writes], msg="write_gate shape mismatch.")
        self.assertAllEqual(allocation_gate.shape, [self.batch_size, self.num_writes], msg="allocation_gate shape mismatch.")
        self.assertAllEqual(free_gate.shape, [self.batch_size, self.num_reads], msg="free_gate shape mismatch.")

        # Check value ranges (write_gate, allocation_gate should be between 0 and 1 due to sigmoid activation)
        self.assertTrue(tf.reduce_all(write_gate >= 0) and tf.reduce_all(write_gate <= 1), msg="write_gate values out of range.")
        self.assertTrue(tf.reduce_all(allocation_gate >= 0) and tf.reduce_all(allocation_gate <= 1), msg="allocation_gate values out of range.")
        self.assertTrue(tf.reduce_all(free_gate >= 0) and tf.reduce_all(free_gate <= 1), msg="free_gate values out of range.")

    def test_generate_read_mode(self):
        """
        测试 _generate_read_mode 方法。
        """
        controller_output_t = tf.zeros([self.batch_size, INPUT_SIZE], dtype=tf.float32)
        read_mode = self.module._generate_read_mode(controller_output_t)

        # Check shape
        self.assertAllEqual(read_mode.shape, [self.batch_size, self.num_reads, 1 + 2 * self.num_writes], msg="read_mode shape mismatch.")

        # Since activation is None, we cannot assert value ranges

    def test_generate_write_keys_strengths(self):
        """
        测试 _generate_write_keys_strengths 方法。
        """
        controller_output_t = tf.zeros([self.batch_size, INPUT_SIZE], dtype=tf.float32)
        write_keys, write_strengths = self.module._generate_write_keys_strengths(controller_output_t)

        # Check shapes
        self.assertAllEqual(write_keys.shape, [self.batch_size, self.num_writes, self.word_size], msg="write_keys shape mismatch.")
        self.assertAllEqual(write_strengths.shape, [self.batch_size, self.num_writes], msg="write_strengths shape mismatch.")

        # Check write_strengths values (softplus activation ensures positive)
        self.assertTrue(tf.reduce_all(write_strengths > 0), msg="write_strengths contain non-positive values.")

    def test_generate_read_keys_strengths(self):
        """
        测试 _generate_read_keys_strengths 方法。
        """
        controller_output_t = tf.zeros([self.batch_size, INPUT_SIZE], dtype=tf.float32)
        read_keys, read_strengths = self.module._generate_read_keys_strengths(controller_output_t)

        # Check shapes
        self.assertAllEqual(read_keys.shape, [self.batch_size, self.num_reads, self.word_size], msg="read_keys shape mismatch.")
        self.assertAllEqual(read_strengths.shape, [self.batch_size, self.num_reads], msg="read_strengths shape mismatch.")

        # Check read_strengths values (softplus activation ensures positive)
        self.assertTrue(tf.reduce_all(read_strengths > 0), msg="read_strengths contain non-positive values.")

    def test_compute_content_weights(self):
        """
        测试 _compute_content_weights 方法。
        """
        keys = tf.zeros([self.batch_size, self.num_writes, self.word_size], dtype=tf.float32)
        strengths = tf.ones([self.batch_size, self.num_writes], dtype=tf.float32)  # avoid zero strength
        memory = tf.zeros([self.batch_size, self.memory_size, self.word_size], dtype=tf.float32)

        content_weights = self.module._compute_content_weights(keys, strengths, memory)

        # Check shape
        self.assertAllEqual(content_weights.shape, [self.batch_size, self.num_writes, self.memory_size], msg="content_weights shape mismatch.")

        # Since keys and memory are zeros, depending on implementation, content_weights might be uniform or some constant
        # Here, we can't assert exact values, but we can check for valid distributions if softmax is used internally

    def test_compute_write_weights(self):
        """
        测试 _compute_write_weights 方法。
        """
        write_content_weights = tf.ones([self.batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        allocation_gate = tf.ones([self.batch_size, self.num_writes], dtype=tf.float32)  # fully allocate
        write_gate = tf.ones([self.batch_size, self.num_writes], dtype=tf.float32)  # fully write
        prev_usage = tf.zeros([self.batch_size, self.memory_size], dtype=tf.float32)  # no usage
        training = False

        write_weights = self.module._compute_write_weights(write_content_weights, allocation_gate, write_gate, prev_usage, training)

        # Check shape
        self.assertAllEqual(write_weights.shape, [self.batch_size, self.num_writes, self.memory_size], msg="write_weights shape mismatch.")

        # Depending on implementation, values should be non-negative and sum to allocation or write constraints
        # Here, we check for non-negative
        self.assertTrue(tf.reduce_all(write_weights >= 0), msg="write_weights contain negative values.")

    def test_update_memory(self):
        """
        测试 _update_memory 方法。
        """
        memory = tf.zeros([self.batch_size, self.memory_size, self.word_size], dtype=tf.float32)
        write_weights = tf.ones([self.batch_size, self.num_writes, self.memory_size], dtype=tf.float32)  # fully write
        erase_vectors = tf.ones([self.batch_size, self.num_writes, self.word_size], dtype=tf.float32)  # fully erase
        write_vectors = tf.ones([self.batch_size, self.num_writes, self.word_size], dtype=tf.float32)  # write ones

        memory_updated = self.module._update_memory(memory, write_weights, erase_vectors, write_vectors)

        # Check shape
        self.assertAllEqual(memory_updated.shape, [self.batch_size, self.memory_size, self.word_size], msg="memory_updated shape mismatch.")

        # Since erase_vectors are ones and memory is zeros, memory_erased = 0
        # Add write_weights * write_vectors, assuming update_memory is implemented correctly
        # Exact values depend on update_memory implementation
        # Here, we can check for non-negative values
        self.assertTrue(tf.reduce_all(memory_updated >= 0), msg="memory_updated contains negative values.")

    def test_update_linkage(self):
        """
        测试 _update_linkage 方法。
        """
        write_weights = tf.ones([self.batch_size, self.num_writes, self.memory_size], dtype=tf.float32)  # fully write
        prev_linkage = {
            'link': tf.zeros([self.batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([self.batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        }
        training = False

        linkage_updated = self.module._update_linkage(write_weights, prev_linkage, training)

        # Check linkage_updated structure
        self.assertIn('link', linkage_updated, msg="'link' key missing in linkage_updated.")
        self.assertIn('precedence_weights', linkage_updated, msg="'precedence_weights' key missing in linkage_updated.")

        # Check shapes
        self.assertAllEqual(linkage_updated['link'].shape, [self.batch_size, self.num_writes, self.memory_size, self.memory_size], msg="link shape mismatch.")
        self.assertAllEqual(linkage_updated['precedence_weights'].shape, [self.batch_size, self.num_writes, self.memory_size], msg="precedence_weights shape mismatch.")

        # Check that 'link' has been updated appropriately (depending on implementation)
        # Here, we can check for non-negative values
        self.assertTrue(tf.reduce_all(linkage_updated['link'] >= 0), msg="'link' contains negative values.")

    def test_update_usage(self):
        """
        测试 _update_usage 方法。
        """
        write_weights = tf.ones([self.batch_size, self.num_writes, self.memory_size], dtype=tf.float32)  # fully write
        free_gate = tf.ones([self.batch_size, self.num_reads], dtype=tf.float32)  # fully free
        read_weights_prev = tf.zeros([self.batch_size, self.num_reads, self.memory_size], dtype=tf.float32)  # no previous reads
        prev_usage = tf.zeros([self.batch_size, self.memory_size], dtype=tf.float32)  # no previous usage
        training = False

        usage_updated = self.module._update_usage(write_weights, free_gate, read_weights_prev, prev_usage, training)

        # Check shape
        self.assertAllEqual(usage_updated.shape, [self.batch_size, self.memory_size], msg="usage_updated shape mismatch.")

        # Depending on implementation, usage should have been updated appropriately
        # Here, we can check for non-negative values
        self.assertTrue(tf.reduce_all(usage_updated >= 0), msg="usage_updated contains negative values.")

    def test_compute_read_weights(self):
        """
        测试 _compute_read_weights 方法。
        """
        read_content_weights = tf.ones([self.batch_size, self.num_reads, self.memory_size], dtype=tf.float32)
        prev_read_weights = tf.ones([self.batch_size, self.num_reads, self.memory_size], dtype=tf.float32)
        link = tf.zeros([self.batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)
        read_mode = tf.zeros([self.batch_size, self.num_reads, 1 + 2 * self.num_writes], dtype=tf.float32)
        training = False

        read_weights = self.module._compute_read_weights(read_content_weights, prev_read_weights, link, read_mode, training)

        # Check shape
        self.assertAllEqual(read_weights.shape, [self.batch_size, self.num_reads, self.memory_size], msg="read_weights shape mismatch.")

        # Check value ranges (assuming softmax normalization inside compute)
        self.assertTrue(tf.reduce_all(read_weights >= 0) and tf.reduce_all(read_weights <= 1), msg="read_weights values out of range.")
        # Check that read_weights sum to 1 across memory_size
        sums = tf.reduce_sum(read_weights, axis=-1)
        self.assertTrue(tf.reduce_all(tf.abs(sums - 1.0) < 1e-6), msg="read_weights do not sum to 1 across memory_size.")

    def test_read_words(self):
        """
        测试 _read_words 方法。
        """
        # Define read_weights and memory_updated with known shapes
        read_weights = tf.ones([self.batch_size, self.num_reads, self.memory_size], dtype=tf.float32)
        memory_updated = tf.ones([self.batch_size, self.memory_size, self.word_size], dtype=tf.float32)

        read_words = self.module._read_words(read_weights, memory_updated)

        # Check shape
        self.assertAllEqual(read_words.shape, [self.batch_size, self.num_reads, self.word_size], msg="read_words shape mismatch.")

        # Check values: tf.matmul with all ones should sum over memory_size, resulting in [memory_size * 1] = [memory_size]
        # Each element in read_words should be memory_size * 1 = memory_size
        expected_read_words = tf.ones([self.batch_size, self.num_reads, self.word_size], dtype=tf.float32) * self.memory_size
        self.assertAllClose(read_words, expected_read_words, atol=1e-6, msg="read_words values mismatch.")

    def test_get_initial_state(self):
        """
        测试 get_initial_state 方法是否正确初始化整个状态。
        """
        initial_state = self.module.get_initial_state(batch_size=self.batch_size, initial_time_steps=self.initial_time_steps)

        # Define expected tensors
        expected_memory = tf.zeros([self.batch_size, self.memory_size, self.word_size], dtype=tf.float32)
        expected_read_weights = tf.zeros([self.batch_size, self.initial_time_steps, self.num_reads, self.memory_size], dtype=tf.float32)
        expected_write_weights = tf.zeros([self.batch_size, self.initial_time_steps, self.num_writes, self.memory_size], dtype=tf.float32)
        expected_link = tf.zeros([self.batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32)
        expected_precedence_weights = tf.zeros([self.batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        expected_usage = tf.zeros([self.batch_size, self.memory_size], dtype=tf.float32)
        expected_read_words = tf.zeros([self.batch_size, self.num_reads, self.word_size], dtype=tf.float32)

        # Check shapes and values
        self.assertAllEqual(initial_state.memory.shape, expected_memory.shape, msg="Initial memory shape mismatch.")
        self.assertAllClose(initial_state.memory, expected_memory, atol=1e-6, msg="Initial memory state mismatch.")

        self.assertAllEqual(initial_state.read_weights.shape, expected_read_weights.shape, msg="Initial read_weights shape mismatch.")
        self.assertAllClose(initial_state.read_weights, expected_read_weights, atol=1e-6, msg="Initial read_weights state mismatch.")

        self.assertAllEqual(initial_state.write_weights.shape, expected_write_weights.shape, msg="Initial write_weights shape mismatch.")
        self.assertAllClose(initial_state.write_weights, expected_write_weights, atol=1e-6, msg="Initial write_weights state mismatch.")

        self.assertIn('link', initial_state.linkage)
        self.assertIn('precedence_weights', initial_state.linkage)

        self.assertAllEqual(initial_state.linkage['link'].shape, expected_link.shape, msg="Initial linkage['link'] shape mismatch.")
        self.assertAllClose(initial_state.linkage['link'], expected_link, atol=1e-6, msg="Initial linkage['link'] state mismatch.")

        self.assertAllEqual(initial_state.linkage['precedence_weights'].shape, expected_precedence_weights.shape, msg="Initial linkage['precedence_weights'] shape mismatch.")
        self.assertAllClose(initial_state.linkage['precedence_weights'], expected_precedence_weights, atol=1e-6, msg="Initial linkage['precedence_weights'] state mismatch.")

        self.assertAllEqual(initial_state.usage.shape, expected_usage.shape, msg="Initial usage shape mismatch.")
        self.assertAllClose(initial_state.usage, expected_usage, atol=1e-6, msg="Initial usage state mismatch.")

        self.assertAllEqual(initial_state.read_words.shape, expected_read_words.shape, msg="Initial read_words shape mismatch.")
        self.assertAllClose(initial_state.read_words, expected_read_words, atol=1e-6, msg="Initial read_words state mismatch.")

    def test_call_updates_state(self):
        """
        测试 call 方法是否正确更新状态。
        """
        # 定义确定性的输入（全零）
        inputs = tf.zeros([self.batch_size, TIME_STEPS, INPUT_SIZE], dtype=tf.float32)
        # 使用初始状态（全零）

        # 调用模块
        output = self.module({
            'inputs': inputs,
            'prev_state': self.initial_state
        }, training=False)

        # 提取 read_words 和 final_state
        read_words = output['read_words']
        final_state = output['final_state']

        # 检查 read_words 的形状
        self.assertAllEqual(read_words.shape, [self.batch_size, TIME_STEPS, self.num_reads, self.word_size],
                            msg="read_words shape mismatch after call.")

        # 检查 read_words 的数值范围
        # 由于偏置初始化为随机正负值，read_words 不应全为零，但应接近偏置引起的初始输出
        # 这里，我们检查 read_words 的值是否在合理的范围内，例如 [-0.1, 0.1]
        self.assertTrue(tf.reduce_all(read_words >= -0.5) and tf.reduce_all(read_words <= 0.5),
                        msg="read_words values out of expected range after call.")

        # 检查 final_state 组件的形状和数值范围
        # 内存应为非负值
        self.assertAllEqual(final_state.memory.shape, [self.batch_size, self.memory_size, self.word_size],
                            msg="final_state.memory shape mismatch after call.")
        self.assertTrue(tf.reduce_all(final_state.memory >= 0),
                        msg="final_state.memory contains negative values after call.")

        # 检查 read_weights 的形状和数值范围
        self.assertAllEqual(final_state.read_weights.shape,
                            [self.batch_size, self.initial_time_steps + TIME_STEPS, self.num_reads, self.memory_size],
                            msg="final_state.read_weights shape mismatch after call.")
        self.assertTrue(tf.reduce_all(final_state.read_weights >= 0) and tf.reduce_all(final_state.read_weights <= 1),
                        msg="final_state.read_weights values out of range after call.")
        # 检查 read_weights 在 memory_size 维度上是否归一化
        sums = tf.reduce_sum(final_state.read_weights, axis=-1)
        self.assertTrue(tf.reduce_all(tf.abs(sums - 1.0) < 1e-3),
                        msg="final_state.read_weights do not sum to 1 across memory_size after call.")

        # 检查 write_weights 的形状和数值范围
        self.assertAllEqual(final_state.write_weights.shape,
                            [self.batch_size, self.initial_time_steps + TIME_STEPS, self.num_writes, self.memory_size],
                            msg="final_state.write_weights shape mismatch after call.")
        self.assertTrue(tf.reduce_all(final_state.write_weights >= 0),
                        msg="final_state.write_weights contain negative values after call.")

        # 检查 linkage
        self.assertIn('link', final_state.linkage, msg="'link' key missing in final_state.linkage.")
        self.assertIn('precedence_weights', final_state.linkage,
                      msg="'precedence_weights' key missing in final_state.linkage.")
        self.assertAllEqual(final_state.linkage['link'].shape,
                            [self.batch_size, self.num_writes, self.memory_size, self.memory_size],
                            msg="final_state.linkage['link'] shape mismatch after call.")
        self.assertAllEqual(final_state.linkage['precedence_weights'].shape,
                            [self.batch_size, self.num_writes, self.memory_size],
                            msg="final_state.linkage['precedence_weights'] shape mismatch after call.")
        self.assertTrue(tf.reduce_all(final_state.linkage['link'] >= 0),
                        msg="final_state.linkage['link'] contains negative values after call.")

        # 检查 usage
        self.assertAllEqual(final_state.usage.shape, [self.batch_size, self.memory_size],
                            msg="final_state.usage shape mismatch after call.")
        self.assertTrue(tf.reduce_all(final_state.usage >= 0),
                        msg="final_state.usage contains negative values after call.")

        # 检查 final_state.read_words 的形状和数值范围
        self.assertAllEqual(final_state.read_words.shape, [self.batch_size, self.num_reads, self.word_size],
                            msg="final_state.read_words shape mismatch after call.")
        self.assertTrue(tf.reduce_all(final_state.read_words >= -0.5) and tf.reduce_all(final_state.read_words <= 0.5),
                        msg="final_state.read_words values out of expected range after call.")

    def test_call_step_by_step(self):
        """
        测试 call 方法的每个时间步，逐步验证状态更新。
        """
        # 定义单步输入（全零）
        single_step_inputs = tf.zeros([self.batch_size, 1, INPUT_SIZE], dtype=tf.float32)

        # 调用模块
        output = self.module({
            'inputs': single_step_inputs,
            'prev_state': self.initial_state
        }, training=False)

        # 提取 read_words 和 final_state
        read_words = output['read_words']
        final_state = output['final_state']

        # 检查 read_words 的形状
        self.assertAllEqual(read_words.shape, [self.batch_size, 1, self.num_reads, self.word_size],
                            msg="read_words shape mismatch after single step call.")

        # 检查 read_words 的数值范围
        self.assertTrue(tf.reduce_all(read_words >= -0.5) and tf.reduce_all(read_words <= 0.5),
                        msg="read_words values out of expected range after single step call.")

        # 检查 final_state 组件的形状和数值范围
        # 内存应为非负值
        self.assertAllEqual(final_state.memory.shape, [self.batch_size, self.memory_size, self.word_size],
                            msg="final_state.memory shape mismatch after single step call.")
        self.assertTrue(tf.reduce_all(final_state.memory >= 0),
                        msg="final_state.memory contains negative values after single step call.")

        # 检查 read_weights 的形状和数值范围
        self.assertAllEqual(final_state.read_weights.shape,
                            [self.batch_size, self.initial_time_steps + 1, self.num_reads, self.memory_size],
                            msg="final_state.read_weights shape mismatch after single step call.")
        self.assertTrue(tf.reduce_all(final_state.read_weights >= 0) and tf.reduce_all(final_state.read_weights <= 1),
                        msg="final_state.read_weights values out of range after single step call.")
        # 检查 read_weights 在 memory_size 维度上是否归一化
        sums = tf.reduce_sum(final_state.read_weights, axis=-1)
        self.assertTrue(tf.reduce_all(tf.abs(sums - 1.0) < 1e-3),
                        msg="final_state.read_weights do not sum to 1 across memory_size after single step call.")

        # 检查 write_weights 的形状和数值范围
        self.assertAllEqual(final_state.write_weights.shape,
                            [self.batch_size, self.initial_time_steps + 1, self.num_writes, self.memory_size],
                            msg="final_state.write_weights shape mismatch after single step call.")
        self.assertTrue(tf.reduce_all(final_state.write_weights >= 0),
                        msg="final_state.write_weights contain negative values after single step call.")

        # 检查 linkage
        self.assertIn('link', final_state.linkage, msg="'link' key missing in final_state.linkage after single step call.")
        self.assertIn('precedence_weights', final_state.linkage,
                      msg="'precedence_weights' key missing in final_state.linkage after single step call.")
        self.assertAllEqual(final_state.linkage['link'].shape,
                            [self.batch_size, self.num_writes, self.memory_size, self.memory_size],
                            msg="final_state.linkage['link'] shape mismatch after single step call.")
        self.assertAllEqual(final_state.linkage['precedence_weights'].shape,
                            [self.batch_size, self.num_writes, self.memory_size],
                            msg="final_state.linkage['precedence_weights'] shape mismatch after single step call.")
        self.assertTrue(tf.reduce_all(final_state.linkage['link'] >= 0),
                        msg="final_state.linkage['link'] contains negative values after single step call.")

        # 检查 usage
        self.assertAllEqual(final_state.usage.shape, [self.batch_size, self.memory_size],
                            msg="final_state.usage shape mismatch after single step call.")
        self.assertTrue(tf.reduce_all(final_state.usage >= 0),
                        msg="final_state.usage contains negative values after single step call.")

        # 检查 final_state.read_words 的形状和数值范围
        self.assertAllEqual(final_state.read_words.shape, [self.batch_size, self.num_reads, self.word_size],
                            msg="final_state.read_words shape mismatch after single step call.")
        self.assertTrue(tf.reduce_all(final_state.read_words >= -0.5) and tf.reduce_all(final_state.read_words <= 0.5),
                        msg="final_state.read_words values out of expected range after single step call.")

        # 进一步验证多个时间步的调用
        # 定义多步输入（全零）
        multi_step_inputs = tf.zeros([self.batch_size, 3, INPUT_SIZE], dtype=tf.float32)

        # 调用模块
        output_multi = self.module({
            'inputs': multi_step_inputs,
            'prev_state': self.initial_state
        }, training=False)

        # 提取 read_words 和 final_state
        read_words_multi = output_multi['read_words']
        final_state_multi = output_multi['final_state']

        # 检查 read_words 的形状
        self.assertAllEqual(read_words_multi.shape, [self.batch_size, 3, self.num_reads, self.word_size],
                            msg="read_words shape mismatch after multi-step call.")

        # 检查 read_words 的数值范围
        self.assertTrue(tf.reduce_all(read_words_multi >= -0.5) and tf.reduce_all(read_words_multi <= 0.5),
                        msg="read_words values out of expected range after multi-step call.")

        # 检查 final_state 组件的形状和数值范围
        self.assertAllEqual(final_state_multi.memory.shape, [self.batch_size, self.memory_size, self.word_size],
                            msg="final_state.memory shape mismatch after multi-step call.")
        self.assertTrue(tf.reduce_all(final_state_multi.memory >= 0),
                        msg="final_state.memory contains negative values after multi-step call.")

        self.assertAllEqual(final_state_multi.read_weights.shape,
                            [self.batch_size, self.initial_time_steps + 3, self.num_reads, self.memory_size],
                            msg="final_state.read_weights shape mismatch after multi-step call.")
        self.assertTrue(
            tf.reduce_all(final_state_multi.read_weights >= 0) and tf.reduce_all(final_state_multi.read_weights <= 1),
            msg="final_state.read_weights values out of range after multi-step call.")
        sums_multi = tf.reduce_sum(final_state_multi.read_weights, axis=-1)
        self.assertTrue(tf.reduce_all(tf.abs(sums_multi - 1.0) < 1e-3),
                        msg="final_state.read_weights do not sum to 1 across memory_size after multi-step call.")

        self.assertAllEqual(final_state_multi.write_weights.shape,
                            [self.batch_size, self.initial_time_steps + 3, self.num_writes, self.memory_size],
                            msg="final_state.write_weights shape mismatch after multi-step call.")
        self.assertTrue(tf.reduce_all(final_state_multi.write_weights >= 0),
                        msg="final_state.write_weights contain negative values after multi-step call.")

        self.assertIn('link', final_state_multi.linkage,
                      msg="'link' key missing in final_state.linkage after multi-step call.")
        self.assertIn('precedence_weights', final_state_multi.linkage,
                      msg="'precedence_weights' key missing in final_state.linkage after multi-step call.")
        self.assertAllEqual(final_state_multi.linkage['link'].shape,
                            [self.batch_size, self.num_writes, self.memory_size, self.memory_size],
                            msg="final_state.linkage['link'] shape mismatch after multi-step call.")
        self.assertAllEqual(final_state_multi.linkage['precedence_weights'].shape,
                            [self.batch_size, self.num_writes, self.memory_size],
                            msg="final_state.linkage['precedence_weights'] shape mismatch after multi-step call.")
        self.assertTrue(tf.reduce_all(final_state_multi.linkage['link'] >= 0),
                        msg="final_state.linkage['link'] contains negative values after multi-step call.")

        self.assertAllEqual(final_state_multi.usage.shape, [self.batch_size, self.memory_size],
                            msg="final_state.usage shape mismatch after multi-step call.")
        self.assertTrue(tf.reduce_all(final_state_multi.usage >= 0),
                        msg="final_state.usage contains negative values after multi-step call.")

        self.assertAllEqual(final_state_multi.read_words.shape, [self.batch_size, self.num_reads, self.word_size],
                            msg="final_state.read_words shape mismatch after multi-step call.")
        self.assertTrue(
            tf.reduce_all(final_state_multi.read_words >= -0.5) and tf.reduce_all(final_state_multi.read_words <= 0.5),
            msg="final_state.read_words values out of expected range after multi-step call.")


if __name__ == '__main__':
    tf.test.main()
