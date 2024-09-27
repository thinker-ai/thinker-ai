import numpy as np
import tensorflow as tf
import os

from thinker_ai.agent.memory.humanoid_memory.dnc_new.access import MemoryAccess, AccessState

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 定义测试常量
BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4
SEQUENCE_LENGTH = 1
INPUT_SIZE = 12  # 输入大小
EPSILON = 1e-6


class MemoryAccessDataTypesAndConfigurationsTest(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessDataTypesAndConfigurationsTest, self).setUp()

        # 定义一个可训练的 Dense 层用于生成 write_content_weights
        self.write_content_weights_layer = tf.keras.layers.Dense(
            units=NUM_WRITES * MEMORY_SIZE,
            activation=None,
            name='write_content_weights',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )

        # 定义 write_content_weights_fn，使其依赖于 inputs
        def write_content_weights_fn(inputs):
            # 假设 'inputs' 中包含 'inputs' 键，其形状为 [batch_size, sequence_length, input_size]
            batch_size = tf.shape(inputs['inputs'])[0]
            sequence_length = tf.shape(inputs['inputs'])[1]
            controller_output = tf.reshape(inputs['inputs'],
                                           [-1, INPUT_SIZE])  # [batch_size * sequence_length, input_size]
            write_weights = self.write_content_weights_layer(
                controller_output)  # [batch_size * sequence_length, num_writes * memory_size]
            write_weights = tf.reshape(write_weights, [-1, NUM_WRITES,
                                                       MEMORY_SIZE])  # [batch_size * sequence_length, num_writes, memory_size]
            return write_weights

        # 将函数赋值给类成员
        self.write_content_weights_fn = write_content_weights_fn

        # 初始化 MemoryAccess 模块，传入 write_content_weights_fn
        self.module = MemoryAccess(
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_reads=NUM_READS,
            num_writes=NUM_WRITES,
            epsilon=EPSILON,
            write_content_weights_fn=self.write_content_weights_fn  # 传入函数
        )

        # 初始化状态
        batch_size = BATCH_SIZE  # 直接使用整数
        dummy_input = {
            'inputs': tf.zeros([batch_size, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32),
            'prev_state': self.module.get_initial_state(batch_size=batch_size, initial_time_steps=1)
        }
        _ = self.module(dummy_input, training=False)
        self.initial_state = self.module.get_initial_state(batch_size=batch_size, initial_time_steps=1)

    def _run_forward_pass(self, module, initial_state, inputs, targets, track_gradients=True):
        """
        辅助方法：运行前向传播并计算损失。
        """
        if track_gradients:
            with tf.GradientTape() as tape:
                output = module({'inputs': inputs, 'prev_state': initial_state}, training=True)
                loss = tf.reduce_mean(tf.square(output['read_words'] - targets))
            gradients = tape.gradient(loss, [inputs] + module.trainable_variables)
            return loss, gradients
        else:
            output = module({'inputs': inputs, 'prev_state': initial_state}, training=True)
            loss = tf.reduce_mean(tf.square(output['read_words'] - targets))
            return loss, None

    def _check_gradients(self, gradients, variables):
        """
        辅助方法：检查梯度是否存在且在合理范围内。
        """
        for grad, var in zip(gradients, variables):
            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
            if grad is None:
                self.fail(f"Gradient for {var_name} is None.")
            else:
                grad_norm = tf.norm(grad).numpy()
                self.assertGreater(grad_norm, 1e-12, f"Gradient for variable '{var_name}' is too small.")
                self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var_name}' is too large.")

    def testDifferentDataTypes(self):
        """测试 MemoryAccess 模型在不同数据类型（如 float16）下的兼容性。"""
        # 创建新的模块实例，并传递 write_content_weights_fn
        module_fp16 = MemoryAccess(
            MEMORY_SIZE,
            WORD_SIZE,
            NUM_READS,
            NUM_WRITES,
            write_content_weights_fn=self.write_content_weights_fn
        )

        # 初始化状态，使用字典替代 TemporalLinkageState
        initial_state_fp16 = AccessState(
            memory=tf.Variable(tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE], dtype=tf.float16), name='memory_fp16'),
            read_weights=tf.Variable(tf.random.normal([BATCH_SIZE, 1, NUM_READS, MEMORY_SIZE], dtype=tf.float16), name='read_weights_fp16'),
            write_weights=tf.Variable(tf.random.normal([BATCH_SIZE, 1, NUM_WRITES, MEMORY_SIZE], dtype=tf.float16), name='write_weights_fp16'),
            linkage={
                'link': tf.Variable(
                    tf.random.normal([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float16),
                    name='link_fp16'
                ),
                'precedence_weights': tf.Variable(
                    tf.random.normal([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float16),
                    name='precedence_weights_fp16'
                )
            },
            usage=tf.Variable(tf.random.normal([BATCH_SIZE, MEMORY_SIZE], dtype=tf.float16), name='usage_fp16'),
            read_words=tf.zeros([BATCH_SIZE, NUM_READS, WORD_SIZE], dtype=tf.float16)  # 确保 read_words 被正确初始化
        )

        # 创建输入和目标，调整维度顺序为 [batch_size, time_steps, input_size]
        inputs = tf.Variable(np.random.randn(BATCH_SIZE, TIME_STEPS, INPUT_SIZE).astype(np.float16), dtype=tf.float16, trainable=True)
        targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE], dtype=tf.float16)

        with tf.GradientTape() as tape:
            output = module_fp16({'inputs': inputs, 'prev_state': initial_state_fp16}, training=True)
            loss = tf.reduce_sum(output['read_words'])

        gradients = tape.gradient(loss, [inputs] + list(module_fp16.trainable_variables))

        # 检查梯度
        for grad, var in zip(gradients, [inputs] + list(module_fp16.trainable_variables)):
            var_name = var.name
            if grad is None:
                self.fail(f"Gradient for variable '{var_name}' is None.")
            else:
                grad_norm = tf.norm(grad).numpy()
                self.assertGreater(grad_norm, 1e-3, f"Gradient norm for '{var_name}' is too small for float16.")
                self.assertLess(grad_norm, 1e3, f"Gradient norm for '{var_name}' is too large for float16.")

    def testDifferentBatchSizesAndSequenceLengths(self):
        """测试模型在不同批次大小和序列长度下的表现。"""
        batch_sizes = [1, 4, 8]
        time_steps_options = [1, 10, 20]

        for batch_size in batch_sizes:
            for time_steps in time_steps_options:
                print(f"Testing with BATCH_SIZE={batch_size}, TIME_STEPS={time_steps}")

                # 创建新的模块实例，并传递 write_content_weights_fn
                module = MemoryAccess(
                    MEMORY_SIZE,
                    WORD_SIZE,
                    NUM_READS,
                    NUM_WRITES,
                    write_content_weights_fn=self.write_content_weights_fn
                )

                # 初始化随机状态，使用字典替代 TemporalLinkageState
                initial_state = AccessState(
                    memory=tf.zeros([batch_size, MEMORY_SIZE, WORD_SIZE], dtype=tf.float32),
                    read_weights=tf.zeros([batch_size, 1, NUM_READS, MEMORY_SIZE], dtype=tf.float32),
                    write_weights=tf.zeros([batch_size, 1, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32),
                    linkage={
                        'link': tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
                        'precedence_weights': tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
                    },
                    usage=tf.zeros([batch_size, MEMORY_SIZE], dtype=tf.float32),
                    read_words=tf.zeros([batch_size, NUM_READS, WORD_SIZE], dtype=tf.float32)  # 确保 read_words 被正确初始化
                )

                # 创建输入和目标，调整维度顺序为 [batch_size, time_steps, input_size]
                inputs = tf.Variable(tf.random.normal([batch_size, time_steps, INPUT_SIZE]), trainable=True)
                targets = tf.random.normal([batch_size, time_steps, NUM_READS, WORD_SIZE])

                loss, gradients = self._run_forward_pass(module, initial_state, inputs, targets, track_gradients=True)

                # 检查梯度
                self._check_gradients(gradients, [inputs] + list(module.trainable_variables))

    def testDynamicConfigurationChange(self):
        """测试模型在运行时动态改变配置（如内存大小、读取/写入数量）后的行为。"""
        # 初始配置
        initial_memory_size = MEMORY_SIZE
        initial_num_reads = NUM_READS
        initial_num_writes = NUM_WRITES

        # 创建模块，并传递 write_content_weights_fn
        module = MemoryAccess(
            initial_memory_size,
            WORD_SIZE,
            initial_num_reads,
            initial_num_writes,
            write_content_weights_fn=self.write_content_weights_fn
        )

        # 初始化状态，使用字典替代 TemporalLinkageState
        initial_state = AccessState(
            memory=tf.zeros([BATCH_SIZE, initial_memory_size, WORD_SIZE], dtype=tf.float32),
            read_weights=tf.zeros([BATCH_SIZE, 1, initial_num_reads, initial_memory_size], dtype=tf.float32),
            write_weights=tf.zeros([BATCH_SIZE, 1, initial_num_writes, initial_memory_size], dtype=tf.float32),
            linkage={
                'link': tf.zeros([BATCH_SIZE, initial_num_writes, initial_memory_size, initial_memory_size], dtype=tf.float32),
                'precedence_weights': tf.zeros([BATCH_SIZE, initial_num_writes, initial_memory_size], dtype=tf.float32)
            },
            usage=tf.zeros([BATCH_SIZE, initial_memory_size], dtype=tf.float32),
            read_words=tf.zeros([BATCH_SIZE, initial_num_reads, WORD_SIZE], dtype=tf.float32)  # 确保 read_words 被正确初始化
        )

        # 运行前向传播，调整维度顺序为 [batch_size, time_steps, input_size]
        inputs = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
        targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, initial_num_reads, WORD_SIZE])
        loss, gradients = self._run_forward_pass(module, initial_state, inputs, targets, track_gradients=True)
        self._check_gradients(gradients, [inputs] + list(module.trainable_variables))

        # 动态改变配置
        new_memory_size = 30
        new_num_reads = 3
        new_num_writes = 4

        # 创建新的模块，并传递 write_content_weights_fn
        new_module = MemoryAccess(
            new_memory_size,
            WORD_SIZE,
            new_num_reads,
            new_num_writes,
            write_content_weights_fn=self.write_content_weights_fn
        )

        # 初始化新的状态，使用字典替代 TemporalLinkageState
        new_initial_state = AccessState(
            memory=tf.zeros([BATCH_SIZE, new_memory_size, WORD_SIZE], dtype=tf.float32),
            read_weights=tf.zeros([BATCH_SIZE, 1, new_num_reads, new_memory_size], dtype=tf.float32),
            write_weights=tf.zeros([BATCH_SIZE, 1, new_num_writes, new_memory_size], dtype=tf.float32),
            linkage={
                'link': tf.zeros([BATCH_SIZE, new_num_writes, new_memory_size, new_memory_size], dtype=tf.float32),
                'precedence_weights': tf.zeros([BATCH_SIZE, new_num_writes, new_memory_size], dtype=tf.float32)
            },
            usage=tf.zeros([BATCH_SIZE, new_memory_size], dtype=tf.float32),
            read_words=tf.zeros([BATCH_SIZE, new_num_reads, WORD_SIZE], dtype=tf.float32)  # 确保 read_words 被正确初始化
        )

        # 创建新的输入和目标，调整维度顺序为 [batch_size, time_steps, input_size]
        new_targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, new_num_reads, WORD_SIZE])

        # 运行前向传播
        loss, gradients = self._run_forward_pass(new_module, new_initial_state, inputs, new_targets, track_gradients=True)
        self._check_gradients(gradients, [inputs] + list(new_module.trainable_variables))


if __name__ == '__main__':
    tf.test.main()