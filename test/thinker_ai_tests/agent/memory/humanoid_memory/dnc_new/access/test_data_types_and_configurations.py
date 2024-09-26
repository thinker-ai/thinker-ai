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
SEQUENCE_LENGTH = 1
INPUT_SIZE = 12  # 输入大小
EPSILON = 1e-6


class MemoryAccessDataTypesAndConfigurationsTests(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessDataTypesAndConfigurationsTests, self).setUp()

        # 定义一个简单的 write_content_weights_fn，返回未归一化的 logits
        def write_content_weights_fn(inputs):
            batch_size = tf.shape(inputs['usage'])[0]
            # 返回全1张量作为 logits
            logits = tf.ones([batch_size, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
            return logits  # 移除 softmax

        # 初始化 MemoryAccess 模块，传入 write_content_weights_fn
        self.module = MemoryAccess(
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_reads=NUM_READS,
            num_writes=NUM_WRITES,
            epsilon=EPSILON,
            write_content_weights_fn=write_content_weights_fn  # 传入函数
        )

        # 将 batch_shape 定义为标量 Tensor
        batch_shape = tf.constant(BATCH_SIZE, dtype=tf.int32)

        # 构建模块以初始化权重
        # 通过调用一次模块，Keras会自动构建子层
        dummy_input = {
            'inputs': tf.zeros([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32),
            'prev_state': self.module.get_initial_state(batch_shape=batch_shape, initial_time_steps=1)
            # 设置 initial_time_steps=1
        }
        _ = self.module(dummy_input, training=False)
        self.initial_state = self.module.get_initial_state(batch_shape=batch_shape,
                                                           initial_time_steps=1)  # 设置 initial_time_steps=1
    def _run_forward_pass(self, inputs, targets, track_gradients=True):
        """
        辅助方法：运行前向传播并计算损失。
        """
        if track_gradients:
            with tf.GradientTape() as tape:
                output = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
                loss = tf.reduce_mean(tf.square(output['read_words'] - targets))
            gradients = tape.gradient(loss, [inputs] + self.module.trainable_variables)
            return loss, gradients
        else:
            output = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
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
        # 创建新的模块实例，使用 float16
        module_fp16 = MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES, dtype=tf.float16)

        # 初始化状态
        initial_state_fp16 = AccessState(
            memory=tf.Variable(tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE], dtype=tf.float16), name='memory_fp16'),
            read_weights=tf.Variable(tf.random.normal([BATCH_SIZE, NUM_READS, MEMORY_SIZE], dtype=tf.float16), name='read_weights_fp16'),
            write_weights=tf.Variable(tf.random.normal([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float16), name='write_weights_fp16'),
            linkage=access.TemporalLinkageState(
                link=tf.Variable(tf.random.normal([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float16), name='link_fp16'),
                precedence_weights=tf.Variable(tf.random.normal([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float16), name='precedence_weights_fp16')
            ),
            usage=tf.Variable(tf.random.normal([BATCH_SIZE, MEMORY_SIZE], dtype=tf.float16), name='usage_fp16')
        )

        # 创建输入和目标
        inputs = tf.Variable(np.random.randn(TIME_STEPS, BATCH_SIZE, INPUT_SIZE).astype(np.float16), dtype=tf.float16, trainable=True)
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], dtype=tf.float16)

        with tf.GradientTape() as tape:
            output = module_fp16({'inputs': inputs, 'prev_state': initial_state_fp16})
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

                # 创建新的模块实例
                module = MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)

                # 初始化随机状态
                initial_state = AccessState(
                    memory=tf.zeros([batch_size, MEMORY_SIZE, WORD_SIZE], dtype=tf.float32),
                    read_weights=tf.zeros([batch_size, NUM_READS, MEMORY_SIZE], dtype=tf.float32),
                    write_weights=tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32),
                    linkage=access.TemporalLinkageState(
                        link=tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
                        precedence_weights=tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
                    ),
                    usage=tf.zeros([batch_size, MEMORY_SIZE], dtype=tf.float32)
                )

                # 创建输入和目标
                inputs = tf.random.normal([time_steps, batch_size, INPUT_SIZE])
                targets = tf.random.normal([time_steps, batch_size, NUM_READS, WORD_SIZE])

                loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)

                # 检查梯度
                self._check_gradients(gradients, [inputs] + list(module.trainable_variables))

    def testDynamicConfigurationChange(self):
        """测试模型在运行时动态改变配置（如内存大小、读取/写入数量）后的行为。"""
        # 初始配置
        initial_memory_size = MEMORY_SIZE
        initial_num_reads = NUM_READS
        initial_num_writes = NUM_WRITES

        # 创建模块
        module = MemoryAccess(initial_memory_size, WORD_SIZE, initial_num_reads, initial_num_writes)

        # 初始化状态
        initial_state = AccessState(
            memory=tf.zeros([BATCH_SIZE, initial_memory_size, WORD_SIZE], dtype=tf.float32),
            read_weights=tf.zeros([BATCH_SIZE, initial_num_reads, initial_memory_size], dtype=tf.float32),
            write_weights=tf.zeros([BATCH_SIZE, initial_num_writes, initial_memory_size], dtype=tf.float32),
            linkage=access.TemporalLinkageState(
                link=tf.zeros([BATCH_SIZE, initial_num_writes, initial_memory_size, initial_memory_size], dtype=tf.float32),
                precedence_weights=tf.zeros([BATCH_SIZE, initial_num_writes, initial_memory_size], dtype=tf.float32)
            ),
            usage=tf.zeros([BATCH_SIZE, initial_memory_size], dtype=tf.float32)
        )

        # 运行前向传播
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, initial_num_reads, WORD_SIZE])
        loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
        self._check_gradients(gradients, [inputs] + list(module.trainable_variables))

        # 动态改变配置
        new_memory_size = 30
        new_num_reads = 3
        new_num_writes = 4
        module.memory_size = new_memory_size
        module.num_reads = new_num_reads
        module.num_writes = new_num_writes

        # 重新初始化模块以应用新配置
        module = MemoryAccess(new_memory_size, WORD_SIZE, new_num_reads, new_num_writes)

        # 更新初始状态
        initial_state = AccessState(
            memory=tf.zeros([BATCH_SIZE, new_memory_size, WORD_SIZE], dtype=tf.float32),
            read_weights=tf.zeros([BATCH_SIZE, new_num_reads, new_memory_size], dtype=tf.float32),
            write_weights=tf.zeros([BATCH_SIZE, new_num_writes, new_memory_size], dtype=tf.float32),
            linkage=access.TemporalLinkageState(
                link=tf.zeros([BATCH_SIZE, new_num_writes, new_memory_size, new_memory_size], dtype=tf.float32),
                precedence_weights=tf.zeros([BATCH_SIZE, new_num_writes, new_memory_size], dtype=tf.float32)
            ),
            usage=tf.zeros([BATCH_SIZE, new_memory_size], dtype=tf.float32)
        )

        # 运行前向传播
        new_targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, new_num_reads, WORD_SIZE])
        loss, gradients = self._run_forward_pass(inputs, targets=new_targets, track_gradients=True)
        self._check_gradients(gradients, [inputs] + list(module.trainable_variables))


if __name__ == '__main__':
    tf.test.main()