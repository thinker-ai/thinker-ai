# test_memory_access_gradients.py

import tensorflow as tf
import numpy as np
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
TIME_STEPS = 4  # 模拟运行的时间步数
SEQUENCE_LENGTH = 1  # 输入序列的初始长度
INPUT_SIZE = 12  # 输入大小
EPSILON = 1e-6
default_config = get_default_config(
    memory_size=MEMORY_SIZE,
    num_writes=NUM_WRITES,
    num_reads=NUM_READS,
    word_size=WORD_SIZE
)  # 使用动态配置


class MemoryAccessGradientRelatedTest(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessGradientRelatedTest, self).setUp()

        # 初始化 MemoryAccess 模块，传入配置
        self.module = MemoryAccess(
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_reads=NUM_READS,
            num_writes=NUM_WRITES,
            epsilon=EPSILON,
            name='memory_access_custom_plugin',
            config=default_config  # 传入默认配置
        )

        # 初始化状态
        batch_size_tensor = tf.constant(BATCH_SIZE, dtype=tf.int32)  # 将 batch_size 转换为 tf.Tensor
        self.initial_state = self.module.get_initial_state(batch_size=batch_size_tensor, initial_time_steps=1)

    def test_initial_state(self):
        """
        测试 get_initial_state 方法，确保返回正确的初始状态。
        """
        batch_size = 4
        batch_size_tensor = tf.constant(batch_size, dtype=tf.int32)  # 将 batch_size 转换为 tf.Tensor
        initial_state = self.module.get_initial_state(batch_size=batch_size_tensor, initial_time_steps=1)

        expected_memory = tf.zeros([batch_size, self.module.memory_size, self.module.word_size], dtype=tf.float32)
        expected_read_weights = tf.zeros([batch_size, 1, self.module.num_reads, self.module.memory_size],
                                         dtype=tf.float32)
        expected_write_weights = tf.zeros([batch_size, 1, self.module.num_writes, self.module.memory_size],
                                          dtype=tf.float32)
        expected_link = tf.zeros([batch_size, self.module.num_writes, self.module.memory_size, self.module.memory_size],
                                 dtype=tf.float32)
        expected_precedence_weights = tf.zeros([batch_size, self.module.num_writes, self.module.memory_size],
                                               dtype=tf.float32)
        expected_usage = tf.zeros([batch_size, self.module.memory_size], dtype=tf.float32)
        expected_read_words = tf.zeros([batch_size, self.module.num_reads, self.module.word_size], dtype=tf.float32)

        self.assertAllClose(initial_state.memory.numpy(), expected_memory.numpy(), atol=1e-6)
        self.assertAllClose(initial_state.read_weights.numpy(), expected_read_weights.numpy(), atol=1e-6)
        self.assertAllClose(initial_state.write_weights.numpy(), expected_write_weights.numpy(), atol=1e-6)
        self.assertAllClose(initial_state.linkage['link'].numpy(), expected_link.numpy(), atol=1e-6)
        self.assertAllClose(initial_state.linkage['precedence_weights'].numpy(), expected_precedence_weights.numpy(),
                            atol=1e-6)
        self.assertAllClose(initial_state.usage.numpy(), expected_usage.numpy(), atol=1e-6)
        self.assertAllClose(initial_state.read_words.numpy(), expected_read_words.numpy(), atol=1e-6)

    def testGradients_SingleConfiguration(self):
        """测试单一配置下模型的梯度流动情况。"""
        # 输入形状应为 [batch_size, time_steps, input_size]
        inputs = tf.Variable(np.random.randn(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32, trainable=True)
        targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

        with tf.GradientTape() as tape:
            output = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
            loss = tf.reduce_mean(tf.square(output['read_words'] - targets))

        gradients = tape.gradient(loss, [inputs] + list(self.module.trainable_variables))
        self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))

    def testSimpleGradient(self):
        """测试简单模型的梯度计算。"""
        simple_layer = tf.keras.layers.Dense(units=10, activation='relu')
        inputs = tf.Variable(tf.random.normal([2, 5]), trainable=True)
        targets = tf.random.normal([2, 10])

        with tf.GradientTape() as tape:
            outputs = simple_layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs - targets))

        gradients = tape.gradient(loss, [inputs] + simple_layer.trainable_variables)

        # 检查梯度
        for grad, var in zip(gradients, [inputs] + simple_layer.trainable_variables):
            var_name = var.name
            if grad is None:
                self.fail(f"Gradient for variable '{var_name}' is None.")
            else:
                grad_norm = tf.norm(grad).numpy()
                self.assertGreater(grad_norm, 1e-12, f"Gradient for variable '{var_name}' is too small.")
                self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var_name}' is too large.")

    def testGradientRange_SingleConfiguration(self):
        """测试单一配置下梯度范数是否在合理范围内。"""
        # 输入形状为 [batch_size, time_steps, input_size]
        inputs = tf.Variable(tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE]), dtype=tf.float32, trainable=True)
        targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

        with tf.GradientTape() as tape:
            output = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
            read_words = output['read_words']  # [batch_size, time_steps, num_reads, word_size]
            loss = tf.reduce_mean(tf.square(read_words - targets))
            tf.print("Single Configuration Test: Loss value:", loss)
            tf.print("Single Configuration Test: Output values (sample):", read_words[:2, :2, :])

        gradients = tape.gradient(loss, [inputs] + list(self.module.trainable_variables))
        tf.print("\nAll trainable variables (Single Configuration):")
        for var in self.module.trainable_variables:
            tf.print("Variable:", var.name, ", Shape:", var.shape)

        for grad, var in zip(gradients, [inputs] + list(self.module.trainable_variables)):
            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
            if grad is None:
                tf.print(f"Gradient for {var_name} is None.")
            else:
                grad_norm = tf.norm(grad)
                tf.print(f"Variable: {var_name}, Gradient Norm: {grad_norm.numpy()}")
                self.assertIsNotNone(grad, f"Gradient is None for variable {var_name}")
                self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var_name}' is too large")
                self.assertGreater(grad_norm, 1e-12, f"Gradient for variable '{var_name}' is too small")

    def testGradients_VaryingMemorySize(self):
        """测试不同内存大小下的梯度流动情况。"""
        MEMORY_SIZE_OPTIONS = [20, 50]
        WORD_SIZE = 6
        NUM_READS = 2
        NUM_WRITES = 3

        for MEMORY_SIZE in MEMORY_SIZE_OPTIONS:
            tf.print(
                f"Testing with MEMORY_SIZE={MEMORY_SIZE}, WORD_SIZE={WORD_SIZE}, NUM_READS={NUM_READS}, NUM_WRITES={NUM_WRITES}")

            # 更新配置以匹配新的 MEMORY_SIZE
            config = default_config.copy()
            config['WriteWeightCalculator'] = config['WriteWeightCalculator'].copy()
            config['WriteWeightCalculator']['memory_size'] = MEMORY_SIZE
            config['TemporalLinkageUpdater'] = config['TemporalLinkageUpdater'].copy()
            config['TemporalLinkageUpdater']['memory_size'] = MEMORY_SIZE
            config['UsageUpdater'] = config['UsageUpdater'].copy()
            config['UsageUpdater']['memory_size'] = MEMORY_SIZE
            config['ReadWeightCalculator'] = config['ReadWeightCalculator'].copy()
            config['ReadWeightCalculator']['num_writes'] = NUM_WRITES  # 确保 num_writes 与 WriteWeightCalculator 一致

            # 初始化 MemoryAccess 模块，传入更新后的配置
            module = MemoryAccess(
                memory_size=MEMORY_SIZE,
                word_size=WORD_SIZE,
                num_reads=NUM_READS,
                num_writes=NUM_WRITES,
                epsilon=EPSILON,
                name=f'memory_access_custom_memory_size_{MEMORY_SIZE}',
                config=config  # 传入更新后的配置
            )
            # 获取初始状态
            batch_size_tensor = tf.constant(BATCH_SIZE, dtype=tf.int32)
            initial_state = module.get_initial_state(batch_size=batch_size_tensor, initial_time_steps=1)

            # 生成虚拟输入序列和目标
            input_sequence = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
            targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

            # 使用 GradientTape 记录操作
            with tf.GradientTape() as tape:
                # 执行前向传播
                output = module({'inputs': input_sequence, 'prev_state': initial_state}, training=True)
                read_words = output['read_words']  # [batch_size, time_steps, num_reads, word_size]
                loss = tf.reduce_mean(tf.square(read_words - targets))
                tf.print("Testing Varying Memory Size: Loss value:", loss)
                tf.print("Testing Varying Memory Size: Output values (sample):", read_words[:2, :2, :])

            # 获取所有可训练变量
            tensors_to_check = module.trainable_variables

            # 计算梯度
            gradients = tape.gradient(loss, tensors_to_check)

            # 检查梯度
            for grad, var in zip(gradients, tensors_to_check):
                var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
                if grad is None:
                    tf.print(f"Gradient for {var_name} is None.")
                    self.fail(f"Gradient for {var_name} is None, test failed.")
                else:
                    grad_norm = tf.norm(grad).numpy()
                    if grad_norm == 0.0:
                        tf.print(f"Warning: Gradient for {var_name} is zero.")
                        self.fail(f"Gradient for {var_name} is zero, test failed.")
                    else:
                        tf.print(f"Variable: {var_name}, Gradient Norm: {grad_norm}")

    def _check_gradients(self, gradients, variables):
        """辅助方法：检查梯度是否存在且在合理范围内。"""
        for grad, var in zip(gradients, variables):
            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
            if grad is None:
                self.fail(f"Gradient for {var_name} is None.")
            else:
                grad_norm = tf.norm(grad).numpy()
                self.assertGreater(grad_norm, 1e-12, f"Gradient for variable '{var_name}' is too small.")
                self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var_name}' is too large.")


if __name__ == '__main__':
    tf.test.main()
