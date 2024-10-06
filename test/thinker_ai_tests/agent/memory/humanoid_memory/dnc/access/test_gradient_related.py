# test_gradient_related.py

import tensorflow as tf
import numpy as np
import os
import copy  # 导入 copy 模块以进行深拷贝

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import get_default_config
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 定义全局测试常量（不应在测试函数中修改）
BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4  # 模拟运行的时间步数
SEQUENCE_LENGTH = 1  # 输入序列的初始长度
INPUT_SIZE = 12  # 输入大小
EPSILON = 1e-6

# 获取默认配置
default_config = get_default_config(
    memory_size=MEMORY_SIZE,
    num_writes=NUM_WRITES,
    num_reads=NUM_READS,
    word_size=WORD_SIZE
)


class MemoryAccessGradientRelatedTest(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessGradientRelatedTest, self).setUp()

        # 初始化 MemoryAccess 模块，使用全局常量
        self.module = MemoryAccess(
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_reads=NUM_READS,
            num_writes=NUM_WRITES,
            epsilon=EPSILON,
            name='memory_access_test',
            config=default_config
        )

        # 初始化状态
        batch_size_tensor = tf.constant(BATCH_SIZE, dtype=tf.int32)
        self.initial_state = self.module.get_initial_state(batch_size=batch_size_tensor, initial_time_steps=1)

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
            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
            if grad is None:
                self.fail(f"Gradient for variable '{var_name}' is None.")
            else:
                grad_norm = tf.norm(grad).numpy()
                self.assertGreater(grad_norm, 1e-12, f"Gradient for variable '{var_name}' is too small.")
                self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var_name}' is too large.")

    def testGradientRange_SingleConfiguration(self):
        """测试单一配置下梯度范数是否在合理范围内。"""
        # 使用全局常量
        inputs = tf.Variable(tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE]), trainable=True)
        targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

        with tf.GradientTape() as tape:
            output = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
            read_words = output['read_words']  # [batch_size, time_steps, num_reads, word_size]
            loss = tf.reduce_mean(tf.square(read_words - targets))
            tf.print("Single Configuration Test: Loss value:", loss)

        gradients = tape.gradient(loss, [inputs] + self.module.trainable_variables)

        for grad, var in zip(gradients, [inputs] + self.module.trainable_variables):
            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
            if grad is None:
                tf.print(f"Gradient for {var_name} is None.")
            else:
                grad_norm = tf.norm(grad)
                tf.print(f"Variable: {var_name}, Gradient Norm: {grad_norm.numpy()}")
                self.assertIsNotNone(grad, f"Gradient is None for variable {var_name}")
                self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var_name}' is too large")
                self.assertGreater(grad_norm, 1e-12, f"Gradient for variable '{var_name}' is too small")

    def testGradients_SingleConfiguration(self):
        """测试单一配置下的梯度流动情况。"""
        # 输入形状应为 [batch_size, time_steps, input_size]
        inputs = tf.Variable(np.random.randn(BATCH_SIZE, TIME_STEPS, INPUT_SIZE), dtype=tf.float32, trainable=True)
        targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

        with tf.GradientTape() as tape:
            output = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
            loss = tf.reduce_mean(tf.square(output['read_words'] - targets))

        gradients = tape.gradient(loss, [inputs] + self.module.trainable_variables)
        self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))

    def testGradientRange_SingleConfiguration_Extended(self):
        """扩展测试以检查单一配置下的梯度范数。"""
        # 与 testGradientRange_SingleConfiguration 类似
        inputs = tf.Variable(tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE]), trainable=True)
        targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

        with tf.GradientTape() as tape:
            output = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
            read_words = output['read_words']  # [batch_size, time_steps, num_reads, word_size]
            loss = tf.reduce_mean(tf.square(read_words - targets))
            tf.print("Extended Single Configuration Test: Loss value:", loss)

        gradients = tape.gradient(loss, [inputs] + list(self.module.trainable_variables))
        tf.print("\nAll trainable variables (Extended Single Configuration):")
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
        memory_size_options = [20, 50]
        for memory_size in memory_size_options:
            tf.print(f"\nTesting with MEMORY_SIZE={memory_size}")

            # 使用深拷贝以避免修改全局 default_config
            config = copy.deepcopy(default_config)
            config['WriteWeightCalculator']['memory_size'] = memory_size
            config['UsageUpdater']['memory_size'] = memory_size
            config['TemporalLinkageUpdater']['memory_size'] = memory_size
            config['ContentWeightCalculator']['memory_size'] = memory_size

            # 初始化 MemoryAccess 模块，使用局部变量
            module = MemoryAccess(
                memory_size=memory_size,
                word_size=WORD_SIZE,
                num_reads=NUM_READS,
                num_writes=NUM_WRITES,
                epsilon=EPSILON,
                config=config
            )

            # 获取初始状态
            batch_size_tensor = tf.constant(BATCH_SIZE, dtype=tf.int32)
            initial_state = module.get_initial_state(batch_size=batch_size_tensor, initial_time_steps=1)

            # 使用全局常量
            input_sequence = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
            targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

            with tf.GradientTape() as tape:
                # 前向传播
                output = module({'inputs': input_sequence, 'prev_state': initial_state}, training=True)
                read_words = output['read_words']

                # 计算损失
                loss = tf.reduce_mean(tf.square(read_words - targets))

            # 获取可训练变量
            tensors_to_check = module.trainable_variables

            # 计算梯度
            gradients = tape.gradient(loss, tensors_to_check)

            # 检查梯度
            for grad, var in zip(gradients, tensors_to_check):
                var_name = var.name
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

    def testGradients_VaryingNumReads(self):
        """测试不同读取头数量下的梯度流动情况。"""
        num_reads_options = [2, 4]
        for num_reads in num_reads_options:
            tf.print(f"\nTesting with NUM_READS={num_reads}")

            # 使用深拷贝以避免修改全局 default_config
            config = copy.deepcopy(default_config)
            config['ReadWeightCalculator']['num_reads'] = num_reads
            config['UsageUpdater']['num_reads'] = num_reads
            config['ContentWeightCalculator']['num_reads'] = num_reads

            # 初始化 MemoryAccess 模块，使用局部变量
            module = MemoryAccess(
                memory_size=MEMORY_SIZE,
                word_size=WORD_SIZE,
                num_reads=num_reads,
                num_writes=NUM_WRITES,
                epsilon=EPSILON,
                config=config
            )

            # 获取初始状态
            batch_size_tensor = tf.constant(BATCH_SIZE, dtype=tf.int32)
            initial_state = module.get_initial_state(batch_size=batch_size_tensor, initial_time_steps=1)

            # 使用全局常量
            input_sequence = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
            targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, num_reads, WORD_SIZE])

            with tf.GradientTape() as tape:
                # 前向传播
                output = module({'inputs': input_sequence, 'prev_state': initial_state}, training=True)
                read_words = output['read_words']

                # 计算损失
                loss = tf.reduce_mean(tf.square(read_words - targets))

            # 获取可训练变量
            tensors_to_check = module.trainable_variables

            # 计算梯度
            gradients = tape.gradient(loss, tensors_to_check)

            # 检查梯度
            for grad, var in zip(gradients, tensors_to_check):
                var_name = var.name
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

    def testGradients_VaryingNumWrites(self):
        """测试不同写入头数量下的梯度流动情况。"""
        num_writes_options = [3, 5]
        for num_writes in num_writes_options:
            tf.print(f"\nTesting with NUM_WRITES={num_writes}")

            # 使用深拷贝以避免修改全局 default_config
            config = copy.deepcopy(default_config)
            config['WriteWeightCalculator']['num_writes'] = num_writes
            config['ReadWeightCalculator']['num_writes'] = num_writes
            config['UsageUpdater']['num_writes'] = num_writes
            config['TemporalLinkageUpdater']['num_writes'] = num_writes
            config['ContentWeightCalculator']['num_writes'] = num_writes

            # 初始化 MemoryAccess 模块，使用局部变量
            module = MemoryAccess(
                memory_size=MEMORY_SIZE,
                word_size=WORD_SIZE,
                num_reads=NUM_READS,
                num_writes=num_writes,
                epsilon=EPSILON,
                config=config
            )

            # 获取初始状态
            batch_size_tensor = tf.constant(BATCH_SIZE, dtype=tf.int32)
            initial_state = module.get_initial_state(batch_size=batch_size_tensor, initial_time_steps=1)

            # 使用全局常量
            input_sequence = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
            targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

            with tf.GradientTape() as tape:
                # 前向传播
                output = module({'inputs': input_sequence, 'prev_state': initial_state}, training=True)
                read_words = output['read_words']

                # 计算损失
                loss = tf.reduce_mean(tf.square(read_words - targets))

            # 获取可训练变量
            tensors_to_check = module.trainable_variables

            # 计算梯度
            gradients = tape.gradient(loss, tensors_to_check)

            # 检查梯度
            for grad, var in zip(gradients, tensors_to_check):
                var_name = var.name
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

    def testGradients_VaryingWordSize(self):
        """测试不同词向量大小下的梯度流动情况。"""
        word_size_options = [6, 12]
        for word_size in word_size_options:
            tf.print(f"\nTesting with WORD_SIZE={word_size}")

            # 使用深拷贝以避免修改全局 default_config
            config = copy.deepcopy(default_config)
            config['ContentWeightCalculator']['word_size'] = word_size

            # 初始化 MemoryAccess 模块，使用局部变量
            module = MemoryAccess(
                memory_size=MEMORY_SIZE,
                word_size=word_size,
                num_reads=NUM_READS,
                num_writes=NUM_WRITES,
                epsilon=EPSILON,
                config=config
            )

            # 获取初始状态
            batch_size_tensor = tf.constant(BATCH_SIZE, dtype=tf.int32)
            initial_state = module.get_initial_state(batch_size=batch_size_tensor, initial_time_steps=1)

            # 使用全局常量
            input_sequence = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
            targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, word_size])

            with tf.GradientTape() as tape:
                # 前向传播
                output = module({'inputs': input_sequence, 'prev_state': initial_state}, training=True)
                read_words = output['read_words']

                # 计算损失
                loss = tf.reduce_mean(tf.square(read_words - targets))

            # 获取可训练变量
            tensors_to_check = module.trainable_variables

            # 计算梯度
            gradients = tape.gradient(loss, tensors_to_check)

            # 检查梯度
            for grad, var in zip(gradients, tensors_to_check):
                var_name = var.name
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

    def testSublayersGradients(self):
        """Test gradient flow through all sublayers to ensure no gradient blockage."""
        # Generate random inputs and targets
        inputs = tf.Variable(tf.random.normal([BATCH_SIZE, TIME_STEPS, SEQUENCE_LENGTH, INPUT_SIZE]), dtype=tf.float32,
                             trainable=True)
        targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

        prev_state = self.initial_state

        with tf.GradientTape() as tape:
            tape.watch(self.module.trainable_variables)
            read_words_all = []
            for t in range(TIME_STEPS):
                input_t = inputs[:, t, :, :]  # Shape: [batch_size, sequence_length, input_size]
                # Call the module
                output = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                }, training=True)
                read_words_all.append(output['read_words'])  # [batch_size, num_reads, word_size]
                prev_state = output['final_state']

            # Stack read_words from all time steps
            read_words = tf.stack(read_words_all, axis=1)  # [batch_size, time_steps, num_reads, word_size]

            # Define loss as mean squared error
            loss = tf.reduce_mean(tf.square(read_words - targets))

        # Compute gradients
        gradients = tape.gradient(loss, self.module.trainable_variables)

        # Check gradients
        for grad, var in zip(gradients, self.module.trainable_variables):
            self.assertIsNotNone(grad, f"Gradient for variable {var.name} is None.")
            grad_norm = tf.norm(grad).numpy()
            self.assertGreater(grad_norm, 1e-12, f"Gradient too small for variable {var.name}")
            self.assertLess(grad_norm, 1e3, f"Gradient too large for variable {var.name}")
            tf.print(f"Gradient norm for variable {var.name}: {grad_norm}")


if __name__ == '__main__':
    tf.test.main()
