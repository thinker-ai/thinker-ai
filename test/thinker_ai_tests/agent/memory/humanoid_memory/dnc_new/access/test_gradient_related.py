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


class MemoryAccessGradientRelatedTests(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessGradientRelatedTests, self).setUp()
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

    def _run_forward_pass(self, inputs, targets, track_gradients=True):
        """
        辅助方法：运行前向传播并计算损失。
        """
        if track_gradients:
            with tf.GradientTape() as tape:
                output, final_state = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
                loss = tf.reduce_mean(tf.square(output['read_words'] - targets))
            gradients = tape.gradient(loss, [inputs] + self.module.trainable_variables)
            return loss, gradients
        else:
            output, final_state = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
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

    def testGradients_SingleConfiguration(self):
        """
        测试单一配置下模型的梯度流动情况。
        """
        inputs = tf.Variable(np.random.randn(TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE), dtype=tf.float32)
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                inputs_t = {'inputs': input_t, 'prev_state': prev_state}
                output, prev_state = self.module(inputs_t, training=True)
                output = tf.squeeze(output['read_words'], axis=1)
                outputs.append(output)

            output = tf.stack(outputs, axis=0)
            loss = tf.reduce_mean(tf.square(output - targets))
            tf.print("Single Configuration Test: Loss value:", loss)
            tf.print("Single Configuration Test: Output values (sample):", output[:2, :2, :])

        gradients = tape.gradient(loss, [inputs] + self.module.trainable_variables)
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

    def testGradients_VaryingConfigurations(self):
        """
        测试不同配置下模型的梯度流动情况。
        """
        MEMORY_SIZE_OPTIONS = [20, 50]
        WORD_SIZE_OPTIONS = [6, 12]
        NUM_READS_OPTIONS = [2, 4]
        NUM_WRITES_OPTIONS = [3, 5]

        for MEMORY_SIZE in MEMORY_SIZE_OPTIONS:
            for WORD_SIZE in WORD_SIZE_OPTIONS:
                for NUM_READS in NUM_READS_OPTIONS:
                    for NUM_WRITES in NUM_WRITES_OPTIONS:
                        print(
                            f"Testing with MEMORY_SIZE={MEMORY_SIZE}, WORD_SIZE={WORD_SIZE}, NUM_READS={NUM_READS}, NUM_WRITES={NUM_WRITES}")

                        # 初始化 MemoryAccess 模块
                        module = access.MemoryAccess(
                            memory_size=MEMORY_SIZE,
                            word_size=WORD_SIZE,
                            num_reads=NUM_READS,
                            num_writes=NUM_WRITES,
                            epsilon=EPSILON
                        )

                        # 获取初始状态
                        initial_state = module.get_initial_state(batch_shape=[BATCH_SIZE])

                        # 生成虚拟输入序列和目标
                        input_sequence = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])
                        targets = tf.random.normal([BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE])

                        # 使用 GradientTape 记录操作
                        with tf.GradientTape() as tape:
                            # 执行前向传播
                            state = initial_state
                            read_words_all = []
                            for t in range(TIME_STEPS):
                                input_t = input_sequence[:, t:t+1, :]  # [batch_size, 1, input_size]

                                # 调用模块
                                state = module({
                                    'inputs': input_t,
                                    'prev_state': state
                                }, training=True)

                                # 收集 read_words
                                read_words_all.append(state.read_words)  # [batch_size, num_reads, word_size]

                            # 拼接所有时间步的 read_words
                            read_words = tf.stack(read_words_all, axis=1)  # [batch_size, time_steps, num_reads, word_size]

                            # 计算损失
                            loss = tf.reduce_mean(tf.square(read_words - targets))

                        # 获取所有可训练变量
                        tensors_to_check = module.trainable_variables

                        # 计算梯度
                        gradients = tape.gradient(loss, tensors_to_check)

                        # 检查梯度
                        for grad, var in zip(gradients, tensors_to_check):
                            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
                            if grad is None:
                                print(f"Gradient for {var_name} is None.")
                                self.fail(f"Gradient for {var_name} is None, test failed.")
                            else:
                                grad_norm = tf.norm(grad).numpy()
                                if grad_norm == 0.0:
                                    print(f"Warning: Gradient for {var_name} is zero.")
                                    self.fail(f"Gradient for {var_name} is zero, test failed.")
                                else:
                                    print(f"Variable: {var_name}, Gradient Norm: {grad_norm}")

    def testGradientRange(self):
        """
        测试梯度范数是否在合理范围内。
        """
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE])

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, prev_state = self.module({'inputs': input_t, 'prev_state': prev_state}, training=True)
                output = tf.squeeze(output['read_words'], axis=1)
                outputs.append(output)

            output = tf.stack(outputs, axis=0)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, self.module.trainable_variables)

        for grad in gradients:
            if grad is not None:
                grad_norm = tf.norm(grad).numpy()
                self.assertGreater(grad_norm, 1e-12, "Gradient is too small")
                self.assertLess(grad_norm, 1e3, "Gradient is too large")

    def testWriteContentWeightsGradient(self):
        """
        测试 write_content_weights 计算过程中的梯度流动。
        """
        # 初始化输入变量作为 tf.Variable 以启用梯度计算
        memory = tf.Variable(tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE]))
        keys = tf.random.uniform([BATCH_SIZE, NUM_WRITES, WORD_SIZE])
        strengths = tf.random.uniform([BATCH_SIZE, NUM_WRITES])

        with tf.GradientTape() as tape:
            # 计算 write_content_weights
            write_content_weights = self.module.write_content_weights({
                'memory': memory,
                'keys': keys,
                'strengths': strengths
            })
            # 定义损失为 write_content_weights 的总和
            loss = tf.reduce_sum(write_content_weights)

        # 计算梯度
        gradients = tape.gradient(loss, memory)

        # 检查梯度
        self.assertIsNotNone(gradients, "No gradient for memory in write_content_weights")
        grad_norm = tf.norm(gradients).numpy()
        self.assertGreater(grad_norm, 1e-12, "Gradient too small for memory in write_content_weights")
        tf.print(f"Gradient norm for memory in write_content_weights: {grad_norm}")

    def testWriteGradient(self):
        """
        测试写操作中的梯度流动。
        """
        # 初始化输入变量作为 tf.Variable 以启用梯度计算
        memory = tf.Variable(tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE]))
        usage = tf.Variable(tf.random.uniform([BATCH_SIZE, MEMORY_SIZE]))
        allocation_gate = tf.random.uniform([BATCH_SIZE, NUM_WRITES])
        write_gate = tf.random.uniform([BATCH_SIZE, NUM_WRITES])
        write_content_keys = tf.random.uniform([BATCH_SIZE, NUM_WRITES, WORD_SIZE])
        write_content_strengths = tf.random.uniform([BATCH_SIZE, NUM_WRITES])

        inputs = {
            'allocation_gate': allocation_gate,
            'write_gate': write_gate,
            'write_content_keys': write_content_keys,
            'write_content_strengths': write_content_strengths
        }

        with tf.GradientTape() as tape:
            # 计算 write_content_weights 使用模块的层，放在 GradientTape 内部
            write_content_weights = self.module.write_content_weights({
                'memory': memory,  # [batch_size, memory_size, word_size]
                'keys': write_content_keys,  # [batch_size, num_writes, word_size]
                'strengths': write_content_strengths  # [batch_size, num_writes]
            })  # [batch_size, num_writes, memory_size]

            inputs['write_content_weights'] = write_content_weights

            # 调用 _compute_write_weights 方法
            write_weights = self.module._compute_write_weights(
                write_content_weights=inputs['write_content_weights'],
                allocation_gate=inputs['allocation_gate'],
                write_gate=inputs['write_gate'],
                prev_usage=usage,
                training=True  # 根据具体测试场景调整
            )
            # 定义损失为 write_weights 的总和
            loss = tf.reduce_sum(write_weights)

        # 计算梯度
        gradients = tape.gradient(loss, [memory, usage])

        # 检查梯度
        for grad, var_name in zip(gradients, ['memory', 'usage']):
            self.assertIsNotNone(grad, f"No gradient for {var_name}")
            grad_norm = tf.norm(grad).numpy()
            self.assertGreater(grad_norm, 1e-12, f"Gradient too small for {var_name}")
            tf.print(f"Gradient norm for {var_name}: {grad_norm}")

    def testReadWeightsGradient(self):
        """
        测试读取权重计算过程中的梯度流动。
        """
        # 初始化输入变量
        memory = tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])
        prev_read_weights = tf.random.uniform([BATCH_SIZE, NUM_READS, MEMORY_SIZE])
        link = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE])
        read_content_keys = tf.random.uniform([BATCH_SIZE, NUM_READS, WORD_SIZE])
        read_content_strengths = tf.random.uniform([BATCH_SIZE, NUM_READS])
        read_mode = tf.random.uniform([BATCH_SIZE, NUM_READS, 1 + 2 * NUM_WRITES])

        inputs = {
            'read_content_weights': read_content_keys,
            'read_content_strengths': read_content_strengths,
            'read_mode': read_mode,
        }

        with tf.GradientTape() as tape:
            tape.watch([memory, prev_read_weights, link])
            read_weights = self.module._read_weights(inputs, memory, prev_read_weights, link)
            loss = tf.reduce_sum(read_weights)

        gradients = tape.gradient(loss, [memory, prev_read_weights, link])

        for grad, var_name in zip(gradients, ['memory', 'prev_read_weights', 'link']):
            self.assertIsNotNone(grad, f"No gradient for {var_name}")
            grad_norm = tf.norm(grad).numpy()
            self.assertGreater(grad_norm, 1e-12, f"Gradient too small for {var_name}")
            tf.print(f"Gradient norm for {var_name}: {grad_norm}")

    def testSublayersGradients(self):
        """
        测试所有子层内部的梯度流动情况，确保没有梯度阻断。
        """
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE])
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        prev_state = self.initial_state

        for t in range(TIME_STEPS):
            input_t = inputs[t]  # 形状为 (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
            target_t = targets[t]

            with tf.GradientTape() as tape:
                tape.watch(self.module.trainable_variables)
                output, next_state = self.module({'inputs': input_t, 'prev_state': prev_state}, training=True)
                output = tf.squeeze(output['read_words'], axis=1)  # 假设 output 是一个张量
                loss = tf.reduce_mean((output - target_t) ** 2)

            gradients = tape.gradient(loss, self.module.trainable_variables)
            # 检查梯度是否为 None
            for grad, var in zip(gradients, self.module.trainable_variables):
                self.assertIsNotNone(grad, f"Gradient for variable {var.name} is None.")
                grad_norm = tf.norm(grad).numpy()
                self.assertGreater(grad_norm, 1e-12, f"Gradient too small for variable {var.name}")
                self.assertLess(grad_norm, 1e3, f"Gradient too large for variable {var.name}")
                tf.print(f"Gradient norm for variable {var.name}: {grad_norm}")

            # 更新 prev_state
            prev_state = next_state


if __name__ == '__main__':
    tf.test.main()