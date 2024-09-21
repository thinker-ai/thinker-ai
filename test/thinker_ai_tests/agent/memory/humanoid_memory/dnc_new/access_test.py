#thinker_ai_tests/agent/memory/humanoid_memory/dnc/access_test.py
import os
from collections import namedtuple

from thinker_ai.agent.memory.humanoid_memory.dnc_new.access import MemoryAccess, AccessState
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import TemporalLinkageState

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息
import numpy as np
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc_new import access, addressing
from thinker_ai.agent.memory.humanoid_memory.dnc_new import util
BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4
INPUT_SIZE = 10


class MemoryAccessTest(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessTest, self).setUp()
        self.module = access.MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)

        # 初始化 prev_state 时使用 Tensor 而不是 Variable
        self.initial_state = access.AccessState(
            memory=tf.zeros([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE], dtype=tf.float32),
            read_weights=tf.zeros([BATCH_SIZE, NUM_READS, MEMORY_SIZE], dtype=tf.float32),
            write_weights=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32),
            linkage=addressing.TemporalLinkageState(
                link=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
                precedence_weights=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
            ),
            usage=tf.zeros([BATCH_SIZE, MEMORY_SIZE], dtype=tf.float32)
        )

    def testBuildAndTrain(self):
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])

        optimizer = tf.optimizers.SGD(learning_rate=1.0)

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, prev_state = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                })
                outputs.append(output)

            output = tf.stack(outputs, axis=0)

            targets = tf.random.uniform([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
            loss = tf.reduce_mean(tf.square(output - targets))

        gradients = tape.gradient(loss, self.module.trainable_variables)
        # 检查哪些梯度为 None
        for grad, var in zip(gradients, self.module.trainable_variables):
            if grad is None:
                print(f"No gradient provided for variable {var.name}")
        optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

    def testValidReadMode(self):
        # 构建输入字典，而不是直接传入 Tensor
        inputs = {
            'inputs': tf.random.normal([BATCH_SIZE, INPUT_SIZE]),
            'prev_state': self.initial_state
        }

        # 调用模块的 _read_inputs 方法
        processed_inputs = self.module._read_inputs(inputs)

        # 检查 processed_inputs 的内容
        self.assertIn('read_content_keys', processed_inputs)
        self.assertIn('read_content_strengths', processed_inputs)
        # 其他相关断言

    def testWriteWeights(self):
        memory = 10 * (np.random.rand(BATCH_SIZE, MEMORY_SIZE, WORD_SIZE) - 0.5)
        usage = np.random.rand(BATCH_SIZE, MEMORY_SIZE)

        allocation_gate = np.random.rand(BATCH_SIZE, NUM_WRITES)
        write_gate = np.random.rand(BATCH_SIZE, NUM_WRITES)
        write_content_keys = np.random.rand(BATCH_SIZE, NUM_WRITES, WORD_SIZE)
        write_content_strengths = np.random.rand(BATCH_SIZE, NUM_WRITES)

        usage[:, 3] = 0
        allocation_gate[:, 0] = 1
        write_gate[:, 0] = 1

        inputs = {
            'allocation_gate': tf.constant(allocation_gate, dtype=tf.float32),
            'write_gate': tf.constant(write_gate, dtype=tf.float32),
            'write_content_keys': tf.constant(write_content_keys, dtype=tf.float32),
            'write_content_strengths': tf.constant(write_content_strengths, dtype=tf.float32)
        }

        weights = self.module._write_weights(inputs, tf.constant(memory, dtype=tf.float32),
                                             tf.constant(usage, dtype=tf.float32))

        # 在 Eager Execution 下，直接获取数值
        weights = weights.numpy()

        self.assertAllClose(np.sum(weights, axis=2), write_gate, atol=5e-2)

        weights_0_0_target = util.one_hot(MEMORY_SIZE, 3)
        self.assertAllClose(weights[0, 0], weights_0_0_target, atol=1e-3)

    def testReadWeights(self):
        memory = 10 * (np.random.rand(BATCH_SIZE, MEMORY_SIZE, WORD_SIZE) - 0.5)
        prev_read_weights = np.random.rand(BATCH_SIZE, NUM_READS, MEMORY_SIZE)
        prev_read_weights /= prev_read_weights.sum(2, keepdims=True) + 1e-6

        link = np.random.rand(BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE)
        link /= np.maximum(link.sum(2, keepdims=True), 1e-6)
        link /= np.maximum(link.sum(3, keepdims=True), 1e-6)

        read_content_keys = np.random.rand(BATCH_SIZE, NUM_READS, WORD_SIZE)
        read_content_keys[0, 0] = memory[0, 3]
        read_content_strengths = tf.constant(100., shape=[BATCH_SIZE, NUM_READS], dtype=tf.float32)
        read_mode = np.random.rand(BATCH_SIZE, NUM_READS, 1 + 2 * NUM_WRITES)
        read_mode[0, 0, :] = util.one_hot(1 + 2 * NUM_WRITES, 2 * NUM_WRITES)

        inputs = {
            'read_content_keys': tf.constant(read_content_keys, dtype=tf.float32),
            'read_content_strengths': read_content_strengths,
            'read_mode': tf.constant(read_mode, dtype=tf.float32),
        }

        read_weights = self.module._read_weights(inputs, tf.constant(memory, dtype=tf.float32),
                                                 tf.constant(prev_read_weights, dtype=tf.float32),
                                                 tf.constant(link, dtype=tf.float32))

        # 在 Eager Execution 下，直接获取数值
        read_weights = read_weights.numpy()

        self.assertAllClose(read_weights[0, 0, :], util.one_hot(MEMORY_SIZE, 3), atol=1e-3)

    def testRandomInitialState(self):
        # 使用随机初始化状态而非全零状态
        random_state = access.AccessState(
            memory=tf.Variable(tf.random.uniform([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE]), name='memory'),
            read_weights=tf.Variable(tf.random.uniform([BATCH_SIZE, NUM_READS, MEMORY_SIZE]), name='read_weights'),
            write_weights=tf.Variable(tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE]), name='write_weights'),
            linkage=addressing.TemporalLinkageState(
                link=tf.Variable(tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE]), name='link'),
                precedence_weights=tf.Variable(tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE]),
                                               name='precedence_weights')
            ),
            usage=tf.Variable(tf.random.uniform([BATCH_SIZE, MEMORY_SIZE]), name='usage')
        )

        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])

        optimizer = tf.optimizers.SGD(learning_rate=1.0)

        with tf.GradientTape() as tape:
            prev_state = random_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, prev_state = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                })
                outputs.append(output)

            output = tf.stack(outputs, axis=0)

            targets = tf.random.uniform([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
            loss = tf.reduce_mean(tf.square(output - targets))

        gradients = tape.gradient(loss, self.module.trainable_variables)
        # 检查哪些梯度为 None
        for grad, var in zip(gradients, self.module.trainable_variables):
            self.assertIsNotNone(grad, f"No gradient provided for variable {var.name}")

    def testGradientRange(self):
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, prev_state = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                })
                outputs.append(output)

            output = tf.stack(outputs, axis=0)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, self.module.trainable_variables)

        for grad in gradients:
            if grad is not None:
                grad_norm = tf.norm(grad).numpy()
                self.assertGreater(grad_norm, 1e-12, "Gradient is too small")
                self.assertLess(grad_norm, 1e3, "Gradient is too large")

    def testWriteGradient(self):
        memory = tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])
        usage = tf.random.uniform([BATCH_SIZE, MEMORY_SIZE])
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
            tape.watch([memory, usage])
            write_weights = self.module._write_weights(inputs, memory, usage)
            loss = tf.reduce_sum(write_weights)

        gradients = tape.gradient(loss, [memory, usage])

        for grad, var_name in zip(gradients, ['memory', 'usage']):
            self.assertIsNotNone(grad, f"No gradient for {var_name}")
            self.assertGreater(tf.norm(grad).numpy(), 1e-12, f"Gradient too small for {var_name}")

    def testReadWeightsGradient(self):
        memory = tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])
        prev_read_weights = tf.random.uniform([BATCH_SIZE, NUM_READS, MEMORY_SIZE])
        link = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE])
        read_content_keys = tf.random.uniform([BATCH_SIZE, NUM_READS, WORD_SIZE])
        read_content_strengths = tf.random.uniform([BATCH_SIZE, NUM_READS])
        read_mode = tf.random.uniform([BATCH_SIZE, NUM_READS, 1 + 2 * NUM_WRITES])

        inputs = {
            'read_content_keys': read_content_keys,
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
            self.assertGreater(tf.norm(grad).numpy(), 1e-12, f"Gradient too small for {var_name}")

    def testEdgeCaseInputs(self):
        inputs = tf.zeros([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
        targets = tf.zeros([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, prev_state = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                })
                outputs.append(output)

            output = tf.stack(outputs, axis=0)
            loss = tf.reduce_mean(tf.square(output - targets))
            tf.print("Test: Loss value:", loss)
            tf.print("Test: Output values (sample):", output[:2, :])

        gradients = tape.gradient(loss, self.module.trainable_variables)

        # 打印所有可训练变量的名称和形状
        tf.print("\nAll trainable variables:")
        for var in self.module.trainable_variables:
            tf.print("Variable:", var.name, ", Shape:", var.shape)

        # 添加调试信息，打印变量名称和梯度范数
        for var, grad in zip(self.module.trainable_variables, gradients):
            if grad is None:
                tf.print(f"Gradient for variable '{var.name}' is None")
            else:
                grad_norm = tf.norm(grad)
                tf.print(f"Gradient norm for variable '{var.name}':", grad_norm)
                self.assertGreater(grad_norm, 1e-12,
                                   f"Gradient for variable '{var.name}' is too small for edge case inputs")
                self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var.name}' is too large for edge case inputs")
    def testSmallNonZeroInputs(self):
            inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE], mean=0.0, stddev=1e-3)
            targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], mean=0.0, stddev=1e-3)

            with tf.GradientTape() as tape:
                prev_state = self.initial_state
                outputs = []
                for t in range(TIME_STEPS):
                    input_t = inputs[t]
                    output, prev_state = self.module({
                        'inputs': input_t,
                        'prev_state': prev_state
                    })
                    outputs.append(output)

                output = tf.stack(outputs, axis=0)
                loss = tf.reduce_mean(tf.square(output - targets))
                print("Test Small Non-Zero: Loss value:", loss.numpy())
                print("Test Small Non-Zero: Output values (sample):", output.numpy()[0, :2, :])

            gradients = tape.gradient(loss, self.module.trainable_variables)

            # 添加调试信息，打印变量名称和梯度范数
            for var, grad in zip(self.module.trainable_variables, gradients):
                if grad is None:
                    print(f"Gradient for variable '{var.name}' is None")
                else:
                    grad_norm = tf.norm(grad).numpy()
                    print(f"Gradient norm for variable '{var.name}': {grad_norm}")

                    self.assertGreater(grad_norm, 1e-12,
                                       f"Gradient for variable '{var.name}' is too small for small non-zero inputs")
                    self.assertLess(grad_norm, 1e3,
                                    f"Gradient for variable '{var.name}' is too large for small non-zero inputs")

    def testNonZeroInputs(self):
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, prev_state = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                })
                outputs.append(output)

            output = tf.stack(outputs, axis=0)
            loss = tf.reduce_mean(tf.square(output - targets))
            print("Test Non-Zero: Loss value:", loss.numpy())

        gradients = tape.gradient(loss, self.module.trainable_variables)

        # 添加调试信息
        for i, grad in enumerate(gradients):
            if grad is None:
                print(f"Gradient for variable {i} is None")
            else:
                grad_norm = tf.norm(grad).numpy()
                print(f"Gradient norm for variable {i}: {grad_norm}")

                self.assertGreater(grad_norm, 1e-12, "Gradient is too small for non-zero inputs")
                self.assertLess(grad_norm, 1e3, "Gradient is too large for non-zero inputs")


    def testGradients_SingleConfiguration(self):
        # 将 inputs 定义为 tf.Variable 以确保梯度追踪
        inputs = tf.Variable(np.random.randn(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)

        # 将 inputs 和 initial_state 封装为字典传入
        input_dict = {
            'inputs': inputs,
            'prev_state': self.initial_state  # 初始状态作为字典键值对
        }

        with tf.GradientTape() as tape:
            # 不需要手动 watch，因为 inputs 是 tf.Variable
            output, _ = self.module(input_dict)

            # 使用与 testGradients_old 相同的损失函数
            loss = tf.reduce_sum(output)

            # 检查损失是否包含 NaN 或 Inf
            tf.debugging.check_numerics(loss, "Loss contains NaNs or Infs")

            # 打印损失值以便调试
            # print(f"Loss: {loss.numpy()}")

        # 打印 initial_state 的各部分
        # print("Initial State Variables:")
        # print(f"Memory: {self.initial_state.memory.numpy()}")
        # print(f"Read Weights: {self.initial_state.read_weights.numpy()}")
        # print(f"Write Weights: {self.initial_state.write_weights.numpy()}")
        # print(f"Link: {self.initial_state.linkage.link.numpy()}")
        # print(f"Precedence Weights: {self.initial_state.linkage.precedence_weights.numpy()}")
        # print(f"Usage: {self.initial_state.usage.numpy()}")

        # 包含所有相关变量以检查其梯度
        tensors_to_check = [inputs] + list(self.module.trainable_variables)

        for var in self.module.trainable_variables:
            if var.grad is not None:
                tf.debugging.check_numerics(var.grad, f"Gradient for {var.name} contains NaN or Inf")

        gradients = tape.gradient(loss, tensors_to_check)

        # 打印和检查每个变量的梯度
        for grad, var in zip(gradients, tensors_to_check):
            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
            if grad is None:
                print(f"Gradient for {var_name} is None.")
            else:
                grad_norm = tf.norm(grad)
                print(f"Variable: {var_name}, Gradient Norm: {grad_norm.numpy()}")

                # 额外检查和断言
                if grad_norm.numpy() == 0.0:
                    print(f"Warning: Gradient for {var_name} is zero.")
                self.assertIsNotNone(grad, f"Gradient is None for variable {var_name}")
                self.assertLess(grad_norm, 1e3)
                self.assertGreater(grad_norm, 1e-12)

    def testGradients_VaryingConfigurations(self):
        MEMORY_SIZE_OPTIONS = [20, 50]  # 不同的内存大小
        WORD_SIZE_OPTIONS = [6, 12]  # 不同的字大小
        NUM_READS_OPTIONS = [2, 4]  # 不同的读取数量
        NUM_WRITES_OPTIONS = [3, 5]  # 不同的写入数量

        for MEMORY_SIZE in MEMORY_SIZE_OPTIONS:
            for WORD_SIZE in WORD_SIZE_OPTIONS:
                for NUM_READS in NUM_READS_OPTIONS:
                    for NUM_WRITES in NUM_WRITES_OPTIONS:
                        print(f"Testing with MEMORY_SIZE={MEMORY_SIZE}, WORD_SIZE={WORD_SIZE}, "
                              f"NUM_READS={NUM_READS}, NUM_WRITES={NUM_WRITES}")

                        self.module = access.MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)

                        # Initialize prev_state with random values
                        self.initial_state = access.AccessState(
                            memory=tf.Variable(tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE]), dtype=tf.float32,
                                               name='memory'),
                            read_weights=tf.Variable(
                                tf.random.normal([BATCH_SIZE, NUM_READS, MEMORY_SIZE], dtype=tf.float32),
                                name='read_weights'),
                            write_weights=tf.Variable(
                                tf.random.normal([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32),
                                name='write_weights'),
                            linkage=addressing.TemporalLinkageState(
                                link=tf.Variable(
                                    tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
                                    name='link'),
                                precedence_weights=tf.Variable(
                                    tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32),
                                    name='precedence_weights')
                            ),
                            usage=tf.Variable(tf.zeros([BATCH_SIZE, MEMORY_SIZE], dtype=tf.float32), name='usage')
                        )

                        inputs = tf.Variable(np.random.randn(BATCH_SIZE, INPUT_SIZE), dtype=tf.float32, trainable=True)

                        input_dict = {
                            'inputs': inputs,
                            'prev_state': self.initial_state
                        }

                        with tf.GradientTape() as tape:
                            output, _ = self.module(input_dict)
                            loss = tf.reduce_sum(output)
                            tf.debugging.check_numerics(loss, "Loss contains NaNs or Infs")
                            print(f"Loss: {loss.numpy()}")

                        tensors_to_check = [inputs] + list(self.module.trainable_variables)

                        gradients = tape.gradient(loss, tensors_to_check)

                        # Debug output for gradient information
                        for grad, var in zip(gradients, tensors_to_check):
                            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
                            if grad is None:
                                print(f"Gradient for {var_name} is None.")
                                self.fail(f"Gradient for {var_name} is None, test failed.")
                            else:
                                grad_norm = tf.norm(grad)
                                if grad_norm.numpy() == 0.0:
                                    print(f"Warning: Gradient for {var_name} is zero.")
                                    print(f"Details for {var_name}: {var.numpy()}")
                                    self.fail(f"Gradient for {var_name} is zero, test failed.")
                                else:
                                    print(f"Variable: {var_name}, Gradient Norm: {grad_norm.numpy()}")
                                    self.assertIsNotNone(grad, f"Gradient is None for variable {var_name}")
                                    self.assertLess(grad_norm, 1e3, f"Gradient norm for {var_name} is too large.")
                                    self.assertGreater(grad_norm, 1e-12, f"Gradient norm for {var_name} is too small.")

if __name__ == '__main__':
    tf.test.main()
