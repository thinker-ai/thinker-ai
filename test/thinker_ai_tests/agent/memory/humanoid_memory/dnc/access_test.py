import numpy as np
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc import access, addressing
from thinker_ai.agent.memory.humanoid_memory.dnc import util

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

        # Initialize prev_state with actual tensors
        self.initial_state = access.AccessState(
            memory=tf.Variable(tf.zeros([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE], dtype=tf.float32)),
            read_weights=tf.Variable(tf.zeros([BATCH_SIZE, NUM_READS, MEMORY_SIZE], dtype=tf.float32)),
            write_weights=tf.Variable(tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)),
            linkage=addressing.TemporalLinkageState(
                link=tf.Variable(tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32)),
                precedence_weights=tf.Variable(tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32))
            ),
            usage=tf.Variable(tf.zeros([BATCH_SIZE, MEMORY_SIZE], dtype=tf.float32))
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

    def testGradients_old(self):
        inputs = tf.constant(np.random.randn(BATCH_SIZE, INPUT_SIZE), tf.float32)
        output, _ = self.module(inputs, self.initial_state)
        loss = tf.reduce_sum(output)

        tensors_to_check = [
            inputs, self.initial_state.memory, self.initial_state.read_weights,
            self.initial_state.linkage.precedence_weights,
            self.initial_state.linkage.link
        ]
        shapes = [x.get_shape().as_list() for x in tensors_to_check]
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            err = tf.test.compute_gradient_error(tensors_to_check, shapes, loss, [1])
            self.assertLess(err, 0.1)
    def testGradients(self):
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

            # 确保 output 的形状与 inputs 一致
            if output.shape != inputs.shape:
                flattened_output = tf.reshape(output, [BATCH_SIZE, -1])
                if flattened_output.shape[1] < INPUT_SIZE:
                    # 计算需要重复的次数以匹配 INPUT_SIZE
                    repeats = tf.math.ceil(INPUT_SIZE / tf.cast(flattened_output.shape[1], tf.float32))
                    repeated_output = tf.tile(flattened_output, [1, tf.cast(repeats, tf.int32)])
                    flattened_output = repeated_output[:, :INPUT_SIZE]
                else:
                    flattened_output = flattened_output[:, :INPUT_SIZE]
            else:
                flattened_output = output

            # 使用与 inputs 直接相关的损失函数
            loss = tf.reduce_mean(tf.square(inputs - flattened_output))

            # 检查损失是否包含 NaN 或 Inf
            tf.debugging.check_numerics(loss, "Loss contains NaNs or Infs")

            # 打印损失值以便调试
            print(f"Loss: {loss.numpy()}")

        # 包含 inputs 以检查其梯度
        tensors_to_check = [inputs] + list(self.module.trainable_variables)

        # 打印 trainable_variables 的名称
        print("Trainable variables in the module:")
        for var in self.module.trainable_variables:
            print(var.name)

        gradients = tape.gradient(loss, tensors_to_check)

        # 打印和检查每个变量的梯度
        for grad, var in zip(gradients, tensors_to_check):
            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
            print(f"Variable: {var_name}, Gradient: {'Exists' if grad is not None else 'None'}")
            if grad is not None:
                grad_norm = tf.norm(grad)
                print(f"Gradient norm for {var_name}: {grad_norm.numpy()}")
                if grad_norm.numpy() == 0.0:
                    print(f"Warning: Gradient for {var_name} is zero.")

            self.assertIsNotNone(grad, f"Gradient is None for variable {var_name}")
            if grad is not None:
                grad_norm = tf.norm(grad)
                self.assertLess(grad_norm, 1e3)
                self.assertGreater(grad_norm, 1e-12)


if __name__ == '__main__':
    tf.test.main()
