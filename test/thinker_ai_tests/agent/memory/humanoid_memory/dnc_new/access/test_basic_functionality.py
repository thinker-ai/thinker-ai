import tensorflow as tf
import os

from thinker_ai.agent.memory.humanoid_memory.dnc_new.access import MemoryAccess

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


class MemoryAccessBasicFunctionalityTest(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessBasicFunctionalityTest, self).setUp()

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
            # 检查 inputs 中是否包含 'usage' 键，并根据 'usage' 的 batch size 推导出 batch_size
            if 'usage' in inputs:
                batch_size = tf.shape(inputs['usage'])[0]
            else:
                raise KeyError("The input 'usage' is missing in the inputs dictionary.")

            # 根据输入 'usage' 大小生成 write_content_weights
            logits = tf.ones([batch_size, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
            tf.print("Write Content Weights Shape:", tf.shape(logits))
            return logits  # 移除 softmax

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
    def testBuildAndTrain(self):
        """测试模块的构建和基本训练过程。"""
        # 生成随机输入
        inputs = tf.random.normal([BATCH_SIZE, TIME_STEPS, INPUT_SIZE])

        optimizer = tf.optimizers.SGD(learning_rate=1.0)

        with tf.GradientTape() as tape:
            output = self.module({
                'inputs': inputs,
                'prev_state': self.initial_state
            }, training=True)
            read_words = output['read_words']
            final_state = output['final_state']

            loss = tf.reduce_mean(read_words)

        gradients = tape.gradient(loss, self.module.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

        self.assertIsNotNone(final_state)
        self.assertIsNotNone(read_words)
        self.assertEqual(read_words.shape, (BATCH_SIZE, TIME_STEPS, NUM_READS, WORD_SIZE))
        tf.debugging.assert_greater_equal(final_state.usage, 0.0)
        tf.debugging.assert_less_equal(final_state.usage, 1.0)

    def testEdgeCaseInputs(self):
        """测试边缘情况输入（全零输入）的梯度流动。"""
        inputs = tf.zeros([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32)
        targets = tf.zeros([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], dtype=tf.float32)

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]

                # 正确获取字典中的 'read_words' 和 'final_state'
                result = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                }, training=True)

                # 从字典中提取 'read_words' 和 'final_state'
                output = result['read_words']
                prev_state = result['final_state']

                # 压缩 output
                output = tf.squeeze(output, axis=1)
                outputs.append(output)

            output = tf.stack(outputs, axis=0)
            loss = tf.reduce_mean(tf.square(output - targets))

        gradients = tape.gradient(loss, self.module.trainable_variables)
        for var, grad in zip(self.module.trainable_variables, gradients):
            if grad is None:
                tf.print(f"Gradient for variable '{var.name}' is None")
            else:
                grad_norm = tf.norm(grad)
                tf.print(f"Gradient norm for variable '{var.name}':", grad_norm)
                if 'write_vectors' in var.name:
                    self.assertGreater(grad_norm, 1e-12)

    def testNonEdgeCaseInputs(self):
        """测试非边缘情况输入（随机输入）的训练和梯度流动。"""
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32)
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], dtype=tf.float32)

        optimizer = tf.optimizers.SGD(learning_rate=1.0)

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                }, training=True)
                read_words = output['read_words']
                final_state = output['final_state']
                outputs.append(read_words)
                prev_state = final_state

            loss = tf.reduce_mean(tf.stack(outputs))

        gradients = tape.gradient(loss, self.module.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

        self.assertIsNotNone(prev_state)
        self.assertIsNotNone(read_words)
        self.assertEqual(read_words.shape, (BATCH_SIZE, SEQUENCE_LENGTH, NUM_READS, WORD_SIZE))
        self.assertEqual(final_state.usage.shape, (BATCH_SIZE, MEMORY_SIZE))
        tf.debugging.assert_greater_equal(final_state.usage, 0.0)
        tf.debugging.assert_less_equal(final_state.usage, 1.0)