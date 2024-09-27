import tensorflow as tf
import numpy as np
import os

from thinker_ai.agent.memory.humanoid_memory.dnc_new import access
from thinker_ai.agent.memory.humanoid_memory.dnc_new.access import MemoryAccess, AccessState

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


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


class MemoryAccessOptimizersGradientClippingTests(tf.test.TestCase):
    def setUp(self):
        super(MemoryAccessOptimizersGradientClippingTests, self).setUp()

        # 定义一个简单的 write_content_weights_fn，返回未归一化的 logits
        def write_content_weights_fn(inputs):
            batch_size = tf.shape(inputs['usage'])[0]
            # 返回全1张量作为 logits
            logits = tf.ones([batch_size, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
            tf.print("Write Content Weights Shape:", tf.shape(logits))
            return logits  # 移除 softmax

        # 将 write_content_weights_fn 设为类成员
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

        # 将 batch_shape 定义为标量 Tensor
        batch_shape = tf.constant(BATCH_SIZE, dtype=tf.int32)

        # 构建模块以初始化权重
        # 通过调用一次模块，Keras会自动构建子层
        dummy_input = {
            'inputs': tf.zeros([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32),
            'prev_state': self.module.get_initial_state(batch_shape=batch_shape, initial_time_steps=1)
        }
        _ = self.module(dummy_input, training=False)
        self.initial_state = self.module.get_initial_state(batch_shape=batch_shape, initial_time_steps=1)


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

    def testDifferentOptimizers(self):
        """测试模型在使用不同优化器时的训练表现。"""
        optimizers = {
            'SGD': tf.optimizers.SGD(learning_rate=0.1),
            'Adam': tf.optimizers.Adam(learning_rate=0.001),
            'RMSProp': tf.optimizers.RMSprop(learning_rate=0.001)
        }

        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        for opt_name, optimizer in optimizers.items():
            print(f"Testing with optimizer: {opt_name}")

            with tf.GradientTape() as tape:
                output = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
                loss = tf.reduce_mean(tf.square(output['read_words'] - targets))

            gradients = tape.gradient(loss, [inputs] + self.module.trainable_variables)

            # 检查梯度
            self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))

            # 应用梯度
            optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

    def testGradientClipping(self):
        """测试模型的梯度裁剪功能，确保梯度不会超过指定的范数。"""
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        optimizer = tf.optimizers.SGD(learning_rate=1.0)
        clip_norm = 1.0

        with tf.GradientTape() as tape:
            output = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
            loss = tf.reduce_mean(tf.square(output['read_words'] - targets))

        gradients = tape.gradient(loss, [inputs] + self.module.trainable_variables)

        # 应用梯度裁剪
        clipped_gradients = [tf.clip_by_norm(g, clip_norm) if g is not None else None for g in gradients]

        # 检查梯度裁剪
        for grad, var in zip(clipped_gradients, [inputs] + list(self.module.trainable_variables)):
            var_name = var.name
            if grad is not None:
                grad_norm = tf.norm(grad).numpy()
                self.assertLessEqual(grad_norm, clip_norm + 1e-6, "Gradient norm exceeds clipping threshold")
            else:
                self.fail(f"Gradient for variable '{var_name}' is None after clipping.")

        # 应用裁剪后的梯度
        optimizer.apply_gradients(zip(clipped_gradients, self.module.trainable_variables))

    def testWeightUpdates(self):
        """测试模型权重在训练过程中的更新，确保优化器能够正确地更新权重。"""
        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        inputs = tf.Variable(np.random.randn(TIME_STEPS, BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        # 记录初始权重
        initial_weights = [var.numpy().copy() for var in self.module.trainable_variables]

        with tf.GradientTape() as tape:
            output = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
            loss = tf.reduce_sum(output['read_words'])

        gradients = tape.gradient(loss, self.module.trainable_variables)

        # 应用梯度
        optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

        # 检查权重是否更新
        for initial, var in zip(initial_weights, self.module.trainable_variables):
            updated = var.numpy()
            self.assertFalse(np.array_equal(initial, updated), f"Variable '{var.name}' did not update.")


if __name__ == '__main__':
    tf.test.main()