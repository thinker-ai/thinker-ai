import tensorflow as tf
import numpy as np
import os

from thinker_ai.agent.memory.humanoid_memory.dnc_new import access
from thinker_ai.agent.memory.humanoid_memory.dnc_new.access import MemoryAccess, AccessState

from parameterized import parameterized

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


class MemoryAccessParameterizedTests(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessParameterizedTests, self).setUp()
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

    @parameterized.expand([
        ("all_zero_inputs", tf.zeros, tf.zeros),
        ("random_inputs", tf.random.normal, tf.random.normal),
    ])
    def test_allocation_cases(self, name, input_fn, target_fn):
        """
        参数化测试：测试不同类型的输入（全零、随机）。
        """
        # 构建输入和目标
        inputs = input_fn([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32)
        targets = target_fn([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], dtype=tf.float32)

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]  # [BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]
                # 构建输入字典，包含 'inputs' 和 'prev_state'
                inputs_t = {'inputs': input_t, 'prev_state': prev_state}
                output = self.module(inputs_t, training=True)
                read_words = output['read_words']
                final_state = output['final_state']
                outputs.append(read_words)
                prev_state = final_state

            output = tf.stack(outputs, axis=0)  # [TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE]
            loss = tf.reduce_mean(tf.square(output - targets))
            tf.print(f"{name}: Loss value:", loss)
            tf.print(f"{name}: Output values (sample):", output[:2, :2, :])

        gradients = tape.gradient(loss, [inputs] + list(self.module.trainable_variables))

        # 打印所有可训练变量的名称和形状
        tf.print(f"\nAll trainable variables for {name}:")
        for var in self.module.trainable_variables:
            tf.print("Variable:", var.name, ", Shape:", var.shape)

        # 添加调试信息，打印变量名称和梯度范数
        for var, grad in zip(self.module.trainable_variables, gradients):
            if grad is None:
                tf.print(f"Gradient for variable '{var.name}' is None ({name})")
            else:
                grad_norm = tf.norm(grad)
                tf.print(f"Gradient norm for variable '{var.name}':", grad_norm)
                self.assertGreater(grad_norm, 1e-12,
                                   f"Gradient for variable '{var.name}' is too small for {name}")
                self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var.name}' is too large for {name}")


if __name__ == '__main__':
    tf.test.main()