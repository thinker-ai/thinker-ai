import tensorflow as tf
import numpy as np
import os
import tempfile
import time
import threading

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


class MemoryAccessPerformanceIntegrationTests(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessPerformanceIntegrationTests, self).setUp()
        # 初始化 MemoryAccess 模块
        self.module = MemoryAccess(
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_reads=NUM_READS,
            num_writes=NUM_WRITES,
            epsilon=EPSILON
        )
        # 构建模块以初始化权重
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

    def testInferencePerformance(self):
        """测试模型在推理阶段的计算速度和内存占用。"""
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
        prev_state = self.initial_state

        # 测试多次前向传播以评估平均时间
        num_runs = 100
        start_time = time.time()

        for _ in range(num_runs):
            output, _ = self.module({'inputs': inputs, 'prev_state': prev_state}, training=False)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        print(f"Average inference time per run: {avg_time * 1000:.2f} ms")

        # 简单断言，确保推理时间在合理范围内（具体数值根据硬件而定）
        self.assertLess(avg_time, 0.1, "Inference time is too high.")

    def testParallelExecution(self):
        """测试模型在多线程环境下的并行执行能力。"""
        def run_test():
            inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
            targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
            loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
            self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))

        threads = []
        num_threads = 4
        for _ in range(num_threads):
            thread = threading.Thread(target=run_test)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def testIntegrationWithLargerModel(self):
        """测试 MemoryAccess 模块与更大的模型或系统的集成情况。"""
        # 假设有一个简单的 RNN 模型，集成 MemoryAccess
        class SimpleRNNWithMemory(tf.keras.Model):
            def __init__(self, memory_access):
                super(SimpleRNNWithMemory, self).__init__()
                self.memory_access = memory_access
                self.dense = tf.keras.layers.Dense(NUM_READS * WORD_SIZE, activation=None)

            def call(self, inputs, states):
                read_output, new_state = self.memory_access({
                    'inputs': inputs,
                    'prev_state': states
                })
                output = self.dense(read_output)
                return output, new_state

        # 创建 MemoryAccess 实例
        memory_access = MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
        model = SimpleRNNWithMemory(memory_access)

        # 初始化状态
        initial_state = AccessState(
            memory=tf.zeros([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE], dtype=tf.float32),
            read_weights=tf.zeros([BATCH_SIZE, NUM_READS, MEMORY_SIZE], dtype=tf.float32),
            write_weights=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32),
            linkage=access.TemporalLinkageState(
                link=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
                precedence_weights=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
            ),
            usage=tf.zeros([BATCH_SIZE, MEMORY_SIZE], dtype=tf.float32)
        )

        # 创建输入和目标
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS * WORD_SIZE])

        optimizer = tf.optimizers.Adam()

        with tf.GradientTape() as tape:
            states = initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, states = model(input_t, states)
                outputs.append(output)

            output = tf.stack(outputs, axis=0)
            loss = tf.reduce_mean(tf.square(output - targets))

        gradients = tape.gradient(loss, model.trainable_variables)

        # 检查梯度
        for grad, var in zip(gradients, model.trainable_variables):
            self.assertIsNotNone(grad, f"No gradient provided for variable {var.name}")
            grad_norm = tf.norm(grad).numpy()
            self.assertGreater(grad_norm, 1e-12, f"Gradient too small for {var.name}")
            self.assertLess(grad_norm, 1e3, f"Gradient too large for {var.name}")

        # 应用梯度
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def testLongTermDependencies(self):
        """测试模型在处理具有长期依赖性的序列数据时的性能。"""
        long_time_steps = 100  # 增加时间步长以模拟长期依赖
        inputs = tf.random.normal([long_time_steps, BATCH_SIZE, INPUT_SIZE])
        targets = tf.random.normal([long_time_steps, BATCH_SIZE, NUM_READS, WORD_SIZE])

        loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)

        # 检查梯度
        self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))

    def testDeterministicOutputs(self):
        """测试模型在相同输入和初始状态下是否产生确定性的输出和梯度。"""
        tf.random.set_seed(42)
        np.random.seed(42)

        inputs = tf.Variable(np.random.randn(TIME_STEPS, BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        # 第一次运行
        with tf.GradientTape() as tape1:
            output1 = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
            loss1 = tf.reduce_sum(output1['read_words'])
        gradients1 = tape1.gradient(loss1, [inputs] + list(self.module.trainable_variables))

        # 重置随机种子
        tf.random.set_seed(42)
        np.random.seed(42)

        # 第二次运行
        with tf.GradientTape() as tape2:
            output2 = self.module({'inputs': inputs, 'prev_state': self.initial_state}, training=True)
            loss2 = tf.reduce_sum(output2['read_words'])
        gradients2 = tape2.gradient(loss2, [inputs] + list(self.module.trainable_variables))

        # 比较损失
        self.assertAllClose(loss1, loss2, atol=1e-6, msg="Loss values do not match across runs.")

        # 比较梯度
        for grad1, grad2 in zip(gradients1, gradients2):
            if grad1 is not None and grad2 is not None:
                self.assertAllClose(grad1, grad2, atol=1e-6, msg="Gradients do not match across runs.")
            else:
                self.assertEqual(grad1 is None, grad2 is None,
                                 msg="Gradient presence mismatch across runs.")

    def testCustomTrainingLoop(self):
        """测试模型在自定义训练循环中的工作情况。"""
        optimizer = tf.optimizers.Adam()
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        # 简单的自定义训练步骤
        with tf.GradientTape() as tape:
            loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)

        # 应用梯度
        optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

        # 检查梯度
        self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))

        # 再次运行前向传播，检查是否没有错误
        loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
        self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))


if __name__ == '__main__':
    tf.test.main()