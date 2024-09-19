# test_simple_model_memory_access.py
import tensorflow as tf
import numpy as np

from thinker_ai_tests.agent.memory.humanoid_memory.dnc.simple_model_with_cosine import SimpleModel
from thinker_ai_tests.agent.memory.humanoid_memory.dnc.model_components import TemporalLinkageState  # 确保路径正确


class SimpleModelMemoryAccessTest(tf.test.TestCase):
    def _create_memory_access_inputs(self, config):
        """
        辅助函数，根据配置创建 memory_access 的所有外部输入。
        """
        batch_size = config['batch_size']
        memory_size = config['memory_size']
        word_size = config['word_size']
        num_heads = config['num_heads']
        num_writes = config['num_writes']
        num_reads = config['num_reads']

        memory = tf.random.normal([batch_size, memory_size, word_size], dtype=tf.float32)
        keys = tf.random.normal([batch_size, num_heads, word_size], dtype=tf.float32)
        strengths = tf.random.normal([batch_size, num_heads], dtype=tf.float32)
        write_weights = tf.random.normal([batch_size, num_writes, memory_size], dtype=tf.float32)
        free_gate = tf.random.normal([batch_size, num_reads], dtype=tf.float32)
        read_weights = tf.random.normal([batch_size, num_reads, memory_size], dtype=tf.float32)
        prev_linkage = TemporalLinkageState(
            link=tf.zeros((batch_size, num_writes, memory_size, memory_size), dtype=tf.float32),
            precedence_weights=tf.zeros((batch_size, num_writes, memory_size), dtype=tf.float32)
        )
        prev_usage = tf.zeros((batch_size, memory_size), dtype=tf.float32)

        return {
            'memory': memory,
            'keys': keys,
            'strengths': strengths,
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_linkage': prev_linkage,
            'prev_usage': prev_usage
        }

    def test_different_batch_sizes(self):
        """测试模型在不同批次大小下的表现。"""
        configurations = [
            {'batch_size': 1, 'memory_size': 10, 'num_writes': 2, 'num_reads': 1, 'word_size': 6, 'num_heads': 2},
            {'batch_size': 4, 'memory_size': 50, 'num_writes': 5, 'num_reads': 3, 'word_size': 6, 'num_heads': 2},
            {'batch_size': 8, 'memory_size': 100, 'num_writes': 10, 'num_reads': 5, 'word_size': 6, 'num_heads': 2},
        ]

        for config in configurations:
            with self.subTest(config=config):
                units = 10
                num_heads = config['num_heads']
                word_size = config['word_size']
                memory_size = config['memory_size']
                num_writes = config['num_writes']
                num_reads = config['num_reads']
                batch_size = config['batch_size']
                input_size = 5

                # 创建模型实例
                model = SimpleModel(
                    units=units,
                    num_heads=num_heads,
                    word_size=word_size,
                    memory_size=memory_size,
                    num_writes=num_writes,
                    num_reads=num_reads
                )

                # 定义输入
                inputs = tf.Variable(np.random.randn(batch_size, input_size), dtype=tf.float32)

                # 创建 memory_access 的外部输入
                memory_access_inputs = self._create_memory_access_inputs(config)

                # 组织所有输入为一个字典
                inputs_dict = {
                    'inputs': inputs,
                    **memory_access_inputs
                }

                # 执行前向传播
                with tf.GradientTape() as tape:
                    dense_output, memory_access_output = model(inputs_dict, training=True)
                    # 定义损失为所有输出的均值相加
                    loss = (
                        tf.reduce_mean(dense_output) +
                        tf.reduce_mean(memory_access_output['cosine_output']) +
                        tf.reduce_mean(memory_access_output['temporal_output'].precedence_weights) +
                        tf.reduce_mean(memory_access_output['updated_usage']) +
                        tf.reduce_mean(memory_access_output['updated_memory']) +
                        tf.reduce_mean(memory_access_output['read_output'])
                    )
                    tf.debugging.check_numerics(loss, "Loss contains NaNs or Infs")

                # 计算梯度
                gradients = tape.gradient(loss, [inputs] + model.trainable_variables)

                # 验证梯度
                for grad, var in zip(gradients, [inputs] + model.trainable_variables):
                    var_name = var.name
                    if 'dense/kernel:0' in var_name:
                        # 检查梯度是否大于阈值
                        if grad is not None:
                            grad_norm = tf.norm(grad)
                            self.assertGreater(grad_norm, 1e-6, f"Gradient for {var_name} is too small")
                    # 其他变量不做断言

    def test_edge_case_inputs(self):
        """测试模型在边界输入下的表现。"""
        configurations = [
            {'description': 'All zeros input', 'input': np.zeros((2, 5), dtype=np.float32),
             'expect_kernel_grad': False},
            {'description': 'Large positive values', 'input': np.full((2, 5), 1e6, dtype=np.float32),
             'expect_kernel_grad': False},
            {'description': 'Large negative values', 'input': np.full((2, 5), -1e6, dtype=np.float32),
             'expect_kernel_grad': False},
            {'description': 'Mixed extreme values', 'input': np.array([[1e6, -1e6, 0.0, 1e6, -1e6],
                                                                       [-1e6, 1e6, -1e6, 1e6, 0.0]], dtype=np.float32),
             'expect_kernel_grad': False},
        ]

        # 固定模型参数
        units = 10
        num_heads = 2
        word_size = 6
        memory_size = 20
        num_writes = 3
        num_reads = 2
        batch_size = 2
        input_size = 5

        for config in configurations:
            with self.subTest(config=config):
                # 创建模型实例
                model = SimpleModel(
                    units=units,
                    num_heads=num_heads,
                    word_size=word_size,
                    memory_size=memory_size,
                    num_writes=num_writes,
                    num_reads=num_reads
                )

                # 定义输入
                inputs = tf.Variable(config['input'], dtype=tf.float32)

                # 创建 memory_access 的外部输入
                memory_access_inputs = self._create_memory_access_inputs({
                    'batch_size': batch_size,
                    'memory_size': memory_size,
                    'word_size': word_size,
                    'num_heads': num_heads,
                    'num_writes': num_writes,
                    'num_reads': num_reads
                })

                # 组织所有输入为一个字典
                inputs_dict = {
                    'inputs': inputs,
                    **memory_access_inputs
                }

                # 执行前向传播
                with tf.GradientTape() as tape:
                    dense_output, memory_access_output = model(inputs_dict, training=True)
                    # 定义损失为所有输出的均值相加
                    loss = (
                        tf.reduce_mean(dense_output) +
                        tf.reduce_mean(memory_access_output['cosine_output']) +
                        tf.reduce_mean(memory_access_output['temporal_output'].precedence_weights) +
                        tf.reduce_mean(memory_access_output['updated_usage']) +
                        tf.reduce_mean(memory_access_output['updated_memory']) +
                        tf.reduce_mean(memory_access_output['read_output'])
                    )
                    # 检查损失是否包含 NaNs 或 Infs
                    tf.debugging.check_numerics(loss, "Loss contains NaNs or Infs")

                # 计算梯度
                gradients = tape.gradient(loss, [inputs] + model.trainable_variables)

                # 验证梯度
                for grad, var in zip(gradients, [inputs] + model.trainable_variables):
                    var_name = var.name
                    if 'dense/kernel:0' in var_name:
                        if config.get('expect_kernel_grad', True):
                            if grad is not None:
                                grad_norm = tf.norm(grad)
                                self.assertGreater(grad_norm, 1e-6, f"Gradient for {var_name} is too small")
                        else:
                            # 对于期望梯度为零的配置，验证梯度确实很小
                            if grad is not None:
                                grad_norm = tf.norm(grad)
                                self.assertLessEqual(grad_norm, 1e-6,
                                                     f"Gradient for {var_name} should be small for {config['description']}")
                    # 其他变量不做断言

    def test_numerical_stability(self):
        """测试模型的数值稳定性。"""
        units = 10
        num_heads = 2
        word_size = 6
        memory_size = 20
        num_writes = 3
        num_reads = 2
        batch_size = 2
        input_size = 5

        # 创建模型实例
        model = SimpleModel(
            units=units,
            num_heads=num_heads,
            word_size=word_size,
            memory_size=memory_size,
            num_writes=num_writes,
            num_reads=num_reads
        )

        # 定义输入，使用极端值
        inputs = tf.Variable(np.full((batch_size, input_size), 1e10, dtype=np.float32), dtype=tf.float32)

        # 创建 memory_access 的外部输入
        memory_access_inputs = self._create_memory_access_inputs({
            'batch_size': batch_size,
            'memory_size': memory_size,
            'word_size': word_size,
            'num_heads': num_heads,
            'num_writes': num_writes,
            'num_reads': num_reads
        })

        # 组织所有输入为一个字典
        inputs_dict = {
            'inputs': inputs,
            **memory_access_inputs
        }

        # 执行前向传播
        with tf.GradientTape() as tape:
            dense_output, memory_access_output = model(inputs_dict, training=True)
            # 定义损失为所有输出的均值相加
            loss = (
                tf.reduce_mean(dense_output) +
                tf.reduce_mean(memory_access_output['cosine_output']) +
                tf.reduce_mean(memory_access_output['temporal_output'].precedence_weights) +
                tf.reduce_mean(memory_access_output['updated_usage']) +
                tf.reduce_mean(memory_access_output['updated_memory']) +
                tf.reduce_mean(memory_access_output['read_output'])
            )
            # 检查损失是否包含 NaNs 或 Infs
            tf.debugging.check_numerics(loss, "Loss contains NaNs or Infs")

        # 计算梯度
        gradients = tape.gradient(loss, [inputs] + model.trainable_variables)

        # 验证梯度
        for grad, var in zip(gradients, [inputs] + model.trainable_variables):
            var_name = var.name
            if 'dense/kernel:0' in var_name:
                if grad is not None:
                    grad_norm = tf.norm(grad)
                    self.assertGreater(grad_norm, 1e-6, f"Gradient for {var_name} is too small")
            # 其他变量不做断言


if __name__ == '__main__':
    tf.test.main()