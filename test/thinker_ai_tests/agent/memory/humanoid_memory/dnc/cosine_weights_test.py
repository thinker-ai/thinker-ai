# test_cosine_weights.py
import tensorflow as tf
import numpy as np
from thinker_ai_tests.agent.memory.humanoid_memory.dnc.model_components import CosineWeights

class CosineWeightsTest(tf.test.TestCase):
    def test_cosine_weights_output_shape(self):
        """测试 CosineWeights 层的输出形状是否正确。"""
        num_heads = 3
        word_size = 5
        memory_size = 10
        batch_size = 4

        # 创建 CosineWeights 层
        cosine_weights_layer = CosineWeights(num_heads=num_heads, word_size=word_size)

        # 定义输入
        memory = tf.random.normal([batch_size, memory_size, word_size])
        keys = tf.random.normal([batch_size, num_heads, word_size])
        strengths = tf.random.normal([batch_size, num_heads])

        # 执行前向传播
        cosine_output = cosine_weights_layer({
            'memory': memory,
            'keys': keys,
            'strengths': strengths
        })

        # 验证输出形状
        expected_shape = (batch_size, num_heads, memory_size)
        self.assertEqual(cosine_output.shape, expected_shape)

    def test_cosine_weights_no_nan_inf(self):
        """测试 CosineWeights 层的输出不包含 NaNs 或 Infs。"""
        num_heads = 2
        word_size = 4
        memory_size = 8
        batch_size = 3

        # 创建 CosineWeights 层
        cosine_weights_layer = CosineWeights(num_heads=num_heads, word_size=word_size)

        # 定义输入
        memory = tf.random.normal([batch_size, memory_size, word_size])
        keys = tf.random.normal([batch_size, num_heads, word_size])
        strengths = tf.random.normal([batch_size, num_heads])

        # 执行前向传播
        cosine_output = cosine_weights_layer({
            'memory': memory,
            'keys': keys,
            'strengths': strengths
        })

        # 检查是否包含 NaNs 或 Infs
        tf.debugging.check_numerics(cosine_output, "CosineWeights output contains NaNs or Infs")

if __name__ == '__main__':
    tf.test.main()