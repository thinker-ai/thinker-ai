import tensorflow as tf
import numpy as np
import os
import tempfile

from thinker_ai.agent.memory.humanoid_memory.dnc_new import access
from thinker_ai.agent.memory.humanoid_memory.dnc_new.access import MemoryAccess, AccessState

from tensorflow.keras.models import save_model, load_model


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


class MemoryAccessModelSavingSerializationTests(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessModelSavingSerializationTests, self).setUp()
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

    def testModelSavingAndLoading(self):
        """测试 MemoryAccess 模型的保存和加载功能，确保权重和配置被正确保留。"""
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, 'memory_access_model')
            save_model(self.module, save_path, save_format='tf')

            # 加载模型
            loaded_module = load_model(save_path, custom_objects={
                'MemoryAccess': access.MemoryAccess,
                'AccessState': AccessState,
                'TemporalLinkageState': access.TemporalLinkageState  # 确保导入正确的类
            })

            # 比较权重
            for original_var, loaded_var in zip(self.module.trainable_variables, loaded_module.trainable_variables):
                self.assertAllClose(original_var.numpy(), loaded_var.numpy(), atol=1e-6,
                                    msg=f"Mismatch in variable '{original_var.name}' after loading.")

    def testModelSerialization(self):
        """测试模型的序列化和反序列化，确保所有权重和配置被正确保存和恢复。"""
        with tempfile.TemporaryDirectory() as tmpdirname:
            # 保存模型
            save_path = os.path.join(tmpdirname, 'memory_access_model')
            save_model(self.module, save_path, save_format='tf')

            # 加载模型
            loaded_module = load_model(save_path, custom_objects={
                'MemoryAccess': access.MemoryAccess,
                'AccessState': AccessState,
                'TemporalLinkageState': access.TemporalLinkageState
            })

            # 比较原始模型和加载后的模型的权重
            for original_var, loaded_var in zip(self.module.trainable_variables, loaded_module.trainable_variables):
                original_values = original_var.numpy()
                loaded_values = loaded_var.numpy()
                self.assertAllClose(original_values, loaded_values, atol=1e-6,
                                    msg=f"Mismatch in variable '{original_var.name}' after serialization.")


if __name__ == '__main__':
    tf.test.main()