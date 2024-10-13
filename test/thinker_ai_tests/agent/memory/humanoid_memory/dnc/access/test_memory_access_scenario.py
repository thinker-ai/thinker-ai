import unittest
import tensorflow as tf
import numpy as np

from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess


class MemoryAccessUserIsolationTest(unittest.TestCase):
    def setUp(self):
        self.memory_size = 32
        self.word_size = 16
        self.num_reads = 2
        self.num_writes = 1
        self.batch_size = 2  # 两个用户
        self.controller_output_size = 64  # 固定的控制器输出尺寸

        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size
        )

        self.initial_state = self.memory_access.get_initial_state(self.batch_size)

    def test_user_memory_isolation(self):
        # 模拟两个用户的不同输入
        controller_output_user1 = tf.random.uniform([1, self.controller_output_size], dtype=tf.float32)
        controller_output_user2 = tf.random.uniform([1, self.controller_output_size], dtype=tf.float32)

        controller_output = tf.concat([controller_output_user1, controller_output_user2], axis=0)

        inputs = {
            'inputs': controller_output,
            'prev_state': self.initial_state
        }

        output = self.memory_access(inputs, training=False)

        read_words = output['read_words']
        final_state = output['final_state']

        # 验证每个用户的记忆是隔离的
        memory_user1 = final_state.memory[0]
        memory_user2 = final_state.memory[1]

        self.assertFalse(np.array_equal(memory_user1.numpy(), memory_user2.numpy()))


if __name__ == '__main__':
    unittest.main()
