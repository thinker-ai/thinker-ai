import unittest
import tensorflow as tf
import time

from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess


class MemoryAccessLongSequenceTest(unittest.TestCase):
    def test_long_sequence(self):
        memory_size = 128
        word_size = 64
        num_reads = 4
        num_writes = 1
        batch_size = 1
        controller_output_size = 256  # 固定的控制器输出尺寸
        sequence_length = 1000  # 长序列

        memory_access = MemoryAccess(
            memory_size=memory_size,
            word_size=word_size,
            num_reads=num_reads,
            num_writes=num_writes,
            controller_output_size=controller_output_size
        )

        initial_state = memory_access.get_initial_state(batch_size)
        controller_output_sequence = tf.random.uniform([batch_size, sequence_length, controller_output_size],
                                                       dtype=tf.float32)

        inputs = {
            'inputs': controller_output_sequence,
            'prev_state': initial_state
        }

        start_time = time.time()
        output = memory_access(inputs, training=False)
        end_time = time.time()

        read_words_sequence = output['read_words']
        self.assertEqual(read_words_sequence.shape, (batch_size, sequence_length, num_reads, word_size))

        # 验证运行时间在可接受范围内（例如，不超过 10 秒）
        self.assertLess(end_time - start_time, 10.0)


class MemoryAccessLargeBatchTest(unittest.TestCase):
    def test_large_batch_size(self):
        memory_size = 128
        word_size = 64
        num_reads = 4
        num_writes = 1
        batch_size = 256  # 大批量
        controller_output_size = 256  # 固定的控制器输出尺寸
        sequence_length = 50

        memory_access = MemoryAccess(
            memory_size=memory_size,
            word_size=word_size,
            num_reads=num_reads,
            num_writes=num_writes,
            controller_output_size=controller_output_size
        )

        initial_state = memory_access.get_initial_state(batch_size)
        controller_output_sequence = tf.random.uniform([batch_size, sequence_length, controller_output_size],
                                                       dtype=tf.float32)

        inputs = {
            'inputs': controller_output_sequence,
            'prev_state': initial_state
        }

        start_time = time.time()
        output = memory_access(inputs, training=False)
        end_time = time.time()

        read_words_sequence = output['read_words']
        self.assertEqual(read_words_sequence.shape, (batch_size, sequence_length, num_reads, word_size))

        # 验证运行时间在可接受范围内（例如，不超过 10 秒）
        self.assertLess(end_time - start_time, 10.0)


if __name__ == '__main__':
    unittest.main()
