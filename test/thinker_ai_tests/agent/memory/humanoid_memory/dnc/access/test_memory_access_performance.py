import time
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess


class MemoryAccessLongSequenceTest(tf.test.TestCase):
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

        # 测量开始时间
        start_time = time.time()

        # 执行 MemoryAccess 操作
        output = memory_access(inputs, training=False)

        # 测量结束时间
        end_time = time.time()

        read_words_sequence = output['read_words']

        # 验证输出形状是否正确
        self.assertEqual(read_words_sequence.shape, (batch_size, sequence_length, num_reads, word_size))

        # 验证运行时间在可接受范围内（例如，不超过 10 秒）
        self.assertLess(end_time - start_time, 10.0)


if __name__ == '__main__':
    tf.test.main()
