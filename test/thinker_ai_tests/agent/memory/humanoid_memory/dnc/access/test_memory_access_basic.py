import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess


class MemoryAccessInitializationTest(tf.test.TestCase):
    def test_initialization(self):
        memory_size = 128
        word_size = 64
        num_reads = 4
        num_writes = 1
        self.controller_output_size = 128
        memory_access = MemoryAccess(
            memory_size=memory_size,
            word_size=word_size,
            num_reads=num_reads,
            num_writes=num_writes,
            controller_output_size=self.controller_output_size
        )

        self.assertEqual(memory_access.memory_size, memory_size)
        self.assertEqual(memory_access.word_size, word_size)
        self.assertEqual(memory_access.num_reads, num_reads)
        self.assertEqual(memory_access.num_writes, num_writes)
        self.assertIsNotNone(memory_access.content_weight_calculator)
        self.assertIsNotNone(memory_access.write_weight_calculator)
        self.assertIsNotNone(memory_access.temporal_linkage_updater)
        self.assertIsNotNone(memory_access.read_weight_calculator)
        self.assertIsNotNone(memory_access.usage_updater)
        self.assertIsNotNone(memory_access.memory_updater)


class MemoryAccessSingleStepTest(tf.test.TestCase):
    def setUp(self):
        super(MemoryAccessSingleStepTest, self).setUp()
        self.memory_size = 16
        self.word_size = 8
        self.num_reads = 1
        self.num_writes = 1
        self.batch_size = 1
        self.controller_output_size = 64
        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size
        )

        self.initial_state = self.memory_access.get_initial_state(self.batch_size)

    def test_single_step_read_write(self):
        input_size = 32
        controller = tf.keras.layers.Dense(self.controller_output_size)
        controller_input = tf.random.uniform([self.batch_size, input_size], dtype=tf.float32)
        controller_output = controller(controller_input)

        inputs = {
            'inputs': controller_output,
            'prev_state': self.initial_state
        }

        output = self.memory_access(inputs, training=False)
        read_words = output['read_words']
        final_state = output['final_state']

        self.assertEqual(read_words.shape, (self.batch_size, self.num_reads, self.word_size))
        self.assertEqual(final_state.memory.shape, (self.batch_size, self.memory_size, self.word_size))


if __name__ == '__main__':
    tf.test.main()
