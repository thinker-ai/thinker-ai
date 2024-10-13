from unittest import mock

import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess


class MemoryAccessFlexibleInputTest(tf.test.TestCase):
    def setUp(self):
        super(MemoryAccessFlexibleInputTest, self).setUp()
        self.memory_size = 32
        self.word_size = 16
        self.num_reads = 2
        self.num_writes = 1
        self.batch_size = 1
        self.controller_output_size = 128

        # Mock CacheManager to avoid interference between tests
        self.cache_manager_mock = mock.Mock()
        self.cache_manager_mock.read_from_cache.return_value = None  # Mock to always return None

        # Initialize MemoryAccess with the mocked CacheManager
        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size,
            cache_manager=self.cache_manager_mock  # Inject the mocked cache manager
        )
        self.initial_state = self.memory_access.get_initial_state(self.batch_size)

    def test_fixed_input_size(self):
        controller_output = tf.random.uniform([self.batch_size, self.controller_output_size], dtype=tf.float32)
        inputs = {
            'inputs': controller_output,
            'prev_state': self.initial_state
        }
        output = self.memory_access(inputs, training=False)
        read_words = output['read_words']
        self.assertEqual(read_words.shape, (self.batch_size, self.num_reads, self.word_size))


class MemoryAccessFlexibleMemoryTest(tf.test.TestCase):
    def test_variable_memory_size(self):
        memory_sizes = [16, 32, 64]
        word_sizes = [8, 16, 32]
        num_reads = 2
        num_writes = 1
        batch_size = 1
        input_size = 128
        self.controller_output_size = 128
        # Mock CacheManager to avoid interference between tests
        self.cache_manager_mock = mock.Mock()
        self.cache_manager_mock.read_from_cache.return_value = None  # Mock to always return None
        for memory_size in memory_sizes:
            for word_size in word_sizes:
                with self.subTest(memory_size=memory_size, word_size=word_size):
                    memory_access = MemoryAccess(
                        memory_size=memory_size,
                        word_size=word_size,
                        num_reads=num_reads,
                        num_writes=num_writes,
                        controller_output_size=self.controller_output_size,
                        cache_manager=self.cache_manager_mock
                    )

                    initial_state = memory_access.get_initial_state(batch_size)
                    controller_output = tf.random.uniform([batch_size, input_size], dtype=tf.float32)

                    inputs = {
                        'inputs': controller_output,
                        'prev_state': initial_state
                    }

                    output = memory_access(inputs, training=False)
                    read_words = output['read_words']
                    self.assertEqual(read_words.shape, (batch_size, num_reads, word_size))


if __name__ == '__main__':
    tf.test.main()
