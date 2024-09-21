import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc_new.access import MemoryAccess, AccessState
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import TemporalLinkageState


class TestMemoryAccess(tf.test.TestCase):
    def setUp(self):
        self.memory_size = 128
        self.word_size = 20
        self.num_reads = 1
        self.num_writes = 1
        self.memory_access = MemoryAccess(memory_size=self.memory_size, word_size=self.word_size,
                                          num_reads=self.num_reads, num_writes=self.num_writes)

        # 初始化 prev_state
        self.prev_state = AccessState(
            memory=tf.zeros((1, self.memory_size, self.word_size)),
            read_weights=tf.zeros((1, self.memory_size)),
            write_weights=tf.zeros((1, self.memory_size)),
            linkage=TemporalLinkageState(link=tf.zeros((1, self.num_writes, self.memory_size, self.memory_size)),
                                         precedence_weights=tf.zeros((1, self.memory_size))),  # 初始链接状态
            usage=tf.zeros((1, self.memory_size))
        )

    def test_memory_update(self):
        inputs = {
            'prev_state': self.prev_state,
            'inputs': tf.random.normal((1, self.word_size))  # 随机输入
        }

        print("Running test_memory_update...")
        print(f"Previous state: {self.prev_state}")

        read_words, new_state = self.memory_access(inputs)

        # 预期输出形状应为 (1, 1, word_size)
        self.assertEqual(read_words.shape, (1, self.num_reads, self.word_size))
        print(f"Read words shape: {read_words.shape}")
        print(f"New state: {new_state}")

    def test_usage_update(self):
        inputs = {
            'prev_state': self.prev_state,
            'inputs': tf.random.normal((1, self.word_size))  # 随机输入
        }

        print("Running test_usage_update...")
        read_words, new_state = self.memory_access(inputs)
        self.assertEqual(new_state.usage.shape, (1, self.memory_size))
        print(f"New state usage shape: {new_state.usage.shape}")

    def test_weight_calculation(self):
        inputs = {
            'prev_state': self.prev_state,
            'inputs': tf.random.normal((1, self.word_size))  # 随机输入
        }

        print("Running test_weight_calculation...")
        read_words, new_state = self.memory_access(inputs)

        # 更新预期的形状以匹配实际返回的形状
        self.assertEqual(new_state.write_weights.shape, (1, self.num_writes, self.memory_size))
        print(f"New state write_weights shape: {new_state.write_weights.shape}")

if __name__ == '__main__':
    tf.test.main()