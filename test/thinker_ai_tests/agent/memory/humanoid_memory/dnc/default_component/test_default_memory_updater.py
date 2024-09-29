# test_default_memory_updater.py

import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import DefaultMemoryUpdater


class DefaultMemoryUpdaterTest(tf.test.TestCase):
    def test_update_memory_basic(self):
        """
        测试 DefaultMemoryUpdater 的基本功能，包括输出形状和内存更新逻辑。
        """
        batch_size = 2
        memory_size = 4
        word_size = 5
        num_writes = 3

        # 创建 DefaultMemoryUpdater 实例
        memory_updater = DefaultMemoryUpdater()

        # 创建随机的内存、写入权重、擦除向量和写入向量
        memory = tf.random.uniform([batch_size, memory_size, word_size], minval=0.0, maxval=1.0)
        write_weights = tf.random.uniform([batch_size, num_writes, memory_size], minval=0.0, maxval=1.0)
        erase_vectors = tf.random.uniform([batch_size, num_writes, word_size], minval=0.0, maxval=1.0)
        write_vectors = tf.random.uniform([batch_size, num_writes, word_size], minval=0.0, maxval=1.0)

        # 更新内存
        updated_memory = memory_updater.update_memory(
            memory=memory,
            write_weights=write_weights,
            erase_vectors=erase_vectors,
            write_vectors=write_vectors
        )

        # 检查输出形状
        expected_shape = [batch_size, memory_size, word_size]
        self.assertAllEqual(updated_memory.shape, expected_shape)

        # 检查内存更新逻辑
        # 由于使用随机值，主要检查数值变化
        # 确保更新后的内存与初始内存不同（在一定容差内）
        self.assertNotAllClose(memory.numpy(), updated_memory.numpy(), atol=1e-6)

    def test_update_memory_erase_all(self):
        """
        测试当擦除向量为1时，确保对应内存被完全擦除。
        """
        batch_size = 1
        memory_size = 2
        word_size = 3
        num_writes = 1

        # 创建 DefaultMemoryUpdater 实例
        memory_updater = DefaultMemoryUpdater()

        # 设置内存为特定值
        memory = tf.constant([[[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]]], dtype=tf.float32)  # [1, 2, 3]

        # 设置写入权重为1，擦除向量为1，写入向量为0
        write_weights = tf.constant([[[1.0, 1.0]]], dtype=tf.float32)  # [1, 1, 2]
        erase_vectors = tf.constant([[[1.0, 1.0, 1.0]]], dtype=tf.float32)  # [1, 1, 3]
        write_vectors = tf.constant([[[0.0, 0.0, 0.0]]], dtype=tf.float32)  # [1, 1, 3]

        # 更新内存
        updated_memory = memory_updater.update_memory(
            memory=memory,
            write_weights=write_weights,
            erase_vectors=erase_vectors,
            write_vectors=write_vectors
        )

        # 期望内存被完全擦除并添加0，即全为0
        expected_memory = tf.zeros([1, 2, 3], dtype=tf.float32)
        self.assertAllClose(updated_memory.numpy(), expected_memory.numpy(), atol=1e-6)


if __name__ == '__main__':
    tf.test.main()