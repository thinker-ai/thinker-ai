from thinker_ai.agent.memory.humanoid_memory.dnc.addressing_2 import CosineWeights, TemporalLinkage, Freeness
import tensorflow as tf


class MemoryAccess(tf.keras.layers.Layer):
    """
    该模块允许多个读取和写入头对外部存储器进行操作。DNC 是一种增强型神经网络模型，
    它通过外部存储器来扩展其记忆能力。这个 `MemoryAccess` 模块负责处理外部存储器
    的读取、写入和管理，确保多个读取和写入操作能够高效地并发运行。该模块使用了以下
    机制：

    - `addressing.TemporalLinkage`：用于追踪每个写入头在存储器中的写入顺序，确保写入
      的内容可以根据时间顺序进行管理。
    - `addressing.FreenessAllocator`：用于追踪存储器的使用情况，保证在控制器允许的情况下，
      存储器可以被释放，避免不必要占用。

    动态神经计算机（DNC）不仅能像传统神经网络一样学习数据的模式，还可以借助外部存储器记住大量数据，
    并对其进行复杂的操作，如读写和跟踪时间顺序。
    """

    def __init__(self, num_heads, word_size, memory_size, num_writes, num_reads, name='memory_access'):
        super(MemoryAccess, self).__init__(name=name)
        self.cosine_weights = CosineWeights(
            num_heads=num_heads,
            word_size=word_size,
            strength_op=tf.nn.softplus,
            name='cosine_weights'
        )
        self.temporal_linkage = TemporalLinkage(
            memory_size=memory_size,
            num_writes=num_writes,
            name='temporal_linkage'
        )
        self.freeness = Freeness(
            memory_size=memory_size,
            name='freeness'
        )

    def call(self, inputs, training=False):
        """
        Args:
            inputs: 包含 'memory', 'keys', 'strengths', 'write_weights', 'free_gate', 'read_weights', 'prev_linkage', 'prev_usage' 的字典。
                - 'memory': [batch_size, memory_size, word_size]
                - 'keys': [batch_size, num_heads, word_size]
                - 'strengths': [batch_size, num_heads]
                - 'write_weights': [batch_size, num_writes, memory_size]
                - 'free_gate': [batch_size, num_reads]
                - 'read_weights': [batch_size, num_reads, memory_size]
                - 'prev_linkage': TemporalLinkageState
                - 'prev_usage': [batch_size, memory_size]
        Returns:
            更新后的内存状态和读取输出
        """
        # CosineWeights
        cosine_output = self.cosine_weights({
            'memory': inputs['memory'],
            'keys': inputs['keys'],
            'strengths': inputs['strengths']
        }, training=training)  # [batch_size, num_heads, memory_size]

        # TemporalLinkage
        temporal_output = self.temporal_linkage({
            'write_weights': inputs['write_weights'],
            'prev_linkage': inputs['prev_linkage']
        }, training=training)  # TemporalLinkageState

        # Freeness
        updated_usage = self.freeness({
            'write_weights': inputs['write_weights'],
            'free_gate': inputs['free_gate'],
            'read_weights': inputs['read_weights'],
            'prev_usage': inputs['prev_usage']
        }, training=training)  # [batch_size, memory_size]

        # Erase 操作
        erase_vectors = tf.random.normal([tf.shape(inputs['memory'])[0], inputs['write_weights'].shape[1],
                                          inputs['memory'].shape[2]])  # [batch_size, num_writes, word_size]
        erase = tf.einsum('bnm,bnw->bmw', inputs['write_weights'],
                          erase_vectors)  # [batch_size, memory_size, word_size]
        erase = tf.clip_by_value(erase, 0.0, 1.0)
        memory_erased = inputs['memory'] * (1.0 - erase)  # [batch_size, memory_size, word_size]

        # Write 操作
        write_vectors = tf.random.normal([tf.shape(inputs['memory'])[0], inputs['write_weights'].shape[1],
                                          inputs['memory'].shape[2]])  # [batch_size, num_writes, word_size]
        write = tf.einsum('bnm,bnw->bmw', inputs['write_weights'],
                          write_vectors)  # [batch_size, memory_size, word_size]
        updated_memory = memory_erased + write  # [batch_size, memory_size, word_size]

        # Read 操作
        read_vectors = tf.random.normal([tf.shape(inputs['memory'])[0], inputs['read_weights'].shape[1],
                                         inputs['memory'].shape[2]])  # [batch_size, num_reads, word_size]
        read = tf.einsum('bnm,bnw->bmw', inputs['read_weights'], read_vectors)  # [batch_size, memory_size, word_size]
        read_output = tf.reduce_sum(read, axis=1)  # [batch_size, memory_size, word_size]

        return {
            'updated_memory': updated_memory,
            'cosine_output': cosine_output,
            'temporal_output': temporal_output,
            'updated_usage': updated_usage,
            'read_output': read_output
        }
