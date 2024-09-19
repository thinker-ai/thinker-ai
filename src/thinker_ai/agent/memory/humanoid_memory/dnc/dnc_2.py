# simple_model_with_cosine.py
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.access_2 import MemoryAccess


class DNC(tf.keras.layers.Layer):
    def __init__(self, units=10, num_heads=2, word_size=6, memory_size=20, num_writes=3, num_reads=2):
        super(DNC, self).__init__()
        self.units = units
        self.num_heads = num_heads
        self.word_size = word_size
        self.memory_size = memory_size
        self.num_writes = num_writes
        self.num_reads = num_reads

        self.dense = tf.keras.layers.Dense(units=units)
        self.memory_access = MemoryAccess(
            num_heads=num_heads,
            word_size=word_size,
            memory_size=memory_size,
            num_writes=num_writes,
            num_reads=num_reads,
            name='memory_access'
        )

    def call(self, inputs_dict, training=False):
        """
        Args:
            inputs_dict: 一个包含以下键的字典：
                - 'inputs': [batch_size, input_size] 的输入张量
                - 'memory': [batch_size, memory_size, word_size]
                - 'keys': [batch_size, num_heads, word_size]
                - 'strengths': [batch_size, num_heads]
                - 'write_weights': [batch_size, num_writes, memory_size]
                - 'free_gate': [batch_size, num_reads]
                - 'read_weights': [batch_size, num_reads, memory_size]
                - 'prev_linkage': TemporalLinkageState
                - 'prev_usage': [batch_size, memory_size]
        Returns:
            dense_output: [batch_size, units]
            memory_access_output: 包含以下键的字典
        """
        inputs = inputs_dict['inputs']
        dense_output = self.dense(inputs)  # [batch_size, units]

        memory_access_input = {
            'memory': inputs_dict['memory'],
            'keys': inputs_dict['keys'],
            'strengths': inputs_dict['strengths'],
            'write_weights': inputs_dict['write_weights'],
            'free_gate': inputs_dict['free_gate'],
            'read_weights': inputs_dict['read_weights'],
            'prev_linkage': inputs_dict['prev_linkage'],
            'prev_usage': inputs_dict['prev_usage']
        }

        memory_access_output = self.memory_access(memory_access_input, training=training)

        return dense_output, memory_access_output