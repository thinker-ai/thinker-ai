# model_components.py
import tensorflow as tf
import collections

# Small epsilon value to prevent numerical instability
_EPSILON = 1e-6


def _vector_norms(m):
    """计算向量的L2范数，以防止数值不稳定性。"""
    squared_norms = tf.reduce_sum(m * m, axis=2, keepdims=True)  # [batch_size, num_heads, 1]
    return tf.sqrt(squared_norms + _EPSILON)  # [batch_size, num_heads, 1]


def weighted_softmax(activations, strengths, strengths_op):
    """对激活值进行加权softmax操作。"""
    transformed_strengths = tf.expand_dims(strengths_op(strengths), -1)  # [batch_size, num_heads, 1]
    sharp_activations = activations * transformed_strengths  # [batch_size, num_heads, memory_size]
    return tf.nn.softmax(sharp_activations, axis=-1)  # [batch_size, num_heads, memory_size]


class CosineWeights(tf.keras.layers.Layer):
    """基于余弦相似度的注意力机制，用于存储器寻址。"""

    def __init__(self, num_heads, word_size, strength_op=tf.nn.softplus, name='cosine_weights'):
        super(CosineWeights, self).__init__(name=name)
        self.num_heads = num_heads
        self.word_size = word_size
        self.strength_op = strength_op

    def call(self, inputs, training=False):
        """
        Args:
            inputs: 包含 'memory', 'keys', 和 'strengths' 的字典。
                - 'memory': [batch_size, memory_size, word_size]
                - 'keys': [batch_size, num_heads, word_size]
                - 'strengths': [batch_size, num_heads]
        Returns:
            [batch_size, num_heads, memory_size] 的注意力权重
        """
        memory = inputs['memory']  # [batch_size, memory_size, word_size]
        keys = inputs['keys']  # [batch_size, num_heads, word_size]
        strengths = inputs['strengths']  # [batch_size, num_heads]

        # 计算 keys 与 memory 之间的点积
        dot = tf.matmul(keys, memory, adjoint_b=True)  # [batch_size, num_heads, memory_size]

        # 计算 memory 和 keys 的 L2 范数
        memory_norms = _vector_norms(memory)  # [batch_size, memory_size, 1]
        key_norms = _vector_norms(keys)  # [batch_size, num_heads, 1]

        # 调整 memory_norms 和 key_norms 的形状，使其可以广播
        memory_norms = tf.transpose(memory_norms, [0, 2, 1])  # [batch_size, 1, memory_size]
        memory_norms = tf.broadcast_to(memory_norms, [tf.shape(memory)[0], self.num_heads,
                                                      tf.shape(memory)[1]])  # [batch_size, num_heads, memory_size]

        key_norms = tf.broadcast_to(key_norms, [tf.shape(keys)[0], self.num_heads,
                                                tf.shape(memory)[1]])  # [batch_size, num_heads, memory_size]

        # 计算余弦相似度
        normed_dot = dot / (memory_norms * key_norms + _EPSILON)  # [batch_size, num_heads, memory_size]

        # 使用 strengths 调整相似度，并通过 softmax 得到权重
        return weighted_softmax(normed_dot, strengths, self.strength_op)  # [batch_size, num_heads, memory_size]


# 定义 TemporalLinkageState
TemporalLinkageState = collections.namedtuple('TemporalLinkageState', ('link', 'precedence_weights'))


class TemporalLinkage(tf.keras.layers.Layer):
    """跟踪每个写入头的存储写入顺序的模块。"""

    def __init__(self, memory_size, num_writes, name='temporal_linkage'):
        super(TemporalLinkage, self).__init__(name=name)
        self.memory_size = memory_size
        self.num_writes = num_writes

    def call(self, inputs, training=False):
        """
        Args:
            inputs: 包含 'write_weights' 和 'prev_linkage' 的字典。
                - 'write_weights': [batch_size, num_writes, memory_size]
                - 'prev_linkage': TemporalLinkageState
        Returns:
            更新后的 TemporalLinkageState
        """
        write_weights = inputs['write_weights']  # [batch_size, num_writes, memory_size]
        prev_linkage = inputs['prev_linkage']  # TemporalLinkageState

        # 获取上一次的时序链路矩阵和优先级权重
        prev_link = prev_linkage.link  # [batch_size, num_writes, memory_size, memory_size]
        prev_precedence_weights = prev_linkage.precedence_weights  # [batch_size, num_writes, memory_size]

        # 这里添加 TemporalLinkage 的逻辑
        # 为了简单起见，这里仅仅将 write_weights 作为新的 precedence_weights
        updated_precedence_weights = write_weights  # [batch_size, num_writes, memory_size]

        # 这里暂时不更新 link，保持与上一次相同
        updated_link = prev_link  # [batch_size, num_writes, memory_size, memory_size]

        return TemporalLinkageState(link=updated_link, precedence_weights=updated_precedence_weights)


class Freeness(tf.keras.layers.Layer):
    """跟踪每个内存位置的使用情况，并提供空闲位置的权重。"""

    def __init__(self, memory_size, name='freeness'):
        super(Freeness, self).__init__(name=name)
        self.memory_size = memory_size

    def call(self, inputs, training=False):
        """
        Args:
            inputs: 包含 'write_weights', 'free_gate', 'read_weights', 和 'prev_usage' 的字典。
                - 'write_weights': [batch_size, num_writes, memory_size]
                - 'free_gate': [batch_size, num_reads]
                - 'read_weights': [batch_size, num_reads, memory_size]
                - 'prev_usage': [batch_size, memory_size]
        Returns:
            updated_usage: [batch_size, memory_size]
        """
        write_weights = inputs['write_weights']  # [batch_size, num_writes, memory_size]
        free_gate = inputs['free_gate']  # [batch_size, num_reads]
        read_weights = inputs['read_weights']  # [batch_size, num_reads, memory_size]
        prev_usage = inputs['prev_usage']  # [batch_size, memory_size]

        # 更新使用率
        usage_increase = tf.reduce_sum(write_weights, axis=1)  # [batch_size, memory_size]
        usage_decrease = tf.reduce_sum(read_weights * tf.expand_dims(free_gate, -1),
                                       axis=1)  # [batch_size, memory_size]
        updated_usage = prev_usage + usage_increase - usage_decrease

        # 打印调试信息
        tf.print("write_weights shape:", tf.shape(write_weights))
        tf.print("free_gate shape:", tf.shape(free_gate))
        tf.print("read_weights shape:", tf.shape(read_weights))
        tf.print("usage_increase shape:", tf.shape(usage_increase))
        tf.print("usage_decrease shape:", tf.shape(usage_decrease))
        tf.print("updated_usage shape:", tf.shape(updated_usage))

        # 确保使用率在 [0, 1] 之间
        updated_usage = tf.clip_by_value(updated_usage, 0.0, 1.0)

        return updated_usage  # [batch_size, memory_size]


class MemoryAccess(tf.keras.layers.Layer):
    """集成 CosineWeights、TemporalLinkage、Freeness、Erase、Write 和 Read 操作的内存访问层。"""

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
