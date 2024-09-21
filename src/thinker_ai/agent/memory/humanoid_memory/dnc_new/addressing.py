#thinker_ai/agent/memory/humanoid_memory/dnc/addressing.py
import collections
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息
# Small epsilon value to prevent numerical instability
_EPSILON = 1e-6

# Define TemporalLinkageState for memory linkage tracking
TemporalLinkageState = collections.namedtuple('TemporalLinkageState', ('link', 'precedence_weights'))


def _vector_norms(m):
    squared_norms = tf.reduce_sum(m * m, axis=2, keepdims=True)  # 计算每个向量的平方和
    return tf.sqrt(squared_norms + _EPSILON)  # 计算平方和的平方根，并加上一个小正数以确保数值稳定


def weighted_softmax(activations, strengths, strengths_op):
    # 对strengths进行变换，并扩展维度以与activations相乘
    transformed_strengths = tf.expand_dims(strengths_op(strengths), -1)

    # 将activations与strengths相乘，调整激活值
    sharp_activations = activations * transformed_strengths

    # 对调整后的激活值应用softmax，沿着最后一个维度归一化
    return tf.nn.softmax(sharp_activations, axis=-1)


class CosineWeights(tf.keras.layers.Layer):

    def __init__(self, num_heads, word_size, strength_op=tf.nn.softplus, name='cosine_weights'):
        super(CosineWeights, self).__init__(name=name)
        self._num_heads = num_heads
        self._word_size = word_size
        self._strength_op = strength_op

    def call(self, inputs, training=False):
        # 从输入中提取 memory（存储器）、keys（查询向量）和 strengths（强度）
        memory = inputs['memory']  # [batch_size, memory_size, word_size]
        keys = inputs['keys']  # [batch_size, num_heads, word_size]
        strengths = inputs['strengths']  # [batch_size, num_heads]

        # 提取形状参数
        batch_size = tf.shape(memory)[0]  # 获取 batch_size
        memory_size = tf.shape(memory)[1]  # 获取 memory_size
        num_heads = tf.shape(keys)[1]  # 获取 num_heads

        # 计算 keys 与 memory 之间的点积
        dot = tf.matmul(keys, memory, adjoint_b=True)  # [batch_size, num_heads, memory_size]
        # print("Dot product shape:", dot.shape)
        # print("Dot product sample:", dot.numpy()[0, :2, :2])  # 仅显示前两个值

        # 计算 memory 和 keys 的 L2 范数
        memory_norms = _vector_norms(memory)  # [batch_size, memory_size, 1]
        key_norms = _vector_norms(keys)  # [batch_size, num_heads, 1]
        # print("Memory norms shape:", memory_norms.shape)
        # print("Memory norms sample:", memory_norms.numpy()[:2])  # 仅显示前两个值
        # print("Key norms shape:", key_norms.shape)
        # print("Key norms sample:", key_norms.numpy()[:2])  # 仅显示前两个值

        # 调整 memory_norms 和 key_norms 的形状，使其可以广播
        memory_norms = tf.transpose(memory_norms, [0, 2, 1])  # [batch_size, 1, memory_size]
        memory_norms = tf.broadcast_to(memory_norms,
                                       [batch_size, num_heads, memory_size])  # [batch_size, num_heads, memory_size]

        key_norms = tf.broadcast_to(key_norms,
                                    [batch_size, num_heads, memory_size])  # [batch_size, num_heads, memory_size]

        # 使用余弦相似度公式：cosine_similarity = dot_product / (||key|| * ||memory||)
        normed_dot = dot / (memory_norms * key_norms + _EPSILON)

        # 使用 strengths 调整相似度，并通过 softmax 得到权重
        weights = weighted_softmax(normed_dot, strengths, self._strength_op)
        # print("Calculated weights shape:", weights.shape)
        # print("Calculated weights sample:", weights.numpy()[0, :2, :2])  # 仅显示前两个值

        return weights


class TemporalLinkage(tf.keras.layers.Layer):


    def __init__(self, memory_size, num_writes, name='temporal_linkage'):
        """TemporalLinkage 的初始化方法。

        Args:
          memory_size: 整数，表示存储器的大小，即存储器中有多少个位置可供读写操作。
          num_writes: 整数，表示写头的数量。每个写头可以独立写入不同的存储位置。
          name: 字符串，表示该层的名称，在创建模型时用于区分该层。
        """
        super(TemporalLinkage, self).__init__(name=name)
        self._memory_size = memory_size  # 存储器的大小（即存储器位置的数量）
        self._num_writes = num_writes  # 写头的数量

    def build(self, input_shape):
        # 初始化时序链路矩阵和优先权重
        self.link = self.add_weight(
            shape=(self._num_writes, self._memory_size, self._memory_size),
            initializer='zeros', trainable=False, name='link')
        self.precedence_weights = self.add_weight(
            shape=(self._num_writes, self._memory_size),
            initializer='zeros', trainable=False, name='precedence_weights')

    def call(self, inputs, training=False):
        write_weights = inputs['write_weights']  # 当前写头的写权重，形状为 [batch_size, num_writes, memory_size]
        prev_linkage = inputs['prev_linkage']  # 上一次的时序链路状态

        # 获取上一次的时序链路矩阵和优先级权重
        prev_link = prev_linkage.link
        prev_precedence_weights = prev_linkage.precedence_weights

        # 更新优先级权重
        reset_gate = 1 - tf.reduce_sum(write_weights, axis=2, keepdims=True)
        updated_precedence_weights = reset_gate * prev_precedence_weights + write_weights

        # 计算新的时序链路矩阵，基于当前的写权重和之前的时序链路状态
        new_link = self._link(prev_link, prev_precedence_weights, write_weights)

        # 返回新的 TemporalLinkageState
        return TemporalLinkageState(link=new_link, precedence_weights=updated_precedence_weights)

    def _link(self, prev_link, prev_precedence_weights, write_weights):
        batch_size = tf.shape(prev_link)[0]
        write_weights_i = tf.expand_dims(write_weights, 3)  # [batch_size, num_writes, memory_size, 1]
        write_weights_j = tf.expand_dims(write_weights, 2)  # [batch_size, num_writes, 1, memory_size]
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights,
                                                   2)  # [batch_size, num_writes, 1, memory_size]

        # 计算链路缩放值，以考虑当前的写权重
        prev_link_scale = 1 - write_weights_i - write_weights_j
        new_link = write_weights_i * prev_precedence_weights_j  # 计算新的链路关系
        link = prev_link_scale * prev_link + new_link  # 更新链路矩阵

        # 将对角线元素设为 0，避免自循环（self-looping edges）
        mask = tf.eye(self._memory_size, batch_shape=[batch_size, self._num_writes])
        return link * (1 - mask)

    def directional_read_weights(self, link, prev_read_weights, forward):
        # 为每个写头复制读权重，并根据时序链路矩阵进行前向或后向的权重计算
        expanded_read_weights = tf.stack([prev_read_weights] * self._num_writes, axis=1)
        result = tf.matmul(expanded_read_weights, link, adjoint_b=forward)
        return tf.transpose(result, perm=[0, 2, 1, 3]) # 调整维度顺序，返回新权重

    def _precedence_weights(self, prev_precedence_weights, write_weights):
        write_sum = tf.reduce_sum(write_weights, axis=2, keepdims=True)  # 计算写权重之和
        return (1 - write_sum) * prev_precedence_weights + write_weights  # 更新优先级权重

    @property
    def state_size(self):
        return TemporalLinkageState(
            link=tf.TensorShape([self._num_writes, self._memory_size, self._memory_size]),
            precedence_weights=tf.TensorShape([self._num_writes, self._memory_size])
        )


class Freeness(tf.keras.layers.Layer):

    def __init__(self, memory_size, name='freeness'):
        super(Freeness, self).__init__(name=name)
        self._memory_size = memory_size  # 保存内存的大小（可用内存槽的数目）

    def build(self, input_shape):
        # 初始化使用率向量，初始值为 0，表示所有内存位置都未使用。
        self.usage = self.add_weight(
            shape=(self._memory_size,), initializer='zeros', trainable=False, name='usage')

    def call(self, inputs, training=False):
        # 获取输入的读写权重和上一次的使用率
        write_weights = inputs['write_weights']
        free_gate = inputs['free_gate']
        read_weights = inputs['read_weights']
        prev_usage = inputs['prev_usage']

        # 打印前一状态使用率
        # print("Previous usage shape:", prev_usage.shape)
        # print("Previous usage values:", prev_usage.numpy())  # 仅显示前几个值

        # 更新内存使用率
        new_usage = self._usage_after_write(prev_usage, write_weights)
        # print("Usage after write shape:", new_usage.shape)
        # print("Usage after write values:", new_usage.numpy())  # 仅显示前几个值

        new_usage = self._usage_after_read(new_usage, free_gate, read_weights)
        # print("Usage after read shape:", new_usage.shape)
        # print("Usage after read values:", new_usage.numpy())  # 仅显示前几个值

        # 确保使用率在 0 到 1 之间
        clipped_usage = tf.clip_by_value(new_usage, 0, 1)
        # print("Clipped usage shape:", clipped_usage.shape)
        # print("Clipped usage values:", clipped_usage.numpy())  # 仅显示前几个值

        return clipped_usage

    def _usage_after_write(self, prev_usage, write_weights):
        # 计算写操作对内存位置的影响，权重乘积表示写入的强度
        write_weights = 1 - tf.reduce_prod(1 - write_weights, axis=1)
        return prev_usage + (1 - prev_usage) * write_weights

    def _usage_after_read(self, prev_usage, free_gate, read_weights):
        # 计算读操作对内存位置的影响，释放门控制哪些内存可以被释放
        free_gate = tf.expand_dims(free_gate, -1)
        free_read_weights = free_gate * read_weights
        # 计算释放因子，表示内存使用率的减少量
        phi = tf.reduce_prod(1 - free_read_weights, axis=1)

        # 应用释放因子，更新内存使用率
        return prev_usage * phi

    def _allocation(self, usage):
        # 确保使用率在合理范围内
        usage = _EPSILON + (1 - _EPSILON) * usage
        nonusage = 1 - usage

        # 使用稳定的降序排序获取索引
        sorted_indices = tf.argsort(nonusage, direction='DESCENDING', stable=True)
        sorted_nonusage = tf.gather(nonusage, sorted_indices, batch_dims=1)

        # 计算累积乘积
        sorted_usage = 1 - sorted_nonusage
        prod_sorted_usage = tf.math.cumprod(sorted_usage, axis=1, exclusive=True)
        sorted_allocation = sorted_nonusage * prod_sorted_usage

        # 使用 argsort 恢复原始顺序
        inverse_indices = tf.argsort(sorted_indices, axis=1, stable=True)
        allocation = tf.gather(sorted_allocation, inverse_indices, batch_dims=1)
        return allocation

    def write_allocation_weights(self, usage, write_gates, num_writes):
        # 展开写入门到内存位置上
        write_gates = tf.expand_dims(write_gates, -1)

        allocation_weights = []
        for i in range(num_writes):
            # 为每个写头计算分配权重
            allocation = self._allocation(usage)
            allocation_weights.append(allocation)
            # 更新使用率，考虑新的分配位置
            usage += (1 - usage) * write_gates[:, i, :] * allocation_weights[i]

        # 将分配权重打包成一个张量
        return tf.stack(allocation_weights, axis=1)

    @property
    def state_size(self):
        return tf.TensorShape([self._memory_size])

