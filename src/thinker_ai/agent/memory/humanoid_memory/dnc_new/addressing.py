import collections
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 配置类，用于动态调整 epsilon 等参数
class Config:
    epsilon = 1e-6  # 可调整的全局 epsilon 值

# 定义 TemporalLinkageState，用于跟踪记忆链路
TemporalLinkageState = collections.namedtuple('TemporalLinkageState', ('link', 'precedence_weights'))


def swap_axes(tensor, axis1, axis2):
    """
    交换张量的两个轴。

    Args:
        tensor (tf.Tensor): 输入张量。
        axis1 (int): 第一个轴的索引。
        axis2 (int): 第二个轴的索引。

    Returns:
        tf.Tensor: 交换指定轴后的张量。
    """
    rank = tf.rank(tensor)
    axis1 = tf.cast(axis1, tf.int32)
    axis2 = tf.cast(axis2, tf.int32)

    # 生成新的轴顺序
    axes = tf.range(rank)
    axes = tf.where(tf.equal(axes, axis1), axis2, axes)
    axes = tf.where(tf.equal(axes, axis2), axis1, axes)

    return tf.transpose(tensor, perm=axes)


def _vector_norms(m, epsilon=None):
    if epsilon is None:
        epsilon = Config.epsilon  # 使用全局配置中的 epsilon 值
    squared_norms = tf.reduce_sum(m * m, axis=-1, keepdims=True)
    return tf.sqrt(squared_norms + epsilon)

def weighted_softmax(scores, weights, strength_op=tf.nn.softplus):
    """
    计算加权的 softmax。

    Args:
        scores (tf.Tensor): 分数张量，形状为 [batch_shape..., num_heads, memory_size]
        weights (tf.Tensor): 权重张量，形状为 [batch_shape..., num_heads]
        strength_op (callable): 用于调整权重的操作，默认使用 softplus

    Returns:
        tf.Tensor: 归一化后的权重，形状为 [batch_shape..., num_heads, memory_size]
    """
    # 调整权重形状以便与 scores 相乘
    weights_expanded = tf.expand_dims(weights, axis=-1)  # [batch_shape..., num_heads, 1]

    # 应用 strength_op
    adjusted_weights = strength_op(weights_expanded)  # [batch_shape..., num_heads, 1]

    # 扩展调整后的权重以匹配 scores 的形状
    adjusted_weights = tf.broadcast_to(adjusted_weights, tf.shape(scores))  # [batch_shape..., num_heads, memory_size]

    # 计算加权分数
    weighted_scores = scores * adjusted_weights  # [batch_shape..., num_heads, memory_size]

    # 应用 softmax 归一化
    normalized_weights = tf.nn.softmax(weighted_scores, axis=-1)  # [batch_shape..., num_heads, memory_size]

    return normalized_weights

class CosineWeights(tf.keras.layers.Layer):
    def __init__(self, num_heads, word_size, epsilon=1e-6, strength_op=tf.nn.softplus, name='cosine_weights'):
        """
        初始化 CosineWeights 层。

        Args:
            num_heads (int): 头的数量。
            word_size (int): 词向量的维度。
            epsilon (float, optional): 防止除零的小值。默认值为 1e-6。
            strength_op (callable, optional): 用于调整强度的操作。默认使用 softplus。
            name (str, optional): 层的名称。默认值为 'cosine_weights'。
        """
        super(CosineWeights, self).__init__(name=name)
        self._num_heads = num_heads
        self._word_size = word_size
        self._strength_op = strength_op
        self._epsilon = epsilon

    def call(self, inputs, training=False):
        """
        前向传播方法。

        Args:
            inputs (dict): 包含 'memory'、'keys' 和 'strengths' 的字典。
                - memory: [batch_shape..., memory_size, word_size]
                - keys: [batch_shape..., num_heads, word_size]
                - strengths: [batch_shape..., num_heads]
            training (bool, optional): 指示是否为训练模式。默认值为 False。

        Returns:
            tf.Tensor: 计算得到的权重，形状为 [batch_shape..., num_heads, memory_size]
        """
        # 从输入中提取 memory（存储器）、keys（查询向量）和 strengths（强度）
        memory = inputs['memory']          # [batch_shape..., memory_size, word_size]
        keys = inputs['keys']              # [batch_shape..., num_heads, word_size]
        strengths = inputs['strengths']    # [batch_shape..., num_heads]

        # 获取张量的秩（维度数）
        memory_rank = tf.rank(memory)
        keys_rank = tf.rank(keys)

        # 计算扩展维度的位置
        memory_expand_axis = memory_rank - 2
        keys_expand_axis = keys_rank - 2

        # 扩展 memory 和 keys 的维度以便进行广播和点积
        memory_expanded = tf.expand_dims(memory, axis=memory_expand_axis)  # [batch_shape..., 1, memory_size, word_size]
        keys_expanded = tf.expand_dims(keys, axis=-2)                      # [batch_shape..., num_heads, 1, word_size]

        # 计算点积
        scores = tf.reduce_sum(memory_expanded * keys_expanded, axis=-1)  # [batch_shape..., num_heads, memory_size]

        # 计算 L2 范数
        memory_norms = tf.norm(memory, axis=-1)  # [batch_shape..., memory_size]
        keys_norms = tf.norm(keys, axis=-1)      # [batch_shape..., num_heads]

        # 调整维度以便广播
        memory_norms_expanded = tf.expand_dims(memory_norms, axis=-2)  # [batch_shape..., 1, memory_size]
        keys_norms_expanded = tf.expand_dims(keys_norms, axis=-1)      # [batch_shape..., num_heads, 1]

        # 计算余弦相似度
        normed_scores = scores / (memory_norms_expanded * keys_norms_expanded + self._epsilon)  # [batch_shape..., num_heads, memory_size]

        # 使用 strengths 调整相似度，并通过 weighted_softmax 得到权重
        weights = weighted_softmax(normed_scores, strengths, self._strength_op)  # [batch_shape..., num_heads, memory_size]

        return weights

class TemporalLinkage(tf.keras.layers.Layer):
    def __init__(self, memory_size, num_writes, name='temporal_linkage'):
        super(TemporalLinkage, self).__init__(name=name)
        self.memory_size = memory_size
        self.num_writes = num_writes

    def call(self, inputs, training=False):
        """
        前向传播方法。

        Args:
            inputs (dict): 包含 'write_weights' 和 'prev_linkage' 的字典。
            training (bool): 指示是否为训练模式。

        Returns:
            dict: 包含更新后的 'link' 和 'precedence_weights'。
        """
        write_weights = inputs['write_weights']          # [batch_shape..., num_writes, memory_size]
        prev_linkage = inputs['prev_linkage']

        prev_link = prev_linkage['link']                 # [batch_shape..., num_writes, memory_size, memory_size]
        prev_precedence_weights = prev_linkage['precedence_weights']  # [batch_shape..., num_writes, memory_size]

        # 更新优先级权重
        reset_gate = 1 - tf.reduce_sum(write_weights, axis=-1, keepdims=True)  # [batch_shape..., num_writes, 1]
        updated_precedence_weights = reset_gate * prev_precedence_weights + write_weights  # [batch_shape..., num_writes, memory_size]

        # 更新链路矩阵
        new_link = self._link(prev_link, prev_precedence_weights, write_weights)  # [batch_shape..., num_writes, memory_size, memory_size]

        return {
            'link': new_link,
            'precedence_weights': updated_precedence_weights
        }

    def _link(self, prev_link, prev_precedence_weights, write_weights):
        """
        更新链路矩阵，计算每次写入的影响。

        Args:
            prev_link (tf.Tensor): 之前的链路矩阵 [batch_shape..., num_writes, memory_size, memory_size]
            prev_precedence_weights (tf.Tensor): 之前的优先级权重 [batch_shape..., num_writes, memory_size]
            write_weights (tf.Tensor): 当前写入权重 [batch_shape..., num_writes, memory_size]

        Returns:
            tf.Tensor: 更新后的链路矩阵 [batch_shape..., num_writes, memory_size, memory_size]
        """
        # 扩展维度以进行外积计算
        write_weights_i = tf.expand_dims(write_weights, axis=-1)                   # [batch_shape..., num_writes, memory_size, 1]
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, axis=-2)  # [batch_shape..., num_writes, 1, memory_size]

        # 计算新链路，使用外积
        new_link = write_weights_i * prev_precedence_weights_j                     # [batch_shape..., num_writes, memory_size, memory_size]

        # 计算链路缩放因子
        write_weights_j = tf.expand_dims(write_weights, axis=-2)                   # [batch_shape..., num_writes, 1, memory_size]
        prev_link_scale = 1 - write_weights_i - write_weights_j                      # [batch_shape..., num_writes, memory_size, memory_size]

        # 确保链路缩放因子不为负
        prev_link_scale = tf.clip_by_value(prev_link_scale, 0.0, 1.0)

        # 更新链路矩阵
        updated_link = prev_link_scale * prev_link + new_link                      # [batch_shape..., num_writes, memory_size, memory_size]

        # 避免自连接
        memory_size = tf.shape(updated_link)[-1]
        mask = 1 - tf.eye(memory_size, batch_shape=tf.shape(updated_link)[:-2], dtype=tf.float32)  # [batch_shape..., memory_size, memory_size]
        final_link = updated_link * mask                                              # [batch_shape..., num_writes, memory_size, memory_size]

        return final_link

    def directional_read_weights(self, link, prev_read_weights, forward=True):
        """
        计算前向或后向的读权重。

        Args:
            link (tf.Tensor): 当前的链路矩阵 [batch_shape..., num_writes, memory_size, memory_size]
            prev_read_weights (tf.Tensor): 之前的读权重 [batch_shape..., num_reads, memory_size]
            forward (bool): 指示是否为前向计算

        Returns:
            tf.Tensor: 方向性的读权重 [batch_shape..., num_reads, num_writes, memory_size]
        """
        if not forward:
            # 使用辅助函数交换最后两个轴
            link = swap_axes(link, -1, -2)  # [batch_shape..., num_writes, memory_size, memory_size]

        # 使用 tf.einsum 计算方向性读权重
        # prev_read_weights: [batch_shape..., num_reads, memory_size]
        # link: [batch_shape..., num_writes, memory_size, memory_size]
        # 结果： [batch_shape..., num_reads, num_writes, memory_size]
        result = tf.einsum('...rj,...wij->...rwi', prev_read_weights, link)

        # 应用 softmax 归一化
        directional_weights = tf.nn.softmax(result, axis=-1)  # [batch_shape..., num_reads, num_writes, memory_size]

        tf.print("After einsum, result shape:", tf.shape(result))
        tf.print("After softmax, directional_weights shape:", tf.shape(directional_weights))

        return directional_weights

    def get_initial_state(self, batch_size):
        """
        返回 TemporalLinkage 模块的初始状态。

        Args:
            batch_size (int or tf.Tensor): 批次大小。

        Returns:
            dict: 包含初始化的 'link' 和 'precedence_weights'。
        """
        if isinstance(batch_size, int):
            batch_shape = [batch_size]
        else:
            # 假设 batch_size 是一个标量张量
            batch_shape = tf.shape(batch_size)  # 获取动态批次形状

        # 初始化链路矩阵和优先级权重为零
        link = tf.zeros(tf.concat([batch_shape, [self.num_writes, self.memory_size, self.memory_size]], axis=0), dtype=tf.float32)
        precedence_weights = tf.zeros(tf.concat([batch_shape, [self.num_writes, self.memory_size]], axis=0), dtype=tf.float32)

        return {
            'link': link,
            'precedence_weights': precedence_weights
        }

    @property
    def state_size(self):
        """
        返回状态的大小。

        Returns:
            dict: 包含 'link' 和 'precedence_weights' 的张量形状。
        """
        return {
            'link': tf.TensorShape([self.num_writes, self.memory_size, self.memory_size]),
            'precedence_weights': tf.TensorShape([self.num_writes, self.memory_size])
        }


class Freeness(tf.keras.layers.Layer):
    def __init__(self, memory_size, num_writes, epsilon=1e-6, name='freeness'):
        super(Freeness, self).__init__(name=name)
        self._memory_size = memory_size
        self._num_writes = num_writes
        self._epsilon = epsilon

    def build(self, input_shape):
        # 假设 input_shape 是 usage 的形状，即 [batch_size, memory_size]
        if isinstance(input_shape, tuple):
            self.batch_dims = len(input_shape) - 1
        else:
            self.batch_dims = 1  # 默认设置
        super(Freeness, self).build(input_shape)

    def call(self, inputs, training=False):
        write_weights = inputs['write_weights']  # [batch_shape..., num_writes, memory_size]
        free_gate = inputs['free_gate']  # [batch_shape..., num_reads]
        read_weights = inputs['read_weights']  # [batch_shape..., num_reads, memory_size]
        prev_usage = inputs['prev_usage']  # [batch_shape..., memory_size]

        # 更新使用率：先处理写操作，再处理读操作
        usage_after_write = self._usage_after_write(prev_usage, write_weights)  # [batch_shape..., memory_size]
        tf.print("Usage after write:", usage_after_write)

        usage_after_read = self._usage_after_read(usage_after_write, free_gate,
                                                  read_weights)  # [batch_shape..., memory_size]
        tf.print("Usage after read:", usage_after_read)

        # 裁剪使用率到 [0, 1]
        clipped_usage = tf.clip_by_value(usage_after_read, 0.0, 1.0)
        tf.print("Clipped usage:", clipped_usage)

        return clipped_usage  # [batch_shape..., memory_size]


    def _usage_after_write(self, usage, write_weights):
        """
        计算写操作后内存的使用率。

        Args:
            usage (tf.Tensor): 当前使用率，形状为 [batch_shape..., memory_size]
            write_weights (tf.Tensor): 写权重，形状为 [batch_shape..., num_writes, memory_size]

        Returns:
            tf.Tensor: 写操作后的使用率，形状为 [batch_shape..., memory_size]
        """
        # 计算每个内存槽被任一写操作写入的概率
        write_weights_cumprod = tf.reduce_prod(1 - write_weights, axis=-2)  # [batch_shape..., memory_size]
        write_allocation = 1 - write_weights_cumprod  # [batch_shape..., memory_size]

        # 更新使用率：增加被写入的内存槽，同时保持已使用的部分
        updated_usage = usage + (1 - usage) * write_allocation  # [batch_shape..., memory_size]
        tf.print("Updated usage after write:", updated_usage)
        return updated_usage

    def _usage_after_read(self, usage, free_gate, read_weights):
        """
        计算读操作后内存的使用率。

        Args:
            usage (tf.Tensor): 写操作后的使用率，形状为 [batch_shape..., memory_size]
            free_gate (tf.Tensor): 自由门，形状为 [batch_shape..., num_reads]
            read_weights (tf.Tensor): 读权重，形状为 [batch_shape..., num_reads, memory_size]

        Returns:
            tf.Tensor: 读操作后的使用率，形状为 [batch_shape..., memory_size]
        """
        # 扩展 free_gate 的维度以匹配 read_weights
        free_gate_expanded = tf.expand_dims(free_gate, axis=-1)  # [batch_shape..., num_reads, 1]

        # 计算自由读权重
        free_read_weights = free_gate_expanded * read_weights  # [batch_shape..., num_reads, memory_size]

        # 计算每个内存槽因自由读操作释放的使用率
        total_free_read_weights = tf.reduce_sum(free_read_weights, axis=-2)  # [batch_shape..., memory_size]

        # 更新使用率：减少被自由读释放的部分
        updated_usage = usage - total_free_read_weights  # [batch_shape..., memory_size]
        updated_usage = tf.maximum(updated_usage, 0.0)  # 确保不低于0
        tf.print("Updated usage after read:", updated_usage)
        return updated_usage

    def _allocation(self, usage):
        """
        计算内存分配权重。
        [保持不变]
        """
        adjusted_usage = self._epsilon + (1 - self._epsilon) * usage  # [batch_shape..., memory_size]
        nonusage = 1 - adjusted_usage  # [batch_shape..., memory_size]

        # 对 nonusage 进行降序排序，并获取排序的索引
        sorted_nonusage, indices = tf.nn.top_k(nonusage, k=self._memory_size,
                                               sorted=True)  # [batch_shape..., memory_size]

        # 计算排序后的使用率
        sorted_usage = 1 - sorted_nonusage  # [batch_shape..., memory_size]

        # 计算累积乘积（不包含当前元素）
        cumprod_sorted_usage = tf.math.cumprod(sorted_usage + self._epsilon, axis=-1,
                                               exclusive=True)  # [batch_shape..., memory_size]

        # 计算排序后的分配权重
        sorted_allocation = sorted_nonusage * cumprod_sorted_usage  # [batch_shape..., memory_size]

        # 恢复原始顺序
        inverse_indices = tf.argsort(indices, axis=-1)  # [batch_shape..., memory_size]

        tf.print("Adjusted usage shape:", tf.shape(adjusted_usage))
        tf.print("Non-usage shape:", tf.shape(nonusage))
        tf.print("Sorted non-usage shape:", tf.shape(sorted_nonusage))
        tf.print("Indices shape:", tf.shape(indices))
        tf.print("Sorted usage shape:", tf.shape(sorted_usage))
        tf.print("Cumulative product sorted usage shape:", tf.shape(cumprod_sorted_usage))
        tf.print("Sorted allocation shape:", tf.shape(sorted_allocation))
        tf.print("Inverse indices shape:", tf.shape(inverse_indices))

        # 使用在 build 方法中确定的 batch_dims
        allocation = tf.gather(sorted_allocation, inverse_indices, batch_dims=self.batch_dims)
        tf.print("Allocation shape after gather:", tf.shape(allocation))
        return allocation

    def write_allocation_weights(self, usage, write_gates):
        """
        计算写操作的分配权重。
        """
        write_gates_expanded = tf.expand_dims(write_gates, axis=-1)  # [batch_size, num_writes, 1]

        # 计算 allocation
        allocation = self._allocation(usage)  # [batch_size, memory_size]
        allocation_expanded = tf.expand_dims(allocation, axis=-2)  # [batch_size, 1, memory_size]

        # 计算 allocation_weights
        allocation_weights = allocation_expanded * tf.ones_like(write_gates_expanded)  # [batch_size, num_writes, memory_size]

        # 根据 write_gates 调整 allocation_weights
        write_allocation_weights = write_gates_expanded * allocation_weights  # [batch_size, num_writes, memory_size]

        tf.print("Write allocation weights:", write_allocation_weights)

        return write_allocation_weights

    def get_initial_state(self, batch_shape):
        """
        返回 Freeness 模块的初始状态。

        Args:
            batch_shape (tuple or list): 批次形状，不包括 memory_size。

        Returns:
            tf.Tensor: 初始使用率，形状为 [batch_shape..., memory_size]
        """
        usage = tf.zeros(tf.concat([batch_shape, [self._memory_size]], axis=0), dtype=tf.float32)
        return usage

    @property
    def state_size(self):
        return tf.TensorShape([self._memory_size])