import collections
from typing import Callable, Dict, Optional

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 配置类，用于动态调整 epsilon 等参数
from dataclasses import dataclass


@dataclass
class Config:
    epsilon: float = 1e-6

    @staticmethod
    def set_epsilon(value: float):
        Config.epsilon = value


# 定义 TemporalLinkageState，用于跟踪记忆链路
TemporalLinkageState = collections.namedtuple('TemporalLinkageState', ('link', 'precedence_weights'))


def swap_axes(tensor, axis1, axis2):
    return tf.experimental.numpy.swapaxes(tensor, axis1, axis2)


def _vector_norms(m, epsilon=None):
    if epsilon is None:
        epsilon = Config.epsilon  # 使用全局配置中的 epsilon 值
    squared_norms = tf.reduce_sum(m * m, axis=-1, keepdims=True)
    return tf.sqrt(squared_norms + epsilon)


def weighted_softmax(scores: tf.Tensor, weights: tf.Tensor, strength_op: Callable = tf.nn.softmax) -> tf.Tensor:
    """
    计算加权的 softmax。

    Args:
        scores (tf.Tensor): 分数张量，形状为 [batch_shape..., num_heads, memory_size]
        weights (tf.Tensor): 权重张量，形状为 [batch_shape..., num_heads]
        strength_op (callable): 用于调整权重的操作，默认使用 softmax

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

class CosineWeights:
    def __init__(self, num_heads: int, word_size: int, epsilon: float = 1e-6,
                 strength_op: Optional[Callable] = None):
        """
        初始化 CosineWeights 组件。

        Args:
            num_heads (int): 头的数量。
            word_size (int): 词向量的维度。
            epsilon (float, optional): 防止除零的小值。默认值为 1e-6。
            strength_op (callable, optional): 用于调整强度的操作。默认使用 softmax。
        """
        self._num_heads = num_heads
        self._word_size = word_size
        self._strength_op = strength_op or tf.nn.softmax
        self._epsilon = epsilon

    def compute_cosine_weights(self, keys: tf.Tensor, strengths: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        """
        计算内容权重。

        Args:
            keys (tf.Tensor): [batch_shape..., num_heads, word_size]
            strengths (tf.Tensor): [batch_shape..., num_heads]
            memory (tf.Tensor): [batch_shape..., memory_size, word_size]

        Returns:
            tf.Tensor: 内容权重 [batch_shape..., num_heads, memory_size]
        """
        # 计算 L2 范数
        memory_norms = tf.norm(memory, axis=-1)  # [batch_shape..., memory_size]
        keys_norms = tf.norm(keys, axis=-1)      # [batch_shape..., num_heads]

        # 扩展维度以便广播
        # memory: [batch_shape..., memory_size, word_size] -> [batch_shape..., memory_size, 1, word_size]
        memory_expanded = tf.expand_dims(memory, axis=-2)

        # keys: [batch_shape..., num_heads, word_size] -> [batch_shape..., 1, num_heads, word_size]
        keys_expanded = tf.expand_dims(keys, axis=-3)

        # 计算点积
        # memory_expanded * keys_expanded: [batch_shape..., memory_size, num_heads, word_size]
        # reduce_sum over word_size: [batch_shape..., memory_size, num_heads]
        dot_products = tf.reduce_sum(memory_expanded * keys_expanded, axis=-1)  # [batch_shape..., memory_size, num_heads]

        # 转置到 [batch_shape..., num_heads, memory_size]
        # 假设最后两个维度是 memory_size 和 num_heads
        # 使用 perm 参数静态指定
        # 获取总维度数
        rank = len(dot_products.shape)
        # 构建 perm 列表：前 rank-2 维保持不变，最后两个维度交换
        perm = list(range(rank - 2)) + [rank - 1, rank - 2]
        similarity = tf.transpose(dot_products, perm=perm)  # [batch_shape..., num_heads, memory_size]

        # 计算 norms
        # keys_norms: [batch_shape..., num_heads] -> [batch_shape..., num_heads, 1]
        keys_norms_expanded = tf.expand_dims(keys_norms, axis=-1)

        # memory_norms: [batch_shape..., memory_size] -> [batch_shape..., 1, memory_size]
        memory_norms_expanded = tf.expand_dims(memory_norms, axis=-2)

        # 计算 norms: [batch_shape..., num_heads, memory_size]
        norms = keys_norms_expanded * memory_norms_expanded + self._epsilon

        # 计算余弦相似度
        similarity = similarity / norms  # [batch_shape..., num_heads, memory_size]

        # 使用 strengths 调整相似度，并通过 weighted_softmax 得到权重
        weights = weighted_softmax(similarity, strengths, self._strength_op)  # [batch_shape..., num_heads, memory_size]

        return weights

class WriteAllocation:
    def __init__(
            self,
            memory_size: int,
            num_writes: int,
            epsilon: float = 1e-6,
            write_content_weights_fn: Optional[Callable[[dict, bool, Optional[int]], tf.Tensor]] = None,
            allocation_gate_fn: Optional[Callable[[int, int], tf.Tensor]] = None,
            write_gate_fn: Optional[Callable[[int, int], tf.Tensor]] = None,
    ):
        self.memory_size = memory_size
        self.num_writes = num_writes
        self.epsilon = epsilon
        self.write_content_weights_fn = write_content_weights_fn or self.default_write_content_weights
        self.allocation_gate_fn = allocation_gate_fn or self.default_allocation_gate
        self.write_gate_fn = write_gate_fn or self.default_write_gate

    def default_write_content_weights(
            self,
            inputs: dict,
            training: bool = False,
            num_writes: Optional[int] = None
    ) -> tf.Tensor:
        # 与原始代码的 default_write_content_weights 相同
        # 根据 inputs 中的内容，生成 write_content_weights
        write_content_keys = inputs.get('write_content_keys')
        write_content_strengths = inputs.get('write_content_strengths')
        if write_content_keys is not None and write_content_strengths is not None:
            # 使用 write_content_keys 和 write_content_strengths 以及 memory 来生成 write_content_weights
            memory = inputs.get('memory')
            if memory is None:
                raise KeyError(
                    "'memory' must be provided when using 'write_content_keys' and 'write_content_strengths'")
            # 计算 scores
            scores = tf.matmul(write_content_keys, memory, transpose_b=True)  # [batch_size, num_writes, memory_size]
            # 计算 weighted_softmax
            write_content_weights = weighted_softmax(scores, write_content_strengths, tf.nn.softmax)
            return write_content_weights
        else:
            # 使用 softmax 基于 usage 生成 write_content_weights
            usage = inputs.get('usage')
            write_gates_sum = inputs.get('write_gates_sum')
            if usage is not None and write_gates_sum is not None:
                write_content_weights = tf.nn.softmax(tf.expand_dims(-usage, axis=1), axis=-1)
                return write_content_weights
            else:
                raise KeyError(
                    "Inputs must contain either ('write_content_keys' and 'write_content_strengths') or ('usage' and 'write_gates_sum').")

    def default_allocation_gate(self, batch_size: tf.Tensor, num_writes: int) -> tf.Tensor:
        return tf.ones([batch_size, num_writes], dtype=tf.float32)

    def default_write_gate(self, batch_size: tf.Tensor, num_writes: int) -> tf.Tensor:
        return tf.ones([batch_size, num_writes], dtype=tf.float32)

    def _compute_write_weights(
            self,
            write_content_weights: tf.Tensor,
            allocation_gate: tf.Tensor,
            write_gate: tf.Tensor
    ) -> tf.Tensor:
        allocation_gate_expanded = tf.expand_dims(allocation_gate, axis=-1)
        write_gate_expanded = tf.expand_dims(write_gate, axis=-1)
        write_weights = write_content_weights * allocation_gate_expanded * write_gate_expanded
        return write_weights

    def compute_write_weights(
            self,
            inputs: dict,
            training: bool = False,
            num_writes: Optional[int] = None
    ) -> tf.Tensor:
        if num_writes is None:
            num_writes = self.num_writes

        write_content_weights = self.write_content_weights_fn(inputs, training, num_writes)
        batch_size = tf.shape(write_content_weights)[0]
        allocation_gate = self.allocation_gate_fn(batch_size, num_writes)
        write_gate = self.write_gate_fn(batch_size, num_writes)
        write_weights = self._compute_write_weights(write_content_weights, allocation_gate, write_gate)
        return write_weights


class TemporalLinkage:
    def __init__(self, memory_size: int, num_writes: int, epsilon: float = 1e-6):
        """
        初始化 TemporalLinkage 模块。

        Args:
            memory_size (int): 内存的大小。
            num_writes (int): 写入头的数量。
            epsilon (float): 用于更新优先级权重的小常数。
        """
        self.memory_size = memory_size
        self.num_writes = num_writes
        self.epsilon = epsilon

    def update_linkage(self, write_weights: tf.Tensor, prev_linkage: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        更新链路矩阵和优先级权重。

        Args:
            write_weights (tf.Tensor): 写入权重，形状为 [batch_size, num_writes, memory_size]。
            prev_linkage (Dict[str, tf.Tensor]): 上一时间步的链路矩阵和优先级权重。

        Returns:
            Dict[str, tf.Tensor]: 更新后的链路矩阵和优先级权重。
        """
        prev_link = prev_linkage['link']  # [batch_size, num_writes, memory_size, memory_size]
        prev_precedence_weights = prev_linkage['precedence_weights']  # [batch_size, num_writes, memory_size]

        # 更新 precedence_weights，应用 epsilon
        write_weights_sum = tf.reduce_sum(write_weights, axis=-1, keepdims=True)  # [batch_size, num_writes, 1]
        reset_gate = 1 - write_weights_sum
        updated_precedence_weights = reset_gate * prev_precedence_weights + write_weights * self.epsilon  # 应用 epsilon

        # 计算 write_weights_i 和 prev_precedence_weights_j
        write_weights_i = tf.expand_dims(write_weights, axis=-1)  # [batch_size, num_writes, memory_size, 1]
        prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights,
                                                   axis=-2)  # [batch_size, num_writes, 1, memory_size]

        # 计算 new_link
        new_link = write_weights_i * prev_precedence_weights_j  # [batch_size, num_writes, memory_size, memory_size]

        # 计算链路缩放因子
        write_weights_j = tf.expand_dims(write_weights, axis=-2)  # [batch_size, num_writes, 1, memory_size]
        prev_link_scale = 1 - write_weights_i - write_weights_j  # [batch_size, num_writes, memory_size, memory_size]
        prev_link_scale = tf.clip_by_value(prev_link_scale, 0.0, 1.0)

        # 更新链路矩阵
        updated_link = prev_link_scale * prev_link + new_link

        # 避免自连接
        mask = 1 - tf.eye(self.memory_size, batch_shape=[self.num_writes], dtype=tf.float32)
        updated_link = updated_link * tf.expand_dims(mask, axis=0)

        return {
            'link': updated_link,
            'precedence_weights': updated_precedence_weights
        }

    def directional_read_weights(self, link: tf.Tensor, prev_read_weights: tf.Tensor,
                                 forward: bool = True) -> tf.Tensor:
        """
        计算方向性读权重。

        Args:
            link (tf.Tensor): 链路矩阵，形状为 [batch_size, num_writes, memory_size, memory_size]。
            prev_read_weights (tf.Tensor): 上一时间步的读权重，形状为 [batch_size, num_reads, memory_size]。
            forward (bool): 是否计算前向读权重。

        Returns:
            tf.Tensor: 方向性读权重，形状为 [batch_size, num_reads, num_writes, memory_size]。
        """
        if not forward:
            # 转置最后两个轴
            link = tf.transpose(link, perm=[0, 1, 3, 2])

        # 调整 link 的形状以匹配 einsum 公式
        link_transposed = tf.transpose(link, perm=[0, 2, 3, 1])  # [batch_size, memory_size, memory_size, num_writes]
        result = tf.einsum('brm,bmwn->brwn', prev_read_weights,
                           link_transposed)  # [batch_size, num_reads, num_writes, memory_size]

        # 应用 softmax 归一化
        directional_weights = tf.nn.softmax(result, axis=-1)  # [batch_size, num_reads, num_writes, memory_size]

        return directional_weights

    def state_size(self) -> Dict[str, tf.TensorShape]:
        """
        返回状态的大小。

        Returns:
            Dict[str, tf.TensorShape]: 包含 'link' 和 'precedence_weights' 的张量形状。
        """
        return {
            'link': tf.TensorShape([self.num_writes, self.memory_size, self.memory_size]),
            'precedence_weights': tf.TensorShape([self.num_writes, self.memory_size])
        }


class UsageUpdate:
    def __init__(self, memory_size: int, num_writes: int, num_reads: int, epsilon: float = 1e-6):
        self.memory_size = memory_size
        self.num_writes = num_writes
        self.num_reads = num_reads
        self.epsilon = epsilon

    def update_usage(self, write_weights: tf.Tensor, free_gate: tf.Tensor, read_weights: tf.Tensor,
                     prev_usage: tf.Tensor) -> tf.Tensor:
        """
        更新内存使用率。

        Args:
            write_weights (tf.Tensor): [batch_size, num_writes, memory_size]
            free_gate (tf.Tensor): [batch_size, num_reads]
            read_weights (tf.Tensor): [batch_size, num_reads, memory_size]
            prev_usage (tf.Tensor): [batch_size, memory_size]

        Returns:
            tf.Tensor: 更新后的使用率 [batch_size, memory_size]
        """
        # 计算写操作后的使用率
        write_weights_cumprod = tf.reduce_prod(1 - write_weights, axis=1)  # [batch_size, memory_size]
        write_allocation = 1 - write_weights_cumprod  # [batch_size, memory_size]
        usage_after_write = prev_usage + (1 - prev_usage) * write_allocation  # [batch_size, memory_size]

        # 计算自由读权重
        free_gate_expanded = tf.expand_dims(free_gate, axis=-1)  # [batch_size, num_reads, 1]
        free_read_weights = free_gate_expanded * read_weights  # [batch_size, num_reads, memory_size]

        # 计算每个内存槽因自由读操作释放的使用率
        total_free_read_weights = tf.reduce_sum(free_read_weights, axis=1)  # [batch_size, memory_size]

        # 更新使用率：减少被自由读释放的部分
        usage_after_read = usage_after_write - total_free_read_weights  # [batch_size, memory_size]
        usage_after_read = tf.maximum(usage_after_read, 0.0)  # 确保不低于0

        # 返回更新后的使用率
        return usage_after_read  # [batch_size, memory_size]
