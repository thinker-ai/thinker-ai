# default_component.py
from thinker_ai.agent.memory.humanoid_memory.dnc_new.component_interface import (
    MemoryUpdater, ReadWeightCalculator, TemporalLinkageUpdater, UsageUpdater,
    WriteWeightCalculator
)
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import WriteAllocation
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import TemporalLinkage
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import UsageUpdate

import collections
from typing import Callable, Dict, Optional

import tensorflow as tf
import os

from thinker_ai.agent.memory.humanoid_memory.dnc_new.component_interface import ContentWeightCalculator

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


class CosineWeightsCalculator(ContentWeightCalculator):
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

    def compute(self, keys: tf.Tensor, strengths: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
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
        keys_norms = tf.norm(keys, axis=-1)  # [batch_shape..., num_heads]

        # 扩展维度以便广播
        # memory: [batch_shape..., memory_size, word_size] -> [batch_shape..., memory_size, 1, word_size]
        memory_expanded = tf.expand_dims(memory, axis=-2)

        # keys: [batch_shape..., num_heads, word_size] -> [batch_shape..., 1, num_heads, word_size]
        keys_expanded = tf.expand_dims(keys, axis=-3)

        # 计算点积
        # memory_expanded * keys_expanded: [batch_shape..., memory_size, num_heads, word_size]
        # reduce_sum over word_size: [batch_shape..., memory_size, num_heads]
        dot_products = tf.reduce_sum(memory_expanded * keys_expanded,
                                     axis=-1)  # [batch_shape..., memory_size, num_heads]

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


class DefaultWriteWeightCalculator(WriteWeightCalculator):
    def __init__(self, write_allocation: WriteAllocation):
        self.write_allocation = write_allocation

    def compute_write_weights(self, write_content_weights: tf.Tensor, allocation_gate: tf.Tensor,
                              write_gate: tf.Tensor, prev_usage: tf.Tensor, training: bool) -> tf.Tensor:
        """
        计算写入权重。
        """
        inputs = {
            'write_content_weights': write_content_weights,
            'allocation_gate': allocation_gate,
            'write_gate': write_gate,
            'prev_usage': prev_usage
        }

        # 调用 WriteAllocation 的 compute_write_weights 方法
        return self.write_allocation.compute_write_weights(inputs, training=training)


class DefaultUsageUpdater(UsageUpdater):
    def __init__(self, memory_size: int, num_writes: int, num_reads: int, epsilon: float = 1e-6):
        self.usage_update = UsageUpdate(memory_size=memory_size, num_writes=num_writes, num_reads=num_reads,
                                        epsilon=epsilon)

    def update_usage(self, write_weights: tf.Tensor, free_gate: tf.Tensor, read_weights: tf.Tensor,
                     prev_usage: tf.Tensor, training: bool) -> tf.Tensor:
        """
        使用 UsageUpdate 的 update_usage 方法更新使用率。
        """
        return self.usage_update.update_usage(write_weights, free_gate, read_weights, prev_usage)


class DefaultTemporalLinkageUpdater(TemporalLinkageUpdater):
    def __init__(self, memory_size: int, num_writes: int, epsilon: float = 1e-6):
        self.temporal_linkage = TemporalLinkage(memory_size=memory_size, num_writes=num_writes, epsilon=epsilon)

    def state_size(self) -> dict:
        """
        返回状态大小。
        """
        return {
            'link': tf.TensorShape([None, self.temporal_linkage.num_writes, self.temporal_linkage.memory_size,
                                    self.temporal_linkage.memory_size]),
            'precedence_weights': tf.TensorShape(
                [None, self.temporal_linkage.num_writes, self.temporal_linkage.memory_size])
        }

    def update_linkage(self, write_weights: tf.Tensor, prev_linkage: Dict[str, tf.Tensor],
                       training: bool) -> Dict[str, tf.Tensor]:
        """
        使用 TemporalLinkage 的 update_linkage 方法更新链路。
        """
        return self.temporal_linkage.update_linkage(write_weights, prev_linkage)


class DefaultMemoryUpdater(MemoryUpdater):
    def update_memory(self, memory: tf.Tensor, write_weights: tf.Tensor, erase_vectors: tf.Tensor,
                      write_vectors: tf.Tensor) -> tf.Tensor:
        """
        更新内存。
        """
        # 计算擦除矩阵
        write_weights_expanded = tf.expand_dims(write_weights, axis=-1)  # [batch_size, num_writes, memory_size, 1]
        erase_vectors_expanded = tf.expand_dims(erase_vectors, axis=2)  # [batch_size, num_writes, 1, word_size]
        erase_matrix = tf.reduce_prod(1 - write_weights_expanded * erase_vectors_expanded,
                                      axis=1)  # [batch_size, memory_size, word_size]

        # 更新内存
        memory_erased = memory * erase_matrix

        # 计算添加矩阵
        write_vectors_expanded = tf.expand_dims(write_vectors, axis=2)  # [batch_size, num_writes, 1, word_size]
        add_matrix = tf.reduce_sum(write_weights_expanded * write_vectors_expanded,
                                   axis=1)  # [batch_size, memory_size, word_size]

        memory_updated = memory_erased + add_matrix
        return memory_updated  # [batch_size, memory_size, word_size]


class DefaultReadWeightCalculator(ReadWeightCalculator):
    def __init__(self, temporal_linkage: TemporalLinkage, num_reads: int, num_writes: int):
        """
        初始化 ReadWeightCalculator，传入 TemporalLinkage 实例。
        """
        self.temporal_linkage = temporal_linkage
        self.num_reads = num_reads
        self.num_writes = num_writes

    def compute_read_weights(self, read_content_weights: tf.Tensor, prev_read_weights: tf.Tensor,
                             link: tf.Tensor, read_mode: tf.Tensor, training: bool) -> tf.Tensor:
        """
        计算读取权重。
        """
        # 分割 read_mode
        content_mode = tf.expand_dims(read_mode[:, :, 0], axis=-1)  # [batch_size, num_reads, 1]
        forward_mode = read_mode[:, :, 1:1 + self.num_writes]  # [batch_size, num_reads, num_writes]
        backward_mode = read_mode[:, :, 1 + self.num_writes:]  # [batch_size, num_reads, num_writes]

        # 计算前向和后向权重
        forward_weights = self.temporal_linkage.directional_read_weights(link, prev_read_weights,
                                                                         forward=True)  # [batch_size, num_reads, num_writes, memory_size]
        backward_weights = self.temporal_linkage.directional_read_weights(link, prev_read_weights,
                                                                          forward=False)  # [batch_size, num_reads, num_writes, memory_size]

        # 计算加权和
        forward_mode = tf.expand_dims(forward_mode, axis=-1)  # [batch_size, num_reads, num_writes, 1]
        backward_mode = tf.expand_dims(backward_mode, axis=-1)  # [batch_size, num_reads, num_writes, 1]

        # 计算方向性权重的加权和
        forward_component = tf.reduce_sum(forward_mode * forward_weights,
                                          axis=2)  # [batch_size, num_reads, memory_size]
        backward_component = tf.reduce_sum(backward_mode * backward_weights,
                                           axis=2)  # [batch_size, num_reads, memory_size]

        # 最终读取权重
        read_weights = content_mode * read_content_weights + tf.reduce_sum([forward_component, backward_component],
                                                                           axis=0)
        return read_weights  # [batch_size, num_reads, memory_size]
