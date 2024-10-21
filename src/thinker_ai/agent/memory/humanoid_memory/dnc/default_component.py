# default_component.py
from typing import Dict, Any

from thinker_ai.agent.memory.humanoid_memory.dnc.component_interface import (
    MemoryUpdater, ReadWeightCalculator, TemporalLinkageUpdater, UsageUpdater, WriteWeightCalculator
)
import collections

import tensorflow as tf
import os

from thinker_ai.agent.memory.humanoid_memory.dnc.component_interface import ContentWeightCalculator

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


# 已修改：调整内容权重的计算，符合 DNC 论文要求
# default_component.py
class DefaultContentWeightCalculator(ContentWeightCalculator):
    def __init__(self, word_size: int, epsilon: float = 1e-6):
        self._word_size = word_size
        self._epsilon = epsilon

    def compute(self, keys: tf.Tensor, strengths: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        """
        计算内容权重。

        Args:
            keys (tf.Tensor): 形状为 [batch_dims..., num_heads, word_size]
            strengths (tf.Tensor): 形状为 [batch_dims..., num_heads]
            memory (tf.Tensor): 形状为 [batch_dims..., memory_size, word_size]

        Returns:
            tf.Tensor: 内容权重，形状为 [batch_dims..., num_heads, memory_size]
        """
        # 计算内容权重
        keys = tf.math.l2_normalize(keys, axis=-1)  # [batch_dims..., num_heads, word_size]
        memory = tf.math.l2_normalize(memory, axis=-1)  # [batch_dims..., memory_size, word_size]

        # 计算相似度
        similarity = tf.einsum('...hw,...mw->...hm', keys, memory)  # [batch_dims..., num_heads, memory_size]

        # 应用强度
        strengths = tf.expand_dims(strengths, axis=-1)  # [batch_dims..., num_heads, 1]
        similarity *= strengths  # [batch_dims..., num_heads, memory_size]

        # 调试输出
        tf.print("Keys Sample:", keys[..., :1, :])  # 打印部分 keys
        tf.print("Memory Sample:", memory[..., :1, :])  # 打印部分 memory
        tf.print("Similarity Sample:", similarity[..., :1, :])  # 打印部分 similarity

        # 计算内容权重
        content_weights = tf.nn.softmax(similarity, axis=-1)  # [batch_dims..., num_heads, memory_size]

        # 调试输出
        tf.print("Content Weights Sample:", content_weights[..., :1, :])  # 打印部分 content_weights

        return content_weights


# 已修改：调整使用率的更新，符合 DNC 论文要求
class DefaultUsageUpdater(UsageUpdater):

    def update_usage(self, write_weights, free_gates, prev_read_weights, prev_usage, training):
        """
        更新内存使用率。

        Args:
            write_weights (tf.Tensor): [batch_size, num_writes, memory_size]
            free_gates (tf.Tensor): [batch_size, num_reads]
            prev_read_weights (tf.Tensor): [batch_size, num_reads, memory_size]
            prev_usage (tf.Tensor): [batch_size, memory_size]

        Returns:
            tf.Tensor: 更新后的使用率 [batch_size, memory_size]
        """
        # 计算写入权重的总和
        write_weights_sum = tf.reduce_sum(write_weights, axis=1)  # [batch_size, memory_size]

        # 计算记忆保留向量 ψ_t
        free_gates = tf.expand_dims(free_gates, axis=-1)  # [batch_size, num_reads, 1]
        retention = tf.reduce_prod(1 - free_gates * prev_read_weights, axis=1)  # [batch_size, memory_size]

        # 更新使用率
        usage = (prev_usage + write_weights_sum - prev_usage * write_weights_sum) * retention

        # 裁剪使用率到 [0, 1]
        usage = tf.clip_by_value(usage, 0.0, 1.0)

        return usage


class DefaultWriteWeightCalculator(WriteWeightCalculator):
    def __init__(
            self,
            memory_size: int,
            num_writes: int,
            epsilon: float = 1e-6
    ):
        if memory_size <= 0:
            raise ValueError("memory_size must be greater than 0.")
        if num_writes <= 0:
            raise ValueError("num_writes must be greater than 0.")
        self.memory_size = memory_size
        self.num_writes = num_writes
        self.epsilon = epsilon

    def compute_allocation_weights(self, usage: tf.Tensor) -> tf.Tensor:
        """
        按照 DNC 论文计算分配权重。

        Args:
            usage: [batch_size, memory_size]

        Returns:
            allocation_weights: [batch_size, num_writes, memory_size]
        """
        batch_size = tf.shape(usage)[0]

        # 确保使用率在 [0, 1] 范围内
        usage = tf.clip_by_value(usage, 0.0, 1.0)

        # 判断是否所有内存都已被使用
        is_all_used = tf.reduce_all(tf.greater_equal(usage, 1.0 - self.epsilon), axis=1)  # [batch_size]

        # 计算 allocation_weights
        sorted_usage, free_list = tf.nn.top_k(-usage, k=self.memory_size)
        sorted_usage = -sorted_usage  # [batch_size, memory_size]
        sorted_nonusage = 1 - sorted_usage  # [batch_size, memory_size]

        # 处理 sorted_nonusage 中接近于零的值
        sorted_nonusage = tf.where(sorted_nonusage <= self.epsilon, tf.zeros_like(sorted_nonusage), sorted_nonusage)

        cumprod = tf.math.cumprod(sorted_nonusage + self.epsilon, axis=1, exclusive=True)  # [batch_size, memory_size]
        allocation_weights_sorted = sorted_nonusage * cumprod  # [batch_size, memory_size]

        # 将 allocation_weights_sorted 映射回原来的顺序
        inverted_indices = tf.argsort(free_list, axis=1)
        allocation_weights = tf.gather(allocation_weights_sorted, inverted_indices,
                                       batch_dims=1)  # [batch_size, memory_size]

        # 归一化 allocation_weights
        allocation_weights_sum = tf.reduce_sum(allocation_weights, axis=1,
                                               keepdims=True) + self.epsilon  # [batch_size, 1]
        allocation_weights_normalized = allocation_weights / allocation_weights_sum  # [batch_size, memory_size]

        # 当所有内存已被使用时，将 allocation_weights 设置为零
        is_all_used = tf.reshape(is_all_used, [batch_size, 1])  # [batch_size, 1]
        is_all_used_broadcast = tf.tile(is_all_used, [1, self.memory_size])  # [batch_size, memory_size]

        allocation_weights_normalized = tf.where(
            is_all_used_broadcast,
            tf.zeros_like(allocation_weights_normalized),
            allocation_weights_normalized
        )  # [batch_size, memory_size]

        # 扩展维度以匹配 num_writes
        allocation_weights_normalized = tf.expand_dims(allocation_weights_normalized,
                                                       axis=1)  # [batch_size, 1, memory_size]
        allocation_weights_normalized = tf.tile(allocation_weights_normalized,
                                                [1, self.num_writes, 1])  # [batch_size, num_writes, memory_size]

        return allocation_weights_normalized  # [batch_size, num_writes, memory_size]

    def compute(self, write_content_weights: tf.Tensor, allocation_gate: tf.Tensor,
                write_gate: tf.Tensor, prev_usage: tf.Tensor, training: bool) -> tf.Tensor:
        """
        计算最终的写入权重。

        Args:
            write_content_weights: [batch_size, num_writes, memory_size]
            allocation_gate: [batch_size, num_writes]
            write_gate: [batch_size, num_writes]
            prev_usage: [batch_size, memory_size]
            training: bool

        Returns:
            write_weights: [batch_size, num_writes, memory_size]
        """
        # 计算分配权重
        allocation_weights = self.compute_allocation_weights(prev_usage)  # [batch_size, num_writes, memory_size]

        # 确保 allocation_gate 和 write_gate 在 [0, 1] 范围内
        allocation_gate = tf.clip_by_value(allocation_gate, 0.0, 1.0)  # [batch_size, num_writes]
        write_gate = tf.clip_by_value(write_gate, 0.0, 1.0)  # [batch_size, num_writes]

        # 确保 write_content_weights 为非负
        write_content_weights = tf.maximum(write_content_weights, 0.0)

        # 对 write_content_weights 进行归一化
        write_content_weights_sum = tf.reduce_sum(write_content_weights, axis=-1,
                                                  keepdims=True) + self.epsilon  # [batch_size, num_writes, 1]
        write_content_weights_normalized = write_content_weights / write_content_weights_sum  # [batch_size, num_writes, memory_size]

        # 计算写入权重
        write_weights = write_gate[:, :, tf.newaxis] * (
                allocation_gate[:, :, tf.newaxis] * allocation_weights +
                (1 - allocation_gate[:, :, tf.newaxis]) * write_content_weights_normalized
        )  # [batch_size, num_writes, memory_size]

        return write_weights  # [batch_size, num_writes, memory_size]


class DefaultTemporalLinkageUpdater(TemporalLinkageUpdater):
    def __init__(self, memory_size: int):
        self.memory_size = memory_size

    def update_linkage(self, write_weights: tf.Tensor, prev_linkage: dict,
                       training: bool = False) -> dict:
        """
        更新链接矩阵和先行权重。

        Args:
            write_weights (tf.Tensor): [batch_size, memory_size]
            prev_linkage (dict): 包含 'link' 和 'precedence_weights'
            training (bool): 是否在训练模式

        Returns:
            dict: 更新后的 'link' 和 'precedence_weights'
        """
        prev_link = prev_linkage['link']  # [batch_size, memory_size, memory_size]
        prev_precedence_weights = prev_linkage['precedence_weights']  # [batch_size, memory_size]

        write_weights_i = tf.expand_dims(write_weights, axis=2)  # [batch_size, memory_size, 1]
        write_weights_j = tf.expand_dims(write_weights, axis=1)  # [batch_size, 1, memory_size]

        # 更新链接矩阵
        new_link = (1 - write_weights_i - write_weights_j) * prev_link + \
                   write_weights_i * prev_precedence_weights[:, tf.newaxis, :]

        # 移除自连接
        new_link = new_link * (1 - tf.eye(self.memory_size, batch_shape=[tf.shape(write_weights)[0]]))

        # 更新先行权重
        new_precedence_weights = (1 - tf.reduce_sum(write_weights, axis=1,
                                                    keepdims=True)) * prev_precedence_weights + write_weights

        return {
            'link': new_link,
            'precedence_weights': new_precedence_weights
        }

    def directional_read_weights(self, link: tf.Tensor, read_weights: tf.Tensor, forward: bool) -> tf.Tensor:
        """
        计算方向性读取权重。

        Args:
            link (tf.Tensor): [batch_size, memory_size, memory_size]
            read_weights (tf.Tensor): [batch_size, num_reads, memory_size]
            forward (bool): 是否为前向遍历

        Returns:
            tf.Tensor: 方向性读取权重，[batch_size, num_reads, memory_size]
        """
        if forward:
            weights = tf.matmul(read_weights, link)
        else:
            weights = tf.matmul(read_weights, tf.transpose(link, perm=[0, 2, 1]))
        return weights

    def state_size(self) -> Dict[str, tf.TensorShape]:
        return {
            'link': tf.TensorShape([self.memory_size, self.memory_size]),
            'precedence_weights': tf.TensorShape([self.memory_size])
        }


class DefaultReadWeightCalculator(ReadWeightCalculator):
    def __init__(self, num_reads: int):
        self.num_reads = num_reads
        self.epsilon = 1e-8

    def compute(self, read_content_weights: tf.Tensor, prev_read_weights: tf.Tensor,
                link: tf.Tensor, read_mode: tf.Tensor, training: bool) -> tf.Tensor:
        """
        计算读取权重。

        Args:
            read_content_weights (tf.Tensor): [batch_size, num_reads, memory_size]
            prev_read_weights (tf.Tensor): [batch_size, num_reads, memory_size]
            link (tf.Tensor): [batch_size, memory_size, memory_size]
            read_mode (tf.Tensor): [batch_size, num_reads, 3]
            training (bool): 是否在训练模式

        Returns:
            tf.Tensor: 读取权重，[batch_size, num_reads, memory_size]
        """
        read_mode = tf.nn.softmax(read_mode, axis=-1)  # [batch_size, num_reads, 3]

        # 获取各个模式的权重
        forward_mode = read_mode[:, :, 0:1]
        backward_mode = read_mode[:, :, 1:2]
        content_mode = read_mode[:, :, 2:3]

        # 计算方向性读取权重
        forward_weights = self.directional_read_weights(link, prev_read_weights, forward=True)
        backward_weights = self.directional_read_weights(link, prev_read_weights, forward=False)

        # 组合读取权重
        read_weights = forward_mode * forward_weights + \
                       backward_mode * backward_weights + \
                       content_mode * read_content_weights

        # 归一化读取权重
        read_weights = read_weights / (tf.reduce_sum(read_weights, axis=-1, keepdims=True) + self.epsilon)
        return read_weights  # [batch_size, num_reads, memory_size]

    def directional_read_weights(self, link: tf.Tensor, read_weights: tf.Tensor, forward: bool) -> tf.Tensor:
        if forward:
            weights = tf.matmul(read_weights, link)
        else:
            weights = tf.matmul(read_weights, tf.transpose(link, perm=[0, 2, 1]))
        return weights


class DefaultMemoryUpdater(MemoryUpdater):
    def update_memory(self, memory: tf.Tensor, write_weights: tf.Tensor, erase_vectors: tf.Tensor,
                      write_vectors: tf.Tensor) -> tf.Tensor:
        """
        Update the memory matrix according to the DNC equations.
        """
        # write_weights: [batch_size, num_writes, memory_size]
        # erase_vectors: [batch_size, num_writes, word_size]
        # write_vectors: [batch_size, num_writes, word_size]
        # memory: [batch_size, memory_size, word_size]

        # Compute the erase and add matrices
        w = tf.expand_dims(write_weights, -1)  # [batch_size, num_writes, memory_size, 1]
        e = tf.expand_dims(erase_vectors, 2)  # [batch_size, num_writes, 1, word_size]
        a = tf.expand_dims(write_vectors, 2)  # [batch_size, num_writes, 1, word_size]

        erase_term = tf.reduce_prod(1 - w * e, axis=1)  # [batch_size, memory_size, word_size]
        add_term = tf.reduce_sum(w * a, axis=1)  # [batch_size, memory_size, word_size]

        # Update memory
        memory_updated = memory * erase_term + add_term
        return memory_updated


def get_default_config(memory_size, num_writes, num_reads, word_size) -> Dict[str, Any]:
    return {
        'WriteWeightCalculator': {
            'class_path': 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultWriteWeightCalculator',
            'memory_size': memory_size,  # 动态设置
            'num_writes': num_writes,  # 动态设置
            'epsilon': 1e-6  # 保持不变
        },
        'ReadWeightCalculator': {
            'class_path': 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultReadWeightCalculator',
            'num_reads': num_reads,  # 动态设置
            'num_writes': num_writes  # 动态设置
        },
        'ContentWeightCalculator': {
            'class_path': 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultContentWeightCalculator',
            'word_size': word_size,  # 动态设置
            'epsilon': 1e-6  # 保持不变
        },
        'UsageUpdater': {
            'class_path': 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultUsageUpdater',
            'memory_size': memory_size,  # 动态设置
            'num_writes': num_writes,  # 动态设置
            'num_reads': num_reads  # 动态设置
        },
        'TemporalLinkageUpdater': {
            'class_path': 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultTemporalLinkageUpdater',
            'memory_size': memory_size,  # 动态设置
            'num_writes': num_writes  # 动态设置
        },
        'MemoryUpdater': {
            'class_path': 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultMemoryUpdater'
            # DefaultMemoryUpdater 不需要额外的初始化参数
        }
    }
