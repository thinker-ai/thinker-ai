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
class DefaultContentWeightCalculator(ContentWeightCalculator):
    def __init__(self, num_heads: int, word_size: int, epsilon: float = 1e-6):
        self._num_heads = num_heads
        self._word_size = word_size
        self._epsilon = epsilon

    def compute(self, keys: tf.Tensor, strengths: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        """
        计算内容权重。

        Args:
            keys (tf.Tensor): [batch_size, num_heads, word_size]
            strengths (tf.Tensor): [batch_size, num_heads]
            memory (tf.Tensor): [batch_size, memory_size, word_size]

        Returns:
            tf.Tensor: 内容权重 [batch_size, num_heads, memory_size]
        """
        # 计算键和内存的范数
        keys = tf.math.l2_normalize(keys, axis=-1)  # [batch_size, num_heads, word_size]
        memory = tf.math.l2_normalize(memory, axis=-1)  # [batch_size, memory_size, word_size]

        # 计算点积（余弦相似度）
        similarity = tf.matmul(keys, memory, transpose_b=True)  # [batch_size, num_heads, memory_size]

        # 使用强度参数调整相似度
        strengths = tf.expand_dims(strengths, axis=-1)  # [batch_size, num_heads, 1]
        similarity = similarity * strengths  # [batch_size, num_heads, memory_size]

        # 应用 softmax 归一化
        content_weights = tf.nn.softmax(similarity, axis=-1)  # [batch_size, num_heads, memory_size]

        return content_weights


# 已修改：调整使用率的更新，符合 DNC 论文要求
class DefaultUsageUpdater(UsageUpdater):
    def __init__(self, memory_size: int, num_writes: int, num_reads: int):
        self.memory_size = memory_size
        self.num_writes = num_writes
        self.num_reads = num_reads

    def update_usage(self, write_weights: tf.Tensor, free_gate: tf.Tensor, read_weights: tf.Tensor,
                     prev_usage: tf.Tensor, training: bool = False) -> tf.Tensor:
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
        # 计算写入权重的总和
        write_weights_sum = tf.reduce_sum(write_weights, axis=1)  # [batch_size, memory_size]

        # 计算自由读权重的总和
        read_sum = tf.reduce_sum(free_gate[:, :, tf.newaxis] * read_weights, axis=1)  # [batch_size, memory_size]

        # 更新使用率 u_t
        usage = prev_usage + write_weights_sum - prev_usage * write_weights_sum - read_sum  # [batch_size, memory_size]

        # 裁剪使用率到 [0, 1]
        usage = tf.clip_by_value(usage, 0.0, 1.0)

        return usage


# 已修改：调整内存的更新，符合 DNC 论文要求


class DefaultWriteWeightCalculator(WriteWeightCalculator):
    def __init__(
            self,
            memory_size: int,
            num_writes: int
    ):
        if memory_size <= 0:
            raise ValueError("memory_size must be greater than 0.")
        if num_writes <= 0:
            raise ValueError("num_writes must be greater than 0.")
        self.memory_size = memory_size
        self.num_writes = num_writes
        self.epsilon = 1e-6

    def compute_allocation_weights(self, usage: tf.Tensor) -> tf.Tensor:
        """
        根据使用率计算分配权重。

        Args:
            usage: [batch_size, memory_size]

        Returns:
            allocation_weights: [batch_size, memory_size]
        """
        # 防止除零
        usage = usage + self.epsilon
        # 动态获取当前 memory_size
        k = tf.minimum(self.memory_size, tf.shape(usage)[1])
        # 对使用率进行排序（降序）
        usage_sorted, indices = tf.nn.top_k(-usage, k=k)
        usage_sorted = -usage_sorted  # [batch_size, memory_size]

        # 计算 (1 - u)，确保非负
        one_minus_u = tf.maximum(1 - usage_sorted, 0.0)  # [batch_size, memory_size]

        # 计算累积乘积 (1 - u) 的非 exclusive 版本
        cumprod = tf.math.cumprod(one_minus_u, axis=1, exclusive=False)  # [batch_size, memory_size]
        shifted_cumprod = tf.concat([
            tf.ones([tf.shape(cumprod)[0], 1], dtype=cumprod.dtype),
            cumprod[:, :-1]
        ], axis=1)  # [batch_size, memory_size]

        # 计算 allocation_weights_sorted
        allocation_weights_sorted = one_minus_u * shifted_cumprod  # [batch_size, memory_size]

        # 归一化 allocation_weights_sorted，避免除以零
        sum_allocation = tf.reduce_sum(allocation_weights_sorted, axis=1, keepdims=True)  # [batch_size, 1]
        allocation_weights_normalized = tf.where(
            sum_allocation > self.epsilon,
            allocation_weights_sorted / sum_allocation,
            tf.ones_like(allocation_weights_sorted) / self.memory_size  # 当 sum=0 时，均匀分配
        )  # [batch_size, memory_size]

        # 手动执行 scatter 回原始顺序
        batch_size_dynamic = tf.shape(indices)[0]
        memory_size_dynamic = tf.shape(indices)[1]  # 动态获取 memory_size

        batch_indices = tf.range(batch_size_dynamic)[:, tf.newaxis]  # [B,1]
        batch_indices = tf.tile(batch_indices, [1, memory_size_dynamic])  # [B,M]

        scatter_indices = tf.stack([batch_indices, indices], axis=2)  # [B,M,2]

        # 根据 scatter_indices 重新排列 allocation_weights_normalized
        allocation_weights = tf.scatter_nd(scatter_indices, allocation_weights_normalized, tf.shape(usage))

        return allocation_weights

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
        allocation_weights = self.compute_allocation_weights(prev_usage)  # [batch_size, memory_size]
        allocation_weights = tf.expand_dims(allocation_weights, axis=1)  # [batch_size, 1, memory_size]
        allocation_weights = tf.tile(allocation_weights,
                                     [1, self.num_writes, 1])  # [batch_size, num_writes, memory_size]

        # 确保 allocation_gate 和 write_gate 在 [0, 1] 范围内
        allocation_gate = tf.clip_by_value(allocation_gate, 0.0, 1.0)
        write_gate = tf.clip_by_value(write_gate, 0.0, 1.0)

        # 确保 write_content_weights 为非负
        write_content_weights = tf.maximum(write_content_weights, 0.0)

        # 对 write_content_weights 进行归一化
        write_content_weights_sum = tf.reduce_sum(write_content_weights, axis=-1,
                                                  keepdims=True)  # [batch_size, num_writes, 1]
        write_content_weights_sum = tf.maximum(write_content_weights_sum, self.epsilon)  # 避免除零
        write_content_weights_normalized = write_content_weights / write_content_weights_sum  # [batch_size, num_writes, memory_size]

        # 仅在 (1 - allocation_gate) > 0 时断言 write_content_weights_normalized sum to 1.0
        allocation_gate_expanded = tf.expand_dims(allocation_gate, axis=-1)  # [batch_size, num_writes, 1]
        needs_normalization = tf.cast((1 - allocation_gate_expanded) > self.epsilon,
                                      tf.float32)  # [batch_size, num_writes,1]
        sum_write_content_weights = tf.reduce_sum(write_content_weights_normalized, axis=-1)  # [batch_size, num_writes]
        expected_sum = tf.ones_like(sum_write_content_weights) * needs_normalization[:, :, 0]

        # 调整 write_content_weights_normalized
        write_content_weights_normalized = write_content_weights_normalized * needs_normalization

        tf.debugging.assert_near(
            sum_write_content_weights * needs_normalization[:, :, 0],
            expected_sum,
            atol=1e-5,
            message="write_content_weights_normalized do not sum to 1.0 when (1 - allocation_gate) > 0"
        )

        # 扩展门控以进行元素级乘法
        write_gate_expanded = tf.expand_dims(write_gate, axis=-1)  # [batch_size, num_writes, 1]

        # 计算写入权重
        write_weights = write_gate_expanded * (
                allocation_gate_expanded * allocation_weights +
                (1 - allocation_gate_expanded) * write_content_weights_normalized
        )  # [batch_size, num_writes, memory_size]

        # 调试断言：确保 write_weights <= write_gate 并且 >=0
        tf.debugging.assert_less_equal(
            tf.reduce_max(write_weights),
            tf.reduce_max(write_gate_expanded),
            message="write_weights exceed write_gate"
        )
        tf.debugging.assert_greater_equal(
            tf.reduce_min(write_weights),
            0.0,
            message="write_weights have negative values"
        )

        # 调试断言：确保 sum_write_weights == write_gate
        sum_write_weights = tf.reduce_sum(write_weights, axis=-1)  # [batch_size, num_writes]
        tf.debugging.assert_near(
            sum_write_weights,
            write_gate,
            atol=1e-5,
            message="Sum of write_weights does not equal write_gate"
        )

        # 调试输出
        tf.print("Allocation Weights:", allocation_weights)
        tf.print("Write Content Weights Normalized:", write_content_weights_normalized)
        tf.print("Write Weights:", write_weights)
        return write_weights


class DefaultTemporalLinkageUpdater(TemporalLinkageUpdater):
    def __init__(self, memory_size: int, num_writes: int):
        """
        Initializes the TemporalLinkage module.

        Args:
            memory_size (int): The size of the memory.
            num_writes (int): The number of write heads.
        """
        if memory_size <= 0:
            raise ValueError("memory_size must be greater than 0.")
        if num_writes <= 0:
            raise ValueError("num_writes must be greater than 0.")
        self.memory_size = memory_size
        self.num_writes = num_writes

    def update_linkage(self, write_weights: tf.Tensor, prev_linkage: dict, training: bool = False) -> dict:
        # Unpack previous linkage
        prev_link = prev_linkage['link']  # [batch_size, num_writes, memory_size, memory_size]
        prev_precedence_weights = prev_linkage['precedence_weights']  # [batch_size, num_writes, memory_size]

        # Compute the sum over memory locations for each write head
        write_sum = tf.reduce_sum(write_weights, axis=2, keepdims=True)  # [batch_size, num_writes, 1]

        # Update precedence weights
        updated_precedence_weights = (
                                             1 - write_sum) * prev_precedence_weights + write_weights  # [batch_size, num_writes, memory_size]

        # Compute outer products for new link entries
        write_weights_i = tf.expand_dims(write_weights, axis=3)  # [batch_size, num_writes, memory_size, 1]
        precedence_weights_j = tf.expand_dims(updated_precedence_weights,
                                              axis=2)  # [batch_size, num_writes, 1, memory_size]

        # Compute the new link matrix
        new_link = write_weights_i * precedence_weights_j  # [batch_size, num_writes, memory_size, memory_size]

        # Remove self-links by zeroing the diagonal elements
        identity = tf.eye(self.memory_size, batch_shape=[self.num_writes],
                          dtype=tf.float32)  # [num_writes, memory_size, memory_size]
        identity = tf.expand_dims(identity, axis=0)  # [1, num_writes, memory_size, memory_size]
        identity = tf.tile(identity,
                           [tf.shape(new_link)[0], 1, 1, 1])  # [batch_size, num_writes, memory_size, memory_size]

        new_link = new_link * (1 - identity)  # [batch_size, num_writes, memory_size, memory_size]

        return {
            'link': new_link,
            'precedence_weights': updated_precedence_weights
        }

    def directional_read_weights(self, link: tf.Tensor, prev_read_weights: tf.Tensor,
                                 forward: bool = True) -> tf.Tensor:
        """
        Computes the forward or backward read weights.

        Args:
            link (tf.Tensor): Link matrices, shape [batch_size, num_writes, memory_size, memory_size].
            prev_read_weights (tf.Tensor): Previous read weights, shape [batch_size, num_reads, memory_size].
            forward (bool): Whether to compute forward weights.

        Returns:
            tf.Tensor: Directional read weights, shape [batch_size, num_reads, num_writes, memory_size].
        """
        if forward:
            link_transposed = link  # [batch_size, num_writes, memory_size, memory_size]
        else:
            link_transposed = tf.transpose(link,
                                           perm=[0, 1, 3, 2])  # [batch_size, num_writes, memory_size, memory_size]

        # 使用 tf.einsum 进行批量矩阵乘法
        # 'brm,bwmn->brwn' 的含义：
        # b: batch_size
        # r: num_reads
        # m: memory_size (read)
        # w: num_writes
        # n: memory_size (write)
        read_weights = tf.einsum('brm,bwmn->brwn', prev_read_weights, link_transposed)

        return read_weights  # [batch_size, num_reads, num_writes, memory_size]

    def state_size(self) -> dict:
        """
        Returns the sizes of the state components.

        Returns:
            dict: State sizes with keys 'link' and 'precedence_weights'.
        """
        return {
            'link': tf.TensorShape([self.num_writes, self.memory_size, self.memory_size]),
            'precedence_weights': tf.TensorShape([self.num_writes, self.memory_size])
        }


# 已修改：在计算读取权重前，确保 read_mode 已正确归一化
class DefaultReadWeightCalculator(ReadWeightCalculator):
    def __init__(self, temporal_linkage: DefaultTemporalLinkageUpdater, num_reads: int, num_writes: int):
        """
        Initializes the ReadWeightCalculator with a TemporalLinkage instance.
        """
        self.temporal_linkage = temporal_linkage
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.epsilon = 1e-6

    def compute(self, read_content_weights: tf.Tensor, prev_read_weights: tf.Tensor,
                link: tf.Tensor, read_mode: tf.Tensor, training: bool) -> tf.Tensor:
        # 正确归一化 read_mode，确保内容、前向、后向权重的合适分布
        read_mode_sum = tf.reduce_sum(read_mode, axis=-1, keepdims=True) + 1e-8
        read_mode = read_mode / read_mode_sum

        # 分离 content_mode, forward_mode, backward_mode
        content_mode = tf.expand_dims(read_mode[:, :, 0], axis=-1)  # [batch_size, num_reads, 1]
        forward_mode = read_mode[:, :, 1:1 + self.num_writes]  # [batch_size, num_reads, num_writes]
        backward_mode = read_mode[:, :, 1 + self.num_writes:]  # [batch_size, num_reads, num_writes]

        # 计算前向和后向权重
        forward_weights = self.temporal_linkage.directional_read_weights(link, prev_read_weights, forward=True)
        backward_weights = self.temporal_linkage.directional_read_weights(link, prev_read_weights, forward=False)

        # 前向与后向权重的加权求和
        forward_mode_exp = tf.expand_dims(forward_mode, axis=-1)
        backward_mode_exp = tf.expand_dims(backward_mode, axis=-1)

        forward_component = tf.reduce_sum(forward_mode_exp * forward_weights, axis=2)
        backward_component = tf.reduce_sum(backward_mode_exp * backward_weights, axis=2)

        # 计算最终读取权重
        read_weights = content_mode * read_content_weights + forward_component + backward_component

        # 对 memory_size 维度执行 Softmax 归一化，确保读取权重的和为 1
        read_weights = tf.nn.softmax(read_weights, axis=-1)

        # 调试输出
        tf.print("read_weights:", read_weights)

        # 确保归一化后的权重和为 1
        tf.debugging.assert_near(
            tf.reduce_sum(read_weights, axis=-1),
            tf.ones_like(tf.reduce_sum(read_weights, axis=-1)),
            atol=1e-6,
            message="Read weights do not sum to 1."
        )

        return read_weights  # [batch_size, num_reads, memory_size]


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


def get_default_config(memory_size: int, num_writes: int, num_reads: int, word_size: int) -> Dict[str, Any]:
    return {
        'WriteWeightCalculator': {
            'class_path': 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultWriteWeightCalculator',
            'memory_size': memory_size,  # 动态设置
            'num_writes': num_writes  # 动态设置
        },
        'ReadWeightCalculator': {
            'class_path': 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultReadWeightCalculator',
            'num_reads': num_reads,  # 动态设置
            'num_writes': num_writes  # 动态设置
        },
        'ContentWeightCalculator': {
            'class_path': 'thinker_ai.agent.memory.humanoid_memory.dnc.default_component.DefaultContentWeightCalculator',
            'num_heads': num_writes,  # 动态设置
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
