from typing import Dict
import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import (
    WriteAllocation,
    CosineWeights,
    TemporalLinkage,
    UsageUpdate
)
from thinker_ai.agent.memory.humanoid_memory.dnc_new.component_interface import (
    MemoryUpdater, ReadWeightCalculator, TemporalLinkageUpdater, UsageUpdater,
    ContentWeightCalculator, WriteWeightCalculator
)

from typing import Optional, Callable, Dict


class DefaultWriteWeightCalculator(WriteWeightCalculator):
    def __init__(self, write_allocation: WriteAllocation):
        """
        初始化 WriteWeightCalculator，传入 WriteAllocation 实例。
        """
        self.write_allocation = write_allocation

    def compute_write_weights(self, write_content_weights: tf.Tensor, allocation_gate: tf.Tensor,
                              write_gate: tf.Tensor, prev_usage: tf.Tensor, training: bool) -> tf.Tensor:
        """
        计算最终的写入权重，调用 WriteAllocation 的现有逻辑。
        """
        # 直接调用 WriteAllocation 的逻辑计算写入权重
        write_weights = self.write_allocation({
            'write_content_weights': write_content_weights,
            'allocation_gate': allocation_gate,
            'write_gate': write_gate
        }, training=training)
        return write_weights


class DefaultReadWeightCalculator(ReadWeightCalculator):
    def __init__(self, temporal_linkage: TemporalLinkage):
        """
        初始化 ReadWeightCalculator，传入 TemporalLinkage 实现类实例。
        """
        self.temporal_linkage = temporal_linkage

    def compute_read_weights(self, read_content_weights: tf.Tensor, prev_read_weights: tf.Tensor,
                             link: tf.Tensor, read_mode: tf.Tensor, training: bool) -> tf.Tensor:
        """
        计算读取权重，结合内容权重和方向性权重，使用 CosineWeights 和 TemporalLinkage 的逻辑。
        """
        num_writes = tf.shape(link)[1]
        # 分割 read_mode
        content_mode = read_mode[:, :, 0]  # [batch_size, num_reads]
        forward_mode = read_mode[:, :, 1:1 + num_writes]  # [batch_size, num_reads, num_writes]
        backward_mode = read_mode[:, :, 1 + num_writes:]  # [batch_size, num_reads, num_writes]

        # 计算前向和后向权重（使用传入的 TemporalLinkage 实例）
        forward_weights = self.temporal_linkage.directional_read_weights(link, prev_read_weights, forward=True)
        backward_weights = self.temporal_linkage.directional_read_weights(link, prev_read_weights, forward=False)

        # 结合权重
        content_component = tf.expand_dims(content_mode, axis=-1) * read_content_weights
        forward_component = tf.reduce_sum(tf.expand_dims(forward_mode, axis=-1) * forward_weights, axis=2)
        backward_component = tf.reduce_sum(tf.expand_dims(backward_mode, axis=-1) * backward_weights, axis=2)

        # 最终读取权重
        read_weights = content_component + forward_component + backward_component
        return read_weights


class DefaultMemoryUpdater(MemoryUpdater):
    def update_memory(self, memory: tf.Tensor, write_weights: tf.Tensor, erase_vectors: tf.Tensor,
                      write_vectors: tf.Tensor) -> tf.Tensor:
        """
        更新内存，使用 MemoryUpdater 的默认逻辑，保留原有逻辑，并结合 WriteAllocation 和 UsageUpdate。
        """
        # 使用 WriteAllocation 更新写入权重
        write_weights_expanded = tf.expand_dims(write_weights, axis=-1)
        erase_vectors_expanded = tf.expand_dims(erase_vectors, axis=2)
        erase_gate = tf.reduce_prod(1 - write_weights_expanded * erase_vectors_expanded, axis=1)

        memory_erased = memory * erase_gate
        write_vectors_expanded = tf.expand_dims(write_vectors, axis=2)
        add_matrix = tf.reduce_sum(write_weights_expanded * write_vectors_expanded, axis=1)

        memory_updated = memory_erased + add_matrix
        return memory_updated


class DefaultTemporalLinkageUpdater(TemporalLinkageUpdater):
    def __init__(self, memory_size: int, num_writes: int, epsilon: float = 1e-6):
        self.temporal_linkage = TemporalLinkage(memory_size=memory_size, num_writes=num_writes, epsilon=epsilon)

    def update_linkage(self, write_weights: tf.Tensor, prev_linkage: Dict[str, tf.Tensor],
                       training: bool) -> Dict[str, tf.Tensor]:
        """
        调用 TemporalLinkage 计算更新链路。
        """
        return self.temporal_linkage({
            'write_weights': write_weights,
            'prev_linkage': prev_linkage
        }, training=training)


class DefaultUsageUpdater(UsageUpdater):
    def __init__(self, memory_size: int, num_writes: int, num_reads: int, epsilon: float = 1e-6):
        self.usage_update = UsageUpdate(memory_size=memory_size, num_writes=num_writes, num_reads=num_reads,
                                        epsilon=epsilon)

    def update_usage(self, write_weights: tf.Tensor, free_gate: tf.Tensor, read_weights: tf.Tensor,
                     prev_usage: tf.Tensor, training: bool) -> tf.Tensor:
        """
        使用 UsageUpdate 的现有逻辑更新使用率。
        """
        return self.usage_update({
            'write_weights': write_weights,
            'free_gate': free_gate,
            'read_weights': read_weights,
            'prev_usage': prev_usage
        }, training=training)


class DefaultContentWeightCalculator(ContentWeightCalculator):
    def __init__(self, num_heads: int, word_size: int, epsilon: float = 1e-6):
        self.cosine_weights = CosineWeights(num_heads=num_heads, word_size=word_size, epsilon=epsilon)

    def compute_content_weights(self, keys: tf.Tensor, strengths: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        """
        调用 CosineWeights 计算内容权重。
        """
        return self.cosine_weights({
            'keys': keys,
            'strengths': strengths,
            'memory': memory
        })
