# component_interface.py
import collections
from abc import ABC, abstractmethod
from typing import Dict, Optional
import tensorflow as tf

# 定义 BatchAccessState
BatchAccessState = collections.namedtuple('BatchAccessState', [
    'memory',  # [batch_size, memory_size, word_size]
    'read_weights',  # [batch_size, time_steps, num_reads, memory_size]
    'write_weights',  # [batch_size, time_steps, num_writes, memory_size]
    'linkage',  # {'link': [batch_size, num_writes, memory_size, memory_size],
    #  'precedence_weights': [batch_size, num_writes, memory_size]}
    'usage',  # [batch_size, memory_size]
    'read_words'  # [batch_size, num_reads, word_size] 或 None
])


# 定义抽象类
class WriteWeightCalculator(ABC):
    @abstractmethod
    def compute(self, write_content_weights: tf.Tensor, allocation_gate: tf.Tensor,
                              write_gate: tf.Tensor, prev_usage: tf.Tensor, training: bool) -> tf.Tensor:
        pass


class ReadWeightCalculator(ABC):
    @abstractmethod
    def compute(self, read_content_weights: tf.Tensor, prev_read_weights: tf.Tensor,
                             link: tf.Tensor, read_mode: tf.Tensor, training: bool) -> tf.Tensor:
        pass


class MemoryUpdater(ABC):
    @abstractmethod
    def update_memory(self, memory: tf.Tensor, write_weights: tf.Tensor, erase_vectors: tf.Tensor,
                      write_vectors: tf.Tensor) -> tf.Tensor:
        pass


class UsageUpdater(ABC):
    @abstractmethod
    def update_usage(self, write_weights: tf.Tensor, free_gate: tf.Tensor, read_weights: tf.Tensor,
                     prev_usage: tf.Tensor, training: bool=False) -> tf.Tensor:
        pass


class TemporalLinkageUpdater(ABC):
    @abstractmethod
    def update_linkage(self, write_weights: tf.Tensor, prev_linkage: Dict[str, tf.Tensor],
                       training: bool=False) -> Dict[str, tf.Tensor]:
        pass

    @abstractmethod
    def state_size(self) -> tf.Tensor:
        pass


class ContentWeightCalculator(ABC):
    @abstractmethod
    def compute(self, keys: tf.Tensor, strengths: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        pass
