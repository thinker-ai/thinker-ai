# memory_access.py
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple
from collections import namedtuple

from thinker_ai.agent.memory.humanoid_memory.dnc.cache_manager import CacheManager
# Define the state namedtuple for easier state management
BatchAccessState = namedtuple('BatchAccessState', ('memory', 'read_weights', 'write_weights',
                                                   'linkage', 'usage', 'read_words'))


class MemoryAccess(tf.keras.layers.Layer):
    def __init__(
            self,
            memory_size: int,
            word_size: int,
            num_reads: int,
            num_writes: int,
            controller_output_size: int,
            cache_manager: CacheManager = None,
            name: str = 'memory_access',
            config: Optional[Dict[str, Any]] = None
    ):
        super(MemoryAccess, self).__init__(name=name)
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.controller_output_size = controller_output_size
        self.cache_manager = cache_manager or CacheManager(max_cache_size=memory_size)

        # Define the number of read modes (as per DNC paper)
        self.num_read_modes = 3

        # Calculate the size of the interface vector
        self.interface_size = self._calculate_interface_size()

        # 计算并存储接口向量各部分的索引
        self.interface_vector_indices = self._calculate_interface_vector_indices()

        # Define the interface layer, mapping controller output to interface vector
        self.interface_layer = tf.keras.layers.Dense(
            units=self.interface_size,
            activation=None,
            use_bias=True,
            name='interface_layer'
        )

        # Initialize components (assumed to be defined elsewhere in your codebase)
        from thinker_ai.agent.memory.humanoid_memory.dnc.component_factory import ComponentFactory
        from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import get_default_config

        if config is None:
            # Use default configuration
            config = get_default_config(
                memory_size=self.memory_size,
                num_writes=self.num_writes,
                num_reads=self.num_reads,
                word_size=self.word_size
            )

        self.component_factory = ComponentFactory(
            config=config,
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes
        )

        components = self.component_factory.create_all_components()

        # Ensure all components are properly initialized
        self.content_weight_calculator = components.get('content_weight_calculator')
        self.write_weight_calculator = components.get('write_weight_calculator')
        self.temporal_linkage_updater = components.get('temporal_linkage_updater')
        self.read_weight_calculator = components.get('read_weight_calculator')
        self.usage_updater = components.get('usage_updater')
        self.memory_updater = components.get('memory_updater')

        if None in [self.content_weight_calculator, self.write_weight_calculator, self.temporal_linkage_updater,
                    self.read_weight_calculator, self.usage_updater, self.memory_updater]:
            raise ValueError("One or more components were not properly initialized.")

    def _calculate_interface_size(self):
        # Calculate the size of the interface vector based on the number of reads, writes, and word size
        num_read_keys = self.num_reads * self.word_size
        num_read_strengths = self.num_reads
        num_write_keys = self.num_writes * self.word_size
        num_write_strengths = self.num_writes
        num_erase_vectors = self.num_writes * self.word_size
        num_write_vectors = self.num_writes * self.word_size
        num_free_gates = self.num_reads
        num_allocation_gates = self.num_writes
        num_write_gates = self.num_writes
        num_read_modes = self.num_reads * self.num_read_modes

        interface_size = (
                num_read_keys +
                num_read_strengths +
                num_write_keys +
                num_write_strengths +
                num_erase_vectors +
                num_write_vectors +
                num_free_gates +
                num_allocation_gates +
                num_write_gates +
                num_read_modes
        )
        tf.print(f"Calculating interface size: {interface_size} = "
                 f"{num_read_keys} + {num_read_strengths} + {num_write_keys} + "
                 f"{num_write_strengths} + {num_erase_vectors} + {num_write_vectors} + "
                 f"{num_free_gates} + {num_allocation_gates} + {num_write_gates} + {num_read_modes}")
        return interface_size

    def _calculate_interface_vector_indices(self):
        indices = {}
        idx = 0
        r = self.num_reads
        w = self.num_writes
        W = self.word_size
        R = self.num_read_modes

        # Read keys
        indices['read_keys_start'] = idx
        indices['read_keys_end'] = idx + r * W
        idx = indices['read_keys_end']

        # Read strengths
        indices['read_strengths_start'] = idx
        indices['read_strengths_end'] = idx + r
        idx = indices['read_strengths_end']

        # Write keys
        indices['write_keys_start'] = idx
        indices['write_keys_end'] = idx + w * W
        idx = indices['write_keys_end']

        # Write strengths
        indices['write_strengths_start'] = idx
        indices['write_strengths_end'] = idx + w
        idx = indices['write_strengths_end']

        # Erase vectors
        indices['erase_vectors_start'] = idx
        indices['erase_vectors_end'] = idx + w * W
        idx = indices['erase_vectors_end']

        # Write vectors
        indices['write_vectors_start'] = idx
        indices['write_vectors_end'] = idx + w * W
        idx = indices['write_vectors_end']

        # Free gates
        indices['free_gates_start'] = idx
        indices['free_gates_end'] = idx + r
        idx = indices['free_gates_end']

        # Allocation gates
        indices['allocation_gates_start'] = idx
        indices['allocation_gates_end'] = idx + w
        idx = indices['allocation_gates_end']

        # Write gates
        indices['write_gates_start'] = idx
        indices['write_gates_end'] = idx + w
        idx = indices['write_gates_end']

        # Read modes
        indices['read_modes_start'] = idx
        indices['read_modes_end'] = idx + r * R
        idx = indices['read_modes_end']

        return indices

    def call(self, inputs: Dict[str, Any], training: bool = False):
        controller_output = inputs['inputs']
        prev_state = inputs['prev_state']

        # 检查缓存中是否有持久化的内存状态
        cached_memory_state = self.cache_manager.read_from_cache('memory_state')
        if cached_memory_state is not None:
            prev_state = cached_memory_state
            tf.print("Loaded cached memory state.")

        tf.debugging.assert_equal(
            tf.shape(controller_output)[-1],
            self.controller_output_size,
            message="controller_output size mismatch"
        )

        if len(controller_output.shape) == 3:
            # Sequence input
            seq_len = tf.shape(controller_output)[1]

            state = prev_state
            read_words_list = []
            for t in range(seq_len):
                controller_output_t = controller_output[:, t, :]
                interface_vector = self.interface_layer(controller_output_t)
                interfaces = self._parse_interface_vector(interface_vector)
                read_words_t, state = self._process_time_step(state, interfaces, training)
                read_words_list.append(read_words_t)
            read_words = tf.stack(read_words_list, axis=1)
            final_state = state
        else:
            # Single time step
            interface_vector = self.interface_layer(controller_output)
            interfaces = self._parse_interface_vector(interface_vector)
            read_words, final_state = self._process_time_step(prev_state, interfaces, training)

        tf.print("Final state memory:", final_state.memory)
        tf.print("Final state read_weights:", final_state.read_weights)
        tf.print("Final state write_weights:", final_state.write_weights)

        # 更新后的内存状态持久化到缓存
        self.cache_manager.write_to_cache('memory_state', final_state)

        return {
            'read_words': read_words,
            'final_state': final_state,
            'write_weights': final_state.write_weights  # 添加这一行
        }

    def _process_time_step(self, prev_state: BatchAccessState,
                           interfaces: Dict[str, tf.Tensor], training: bool) -> Tuple[tf.Tensor, BatchAccessState]:
        # Compute write content weights
        write_content_weights = self.content_weight_calculator.compute(
            keys=interfaces['write_keys'],
            strengths=interfaces['write_strengths'],
            memory=prev_state.memory
        )
        tf.print("Write content weights:", write_content_weights)

        # Compute write weights
        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=interfaces['allocation_gates'],
            write_gate=interfaces['write_gates'],
            prev_usage=prev_state.usage,
            training=training
        )
        tf.print("Write weights:", write_weights)

        # Update usage
        usage = self.usage_updater.update_usage(
            write_weights=write_weights,
            free_gates=interfaces['free_gates'],
            prev_read_weights=prev_state.read_weights,
            prev_usage=prev_state.usage,
            training=training
        )
        tf.print("Updated usage:", usage)

        # Update memory
        memory = self.memory_updater.update_memory(
            memory=prev_state.memory,
            write_weights=write_weights,
            erase_vectors=interfaces['erase_vectors'],
            write_vectors=interfaces['write_vectors']
        )
        tf.print("Updated memory:", memory)

        # Update linkage
        linkage = self.temporal_linkage_updater.update_linkage(
            write_weights=tf.reduce_sum(write_weights, axis=1),
            prev_linkage=prev_state.linkage,
            training=training
        )
        tf.print("Updated linkage:", linkage)

        # Compute read content weights
        read_content_weights = self.content_weight_calculator.compute(
            keys=interfaces['read_keys'],
            strengths=interfaces['read_strengths'],
            memory=memory
        )
        tf.print("Read content weights:", read_content_weights)

        # Compute read weights
        read_weights = self.read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            prev_read_weights=prev_state.read_weights,
            link=linkage['link'],
            read_mode=interfaces['read_modes'],
            training=training
        )
        tf.print("Read weights:", read_weights)

        # Read from memory
        read_words = tf.matmul(read_weights, memory)  # Shape: [batch_size, num_reads, word_size]
        tf.print("Read words:", read_words)

        # Prepare the next state
        next_state = BatchAccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage,
            usage=usage,
            read_words=read_words
        )

        return read_words, next_state

    def _parse_interface_vector(self, interface_vector):
        # Parse the interface vector into its components
        batch_size = tf.shape(interface_vector)[0]
        r = self.num_reads
        w = self.num_writes
        W = self.word_size
        R = self.num_read_modes

        idx = 0

        # Read keys
        read_keys = tf.reshape(interface_vector[:, idx:idx + r * W], [batch_size, r, W])
        idx += r * W

        # Read strengths
        read_strengths = tf.nn.softplus(interface_vector[:, idx:idx + r])
        idx += r

        # Write keys
        write_keys = tf.reshape(interface_vector[:, idx:idx + w * W], [batch_size, w, W])
        idx += w * W

        # Write strengths
        write_strengths = tf.nn.softplus(interface_vector[:, idx:idx + w])
        idx += w

        # Erase vectors
        erase_vectors = tf.nn.sigmoid(tf.reshape(interface_vector[:, idx:idx + w * W], [batch_size, w, W]))
        idx += w * W

        # Write vectors
        write_vectors = tf.reshape(interface_vector[:, idx:idx + w * W], [batch_size, w, W])
        idx += w * W

        # Free gates
        free_gates = tf.nn.sigmoid(interface_vector[:, idx:idx + r])
        idx += r

        # Allocation gates
        allocation_gates = tf.nn.sigmoid(interface_vector[:, idx:idx + w])
        idx += w

        # Write gates
        write_gates = tf.nn.sigmoid(interface_vector[:, idx:idx + w])
        idx += w

        # Read modes
        read_modes = tf.nn.softmax(tf.reshape(interface_vector[:, idx:idx + r * R], [batch_size, r, R]), axis=-1)
        idx += r * R

        # Ensure all components have been extracted
        expected_size = self.interface_size
        tf.debugging.assert_equal(idx, expected_size, message="Interface vector size mismatch")

        return {
            'read_keys': read_keys,
            'read_strengths': read_strengths,
            'write_keys': write_keys,
            'write_strengths': write_strengths,
            'erase_vectors': erase_vectors,
            'write_vectors': write_vectors,
            'free_gates': free_gates,
            'allocation_gates': allocation_gates,
            'write_gates': write_gates,
            'read_modes': read_modes
        }

    def get_initial_state(self, batch_size: int) -> BatchAccessState:
        # Initialize memory and other state components
        memory = tf.zeros([batch_size, self.memory_size, self.word_size], dtype=tf.float32)
        read_weights = tf.fill([batch_size, self.num_reads, self.memory_size], 1.0 / self.memory_size)
        write_weights = tf.fill([batch_size, self.num_writes, self.memory_size], 1.0 / self.memory_size)
        linkage = {
            'link': tf.zeros([batch_size, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, self.memory_size], dtype=tf.float32)
        }
        usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)
        read_words = tf.zeros([batch_size, self.num_reads, self.word_size], dtype=tf.float32)

        return BatchAccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage,
            usage=usage,
            read_words=read_words
        )

    def query_history(self, query_vector: tf.Tensor, top_k: int = 1, read_strength: float = 10.0):
        """
        查询记忆中与查询向量相似的记录。

        参数：
            query_vector: [batch_size, word_size]
            top_k: int，返回的记录数量
            read_strength: float，内容寻址的强度

        返回：
            related_records: [batch_size, top_k, word_size]
        """
        batch_size = tf.shape(query_vector)[0]
        # 创建接口向量
        indices = self.interface_vector_indices

        # 读取键
        read_keys = tf.reshape(query_vector, [batch_size, self.num_reads * self.word_size])  # [batch_size, r * W]

        # 读取强度
        read_strengths = tf.fill([batch_size, self.num_reads], read_strength)  # [batch_size, r]

        # 其余部分为零
        remaining_size = self.interface_size - indices['read_strengths_end']
        zeros_remaining = tf.zeros([batch_size, remaining_size], dtype=tf.float32)

        # 构建接口向量
        interface_vector = tf.concat([read_keys, read_strengths, zeros_remaining], axis=1)  # [batch_size, interface_size]

        # 解析接口向量
        interfaces = self._parse_interface_vector(interface_vector)

        # 获取初始状态
        prev_state = self.get_initial_state(batch_size)

        # 处理时间步
        read_words, final_state = self._process_time_step(prev_state, interfaces, training=False)

        # 提取 top_k 个读取内容
        return read_words[:, :top_k, :]