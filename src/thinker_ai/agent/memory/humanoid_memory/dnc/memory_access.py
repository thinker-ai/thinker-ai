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

        factory = ComponentFactory(
            config=config,
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes
        )

        components = factory.create_all_components()

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

        return interface_size

    def call(self, inputs: Dict[str, Any], training: bool = False):
        controller_output = inputs['inputs']
        prev_state = inputs['prev_state']

        # 检查缓存中是否有持久化的内存状态
        cached_memory_state = self.cache_manager.read_from_cache('memory_state')
        if cached_memory_state is not None:
            prev_state = cached_memory_state

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

        # 更新后的内存状态持久化到缓存
        self.cache_manager.write_to_cache('memory_state', final_state)

        return {'read_words': read_words, 'final_state': final_state}

    def _process_time_step(self, prev_state: BatchAccessState,
                           interfaces: Dict[str, tf.Tensor], training: bool) -> Tuple[tf.Tensor, BatchAccessState]:
        # Compute write content weights
        write_content_weights = self.content_weight_calculator.compute(
            keys=interfaces['write_keys'],
            strengths=interfaces['write_strengths'],
            memory=prev_state.memory
        )

        # Compute write weights
        write_weights = self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=interfaces['allocation_gates'],
            write_gate=interfaces['write_gates'],
            prev_usage=prev_state.usage,
            training=training
        )

        # Update usage
        usage = self.usage_updater.update_usage(
            write_weights=write_weights,
            free_gates=interfaces['free_gates'],
            prev_read_weights=prev_state.read_weights,
            prev_usage=prev_state.usage,
            training=training
        )

        # Update memory
        memory = self.memory_updater.update_memory(
            memory=prev_state.memory,
            write_weights=write_weights,
            erase_vectors=interfaces['erase_vectors'],
            write_vectors=interfaces['write_vectors']
        )

        # Update linkage
        linkage = self.temporal_linkage_updater.update_linkage(
            write_weights=tf.reduce_sum(write_weights, axis=1),
            prev_linkage=prev_state.linkage,
            training=training
        )

        # Compute read content weights
        read_content_weights = self.content_weight_calculator.compute(
            keys=interfaces['read_keys'],
            strengths=interfaces['read_strengths'],
            memory=memory
        )

        # Compute read weights
        read_weights = self.read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            prev_read_weights=prev_state.read_weights,
            link=linkage['link'],
            read_mode=interfaces['read_modes'],
            training=training
        )

        # Read from memory
        read_words = tf.matmul(read_weights, memory)  # Shape: [batch_size, num_reads, word_size]

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

    def query_history(self, query_vector: tf.Tensor, top_k: int = 2) -> tf.Tensor:
        """
        基于与当前输入内容相关性查询历史记录。

        Args:
            query_vector (tf.Tensor): [batch_size, word_size]
            top_k (int): 要检索的相关记录数量

        Returns:
            related_records (tf.Tensor): [batch_size, top_k, word_size]
        """
        # 计算余弦相似度
        query_norm = tf.nn.l2_normalize(query_vector, axis=-1)  # [batch_size, word_size]
        memory_norm = tf.nn.l2_normalize(self.state.memory, axis=-1)  # [batch_size, memory_size, word_size]

        # 计算相似度
        similarity = tf.einsum('bw,mw->bm', query_norm, memory_norm)  # [batch_size, memory_size]

        # 获取 top_k 相似度最高的内存索引
        top_k_values, top_k_indices = tf.nn.top_k(similarity, k=top_k, sorted=True)  # [batch_size, top_k]

        # 根据索引从 memory 中检索相关记录
        related_records = tf.gather(self.state.memory, top_k_indices, batch_dims=1)  # [batch_size, top_k, word_size]

        return related_records
