# memory_access.py
from typing import Optional, Dict, Any, Tuple

import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.component_interface import BatchAccessState
from thinker_ai.agent.memory.humanoid_memory.dnc.component_factory import ComponentFactory


class MemoryAccess(tf.keras.layers.Layer):
    def __init__(
            self,
            memory_size: int,
            word_size: int,
            num_reads: int,
            num_writes: int,
            epsilon: float = 1e-6,
            name: str = 'memory_access',
            config: Optional[Dict[str, Any]] = None  # 添加配置参数
    ):
        super(MemoryAccess, self).__init__(name=name)
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.epsilon = epsilon

        # 使用工厂根据配置初始化组件
        factory = ComponentFactory(config or {})
        components = factory.create_all_components()

        self.content_weight_calculator = components['content_weight_calculator']
        self.write_weight_calculator = components['write_weight_calculator']
        self.temporal_linkage_updater = components['temporal_linkage_updater']
        self.read_weight_calculator = components['read_weight_calculator']
        self.usage_updater = components['usage_updater']
        self.memory_updater = components['memory_updater']

        # 定义子层（具有可训练参数的层）
        self.write_vectors_layer = tf.keras.layers.Dense(
            units=self.num_writes * self.word_size,
            activation=None,
            name='write_vectors',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.erase_vectors_layer = tf.keras.layers.Dense(
            units=self.num_writes * self.word_size,
            activation='sigmoid',
            name='erase_vectors',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.write_gate_layer = tf.keras.layers.Dense(
            units=self.num_writes,
            activation='sigmoid',
            name='write_gate',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.allocation_gate_layer = tf.keras.layers.Dense(
            units=self.num_writes,
            activation='sigmoid',
            name='allocation_gate',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.free_gate_layer = tf.keras.layers.Dense(
            units=self.num_reads,
            activation='sigmoid',
            name='free_gate',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.read_mode_layer = tf.keras.layers.Dense(
            units=self.num_reads * (1 + 2 * self.num_writes),
            activation=None,
            name='read_mode',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.write_keys_layer = tf.keras.layers.Dense(
            units=self.num_writes * self.word_size,
            activation=None,
            name='write_keys',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.write_strengths_layer = tf.keras.layers.Dense(
            units=self.num_writes,
            activation='softplus',
            name='write_strengths',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.read_keys_layer = tf.keras.layers.Dense(
            units=self.num_reads * self.word_size,
            activation=None,
            name='read_keys',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.read_strengths_layer = tf.keras.layers.Dense(
            units=self.num_reads,
            activation='softplus',
            name='read_strengths',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> Dict[str, tf.Tensor]:
        """
        处理输入并执行 MemoryAccess 操作。

        Args:
            inputs (Dict[str, tf.Tensor]): 输入字典，包含 'inputs' 和 'prev_state'
                - 'inputs': [batch_size, sequence_length, input_size]
                - 'prev_state': BatchAccessState namedtuple
            training (bool): 是否在训练模式

        Returns:
            Dict[str, tf.Tensor]: 输出字典，包含 'read_words' 和 'final_state'
        """
        controller_output = inputs['inputs']  # [batch_size, sequence_length, input_size]
        prev_state = inputs['prev_state']  # BatchAccessState

        batch_size = tf.shape(controller_output)[0]
        sequence_length = tf.shape(controller_output)[1]

        next_state = prev_state
        read_words_all = []

        for t in range(sequence_length):
            controller_output_t = controller_output[:, t, :]  # [batch_size, input_size]

            # 1. Generate write-related parameters
            write_vectors, erase_vectors, write_gate, allocation_gate, free_gate = self._generate_write_parameters(controller_output_t)

            # 2. Generate read mode
            read_mode = self._generate_read_mode(controller_output_t)

            # 3. Generate write keys and strengths
            write_keys, write_strengths = self._generate_write_keys_strengths(controller_output_t)

            # 4. Generate read keys and strengths
            read_keys, read_strengths = self._generate_read_keys_strengths(controller_output_t)

            # 5. Compute content weights
            write_content_weights = self.content_weight_calculator.compute(
                keys=write_keys,
                strengths=write_strengths,
                memory=next_state.memory
            )  # [batch_size, num_writes, memory_size]

            read_content_weights = self.content_weight_calculator.compute(
                keys=read_keys,
                strengths=read_strengths,
                memory=next_state.memory
            )  # [batch_size, num_reads, memory_size]

            # 6. Compute write weights
            final_write_weights = self.write_weight_calculator.compute(
                write_content_weights=write_content_weights,
                allocation_gate=allocation_gate,
                write_gate=write_gate,
                prev_usage=next_state.usage,
                training=training
            )  # [batch_size, num_writes, memory_size]

            # 7. Update memory
            memory_updated = self.memory_updater.update_memory(
                memory=next_state.memory,
                write_weights=final_write_weights,
                erase_vectors=erase_vectors,
                write_vectors=write_vectors
            )  # [batch_size, memory_size, word_size]

            # 8. Update linkage
            linkage_updated = self.temporal_linkage_updater.update_linkage(
                write_weights=final_write_weights,
                prev_linkage=next_state.linkage,
                training=training
            )

            # 9. Update usage
            read_weights_prev = next_state.read_weights[:, -1] if next_state.read_weights.shape[1] > 0 else tf.zeros_like(write_content_weights)
            usage_updated = self.usage_updater.update_usage(
                write_weights=final_write_weights,
                free_gate=free_gate,
                read_weights=read_weights_prev,
                prev_usage=next_state.usage,
                training=training
            )

            # 10. Compute read weights
            read_weights = self.read_weight_calculator.compute(
                read_content_weights=read_content_weights,
                prev_read_weights=read_weights_prev,
                link=linkage_updated['link'],
                read_mode=read_mode,
                training=training
            )  # [batch_size, num_reads, memory_size]

            # 11. Read words
            read_words = self._read_words(read_weights, memory_updated)

            # Update state
            next_state = BatchAccessState(
                memory=memory_updated,
                read_weights=tf.concat([next_state.read_weights, tf.expand_dims(read_weights, axis=1)], axis=1),
                write_weights=tf.concat([next_state.write_weights, tf.expand_dims(final_write_weights, axis=1)], axis=1),
                linkage=linkage_updated,
                usage=usage_updated,
                read_words=read_words
            )

            read_words_all.append(read_words)  # [batch_size, num_reads, word_size]

        # Stack read_words over time steps
        read_words = tf.stack(read_words_all, axis=1)  # [batch_size, sequence_length, num_reads, word_size]

        return {'read_words': read_words, 'final_state': next_state}

    def _generate_write_parameters(self, controller_output_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Generate write-related parameters from controller output at time t.

        Args:
            controller_output_t (tf.Tensor): [batch_size, input_size]

        Returns:
            Tuple[tf.Tensor]: write_vectors, erase_vectors, write_gate, allocation_gate, free_gate
        """
        write_vectors = self.write_vectors_layer(controller_output_t)  # [batch_size, num_writes * word_size]
        write_vectors = tf.reshape(write_vectors, [tf.shape(write_vectors)[0], self.num_writes, self.word_size])  # [batch_size, num_writes, word_size]

        erase_vectors = self.erase_vectors_layer(controller_output_t)  # [batch_size, num_writes * word_size]
        erase_vectors = tf.reshape(erase_vectors, [tf.shape(erase_vectors)[0], self.num_writes, self.word_size])  # [batch_size, num_writes, word_size]

        write_gate = self.write_gate_layer(controller_output_t)  # [batch_size, num_writes]
        allocation_gate = self.allocation_gate_layer(controller_output_t)  # [batch_size, num_writes]
        free_gate = self.free_gate_layer(controller_output_t)  # [batch_size, num_reads]

        return write_vectors, erase_vectors, write_gate, allocation_gate, free_gate

    def _generate_read_mode(self, controller_output_t: tf.Tensor) -> tf.Tensor:
        """
        Generate read mode from controller output at time t.

        Args:
            controller_output_t (tf.Tensor): [batch_size, input_size]

        Returns:
            tf.Tensor: [batch_size, num_reads, 1 + 2*num_writes]
        """
        read_mode = self.read_mode_layer(controller_output_t)  # [batch_size, num_reads * (1 + 2*num_writes)]
        read_mode = tf.reshape(read_mode, [tf.shape(read_mode)[0], self.num_reads, 1 + 2 * self.num_writes])  # [batch_size, num_reads, 1 + 2*num_writes]
        return read_mode

    def _generate_write_keys_strengths(self, controller_output_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate write keys and strengths from controller output at time t.

        Args:
            controller_output_t (tf.Tensor): [batch_size, input_size]

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: write_keys, write_strengths
        """
        write_keys = self.write_keys_layer(controller_output_t)  # [batch_size, num_writes * word_size]
        write_keys = tf.reshape(write_keys, [tf.shape(write_keys)[0], self.num_writes, self.word_size])  # [batch_size, num_writes, word_size]

        write_strengths = self.write_strengths_layer(controller_output_t)  # [batch_size, num_writes]
        return write_keys, write_strengths

    def _generate_read_keys_strengths(self, controller_output_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate read keys and strengths from controller output at time t.

        Args:
            controller_output_t (tf.Tensor): [batch_size, input_size]

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: read_keys, read_strengths
        """
        read_keys = self.read_keys_layer(controller_output_t)  # [batch_size, num_reads * word_size]
        read_keys = tf.reshape(read_keys, [tf.shape(read_keys)[0], self.num_reads, self.word_size])  # [batch_size, num_reads, word_size]

        read_strengths = self.read_strengths_layer(controller_output_t)  # [batch_size, num_reads]
        return read_keys, read_strengths

    def _compute_content_weights(self, keys: tf.Tensor, strengths: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        """
        Compute content weights.

        Args:
            keys (tf.Tensor): [batch_size, heads, word_size]
            strengths (tf.Tensor): [batch_size, heads]
            memory (tf.Tensor): [batch_size, memory_size, word_size]

        Returns:
            tf.Tensor: [batch_size, heads, memory_size]
        """
        return self.content_weight_calculator.compute(
            keys=keys,
            strengths=strengths,
            memory=memory
        )

    def _compute_write_weights(self, write_content_weights: tf.Tensor, allocation_gate: tf.Tensor, write_gate: tf.Tensor, prev_usage: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Compute write weights.

        Args:
            write_content_weights (tf.Tensor): [batch_size, num_writes, memory_size]
            allocation_gate (tf.Tensor): [batch_size, num_writes]
            write_gate (tf.Tensor): [batch_size, num_writes]
            prev_usage (tf.Tensor): [batch_size, memory_size]
            training (bool): 是否在训练模式

        Returns:
            tf.Tensor: [batch_size, num_writes, memory_size]
        """
        return self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=prev_usage,
            training=training
        )

    def _update_memory(self, memory: tf.Tensor, write_weights: tf.Tensor, erase_vectors: tf.Tensor, write_vectors: tf.Tensor) -> tf.Tensor:
        """
        Update memory.

        Args:
            memory (tf.Tensor): [batch_size, memory_size, word_size]
            write_weights (tf.Tensor): [batch_size, num_writes, memory_size]
            erase_vectors (tf.Tensor): [batch_size, num_writes, word_size]
            write_vectors (tf.Tensor): [batch_size, num_writes, word_size]

        Returns:
            tf.Tensor: [batch_size, memory_size, word_size]
        """
        return self.memory_updater.update_memory(
            memory=memory,
            write_weights=write_weights,
            erase_vectors=erase_vectors,
            write_vectors=write_vectors
        )

    def _update_linkage(self, write_weights: tf.Tensor, prev_linkage: Dict[str, tf.Tensor], training: bool) -> Dict[str, tf.Tensor]:
        """
        Update linkage.

        Args:
            write_weights (tf.Tensor): [batch_size, num_writes, memory_size]
            prev_linkage (Dict[str, tf.Tensor]): Previous linkage
            training (bool): 是否在训练模式

        Returns:
            Dict[str, tf.Tensor]: Updated linkage
        """
        return self.temporal_linkage_updater.update_linkage(
            write_weights=write_weights,
            prev_linkage=prev_linkage,
            training=training
        )

    def _update_usage(self, write_weights: tf.Tensor, free_gate: tf.Tensor, read_weights_prev: tf.Tensor, prev_usage: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Update usage.

        Args:
            write_weights (tf.Tensor): [batch_size, num_writes, memory_size]
            free_gate (tf.Tensor): [batch_size, num_reads]
            read_weights_prev (tf.Tensor): [batch_size, num_writes, memory_size]
            prev_usage (tf.Tensor): [batch_size, memory_size]
            training (bool): 是否在训练模式

        Returns:
            tf.Tensor: [batch_size, memory_size]
        """
        return self.usage_updater.update_usage(
            write_weights=write_weights,
            free_gate=free_gate,
            read_weights=read_weights_prev,
            prev_usage=prev_usage,
            training=training
        )

    def _compute_read_weights(self, read_content_weights: tf.Tensor, prev_read_weights: tf.Tensor, link: tf.Tensor, read_mode: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Compute read weights.

        Args:
            read_content_weights (tf.Tensor): [batch_size, num_reads, memory_size]
            prev_read_weights (tf.Tensor): [batch_size, num_reads, memory_size]
            link (tf.Tensor): [batch_size, num_writes, memory_size, memory_size]
            read_mode (tf.Tensor): [batch_size, num_reads, 1 + 2*num_writes]
            training (bool): 是否在训练模式

        Returns:
            tf.Tensor: [batch_size, num_reads, memory_size]
        """
        return self.read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            prev_read_weights=prev_read_weights,
            link=link,
            read_mode=read_mode,
            training=training
        )

    def _read_words(self, read_weights: tf.Tensor, memory_updated: tf.Tensor) -> tf.Tensor:
        """
        Read words from memory using read weights.

        Args:
            read_weights (tf.Tensor): [batch_size, num_reads, memory_size]
            memory_updated (tf.Tensor): [batch_size, memory_size, word_size]

        Returns:
            tf.Tensor: [batch_size, num_reads, word_size]
        """
        return tf.matmul(read_weights, memory_updated)  # [batch_size, num_reads, word_size]

    def get_initial_state(self, batch_size: int, initial_time_steps: int = 0) -> BatchAccessState:
        """
        返回 MemoryAccess 模块的初始状态。

        Args:
            batch_size (int): 批次大小。
            initial_time_steps (int, optional): 初始时间步数。默认值为 0。

        Returns:
            BatchAccessState: 初始状态
        """
        memory = tf.zeros([batch_size, self.memory_size, self.word_size], dtype=tf.float32)
        read_weights = tf.zeros([batch_size, initial_time_steps, self.num_reads, self.memory_size], dtype=tf.float32)
        write_weights = tf.zeros([batch_size, initial_time_steps, self.num_writes, self.memory_size], dtype=tf.float32)

        # 初始化链接
        linkage = {
            'link': tf.zeros([batch_size, self.num_writes, self.memory_size, self.memory_size], dtype=tf.float32),
            'precedence_weights': tf.zeros([batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        }

        # 初始化使用率
        usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)

        read_words = tf.zeros([batch_size, self.num_reads, self.word_size], dtype=tf.float32)

        return BatchAccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage,
            usage=usage,
            read_words=read_words,
        )

    def state_size(self):
        return BatchAccessState(
            memory=tf.TensorShape([self.memory_size, self.word_size]),
            read_weights=tf.TensorShape([None, self.num_reads, self.memory_size]),
            write_weights=tf.TensorShape([None, self.num_writes, self.memory_size]),
            linkage=self.temporal_linkage_updater.state_size(),
            usage=tf.TensorShape([self.memory_size]),
            read_words=tf.TensorShape([self.num_reads, self.word_size])
        )