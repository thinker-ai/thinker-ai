# memory_access_rnn_cell.py

from typing import Tuple, Dict, Any, List

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AbstractRNNCell
from thinker_ai.agent.memory.humanoid_memory.dnc.component_factory import ComponentFactory


class MemoryAccessRNNCell(AbstractRNNCell):
    def __init__(
        self,
        memory_size: int,
        word_size: int,
        num_reads: int,
        num_writes: int,
        controller_size: int,
        config: Dict[str, Any],
        epsilon: float = 1e-6,
        **kwargs
    ):
        super(MemoryAccessRNNCell, self).__init__(**kwargs)
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.controller_size = controller_size
        self.config = config
        self.epsilon = epsilon

        # Initialize MemoryAccess related components
        factory = ComponentFactory(config or {})
        components = factory.create_all_components()

        self.content_weight_calculator = components['content_weight_calculator']
        self.write_weight_calculator = components['write_weight_calculator']
        self.temporal_linkage_updater = components['temporal_linkage_updater']
        self.read_weight_calculator = components['read_weight_calculator']
        self.usage_updater = components['usage_updater']
        self.memory_updater = components['memory_updater']

        # Define memory interaction layers
        self.write_vectors_layer = Dense(
            units=self.num_writes * self.word_size,
            activation=None,
            name='write_vectors',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer='zeros',
            use_bias=True
        )
        self.erase_vectors_layer = Dense(
            units=self.num_writes * self.word_size,
            activation='sigmoid',
            name='erase_vectors',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.Constant(-1.0),
            use_bias=True
        )
        self.write_gate_layer = Dense(
            units=self.num_writes,
            activation='sigmoid',
            name='write_gate',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.Constant(-1.0),
            use_bias=True
        )
        self.allocation_gate_layer = Dense(
            units=self.num_writes,
            activation='sigmoid',
            name='allocation_gate',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.Constant(-1.0),
            use_bias=True
        )
        self.free_gate_layer = Dense(
            units=self.num_reads,
            activation='sigmoid',
            name='free_gate',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.Constant(-1.0),
            use_bias=True
        )
        self.read_mode_layer = Dense(
            units=self.num_reads * (1 + 2 * self.num_writes),
            activation=None,
            name='read_mode',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer='zeros',
            use_bias=True
        )
        self.write_keys_layer = Dense(
            units=self.num_writes * self.word_size,
            activation=None,
            name='write_keys',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer='zeros',
            use_bias=True
        )
        self.write_strengths_layer = Dense(
            units=self.num_writes,
            activation='softplus',
            name='write_strengths',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer='zeros',
            use_bias=True
        )
        self.read_keys_layer = Dense(
            units=self.num_reads * self.word_size,
            activation=None,
            name='read_keys',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer='zeros',
            use_bias=True
        )
        self.read_strengths_layer = Dense(
            units=self.num_reads,
            activation='softplus',
            name='read_strengths',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer='zeros',
            use_bias=True
        )

    @property
    def state_size(self):
        return [
            tf.TensorShape([self.memory_size, self.word_size]),  # memory
            tf.TensorShape([self.num_reads, self.memory_size]),  # read_weights
            tf.TensorShape([self.num_writes, self.memory_size]),  # write_weights
            tf.TensorShape([self.num_writes, self.memory_size, self.memory_size]),  # linkage_link
            tf.TensorShape([self.num_writes, self.memory_size]),  # linkage_precedence_weights
            tf.TensorShape([self.memory_size]),  # usage
            tf.TensorShape([self.controller_size]),  # controller_hidden
            tf.TensorShape([self.controller_size])   # controller_cell
        ]

    @property
    def output_size(self):
        return tf.TensorShape([self.num_reads, self.word_size])

    def call(self, inputs: tf.Tensor, states: List[tf.Tensor], training: bool = False) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Process a single time step's input and update the state.

        Args:
            inputs (tf.Tensor): [batch_size, input_size]
            states (List[tf.Tensor]): Previous state list
            training (bool): Whether in training mode

        Returns:
            Tuple[tf.Tensor, List[tf.Tensor]]: Output read_words and updated state list
        """
        # Unpack the state
        memory, read_weights, write_weights, link, precedence_weights, usage, controller_hidden, controller_cell = states

        # Here, you should integrate the controller's operations.
        # For simplicity, let's assume the controller_hidden and controller_cell are updated via some operations.
        # Replace the following lines with actual controller logic.
        controller_hidden_new = tf.keras.activations.relu(
            Dense(self.controller_size)(inputs)
        )
        controller_cell_new = tf.keras.activations.relu(
            Dense(self.controller_size)(inputs)
        )

        # Generate write-related parameters
        write_vectors, erase_vectors, write_gate, allocation_gate, free_gate = self._generate_write_parameters(controller_hidden_new)
        read_mode = self._generate_read_mode(controller_hidden_new)
        write_keys, write_strengths = self._generate_write_keys_strengths(controller_hidden_new)
        read_keys, read_strengths = self._generate_read_keys_strengths(controller_hidden_new)

        # Compute content weights
        write_content_weights = self._compute_content_weights(write_keys, write_strengths, memory)
        read_content_weights = self._compute_content_weights(read_keys, read_strengths, memory)

        # Compute write weights
        final_write_weights = self._compute_write_weights(write_content_weights, allocation_gate, write_gate, usage, training)

        # Update memory
        memory_updated = self._update_memory(memory, final_write_weights, erase_vectors, write_vectors)

        # Update linkage
        linkage_updated = self._update_linkage(final_write_weights, {'link': link, 'precedence_weights': precedence_weights}, training)

        # Update usage
        usage_updated = self._update_usage(final_write_weights, free_gate, read_weights, usage, training)

        # Compute read weights
        read_weights_updated = self._compute_read_weights(read_content_weights, read_weights, linkage_updated['link'], read_mode, training)

        # Read words
        read_words = self._read_words(read_weights_updated, memory_updated)

        # Update controller hidden and cell
        # Replace this with actual controller update logic
        controller_hidden_final = controller_hidden_new
        controller_cell_final = controller_cell_new

        # Create the new state list
        new_state = [
            memory_updated,
            read_weights_updated,
            final_write_weights,
            linkage_updated['link'],
            linkage_updated['precedence_weights'],
            usage_updated,
            controller_hidden_final,
            controller_cell_final
        ]

        return read_words, new_state

    # Implement helper methods (dummy implementations; replace with actual logic)
    def _generate_write_parameters(self, controller_output_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        write_vectors = self.write_vectors_layer(controller_output_t)  # [batch_size, num_writes * word_size]
        write_vectors = tf.reshape(write_vectors, [tf.shape(write_vectors)[0], self.num_writes, self.word_size])  # [batch_size, num_writes, word_size]

        erase_vectors = self.erase_vectors_layer(controller_output_t)  # [batch_size, num_writes * word_size]
        erase_vectors = tf.reshape(erase_vectors, [tf.shape(erase_vectors)[0], self.num_writes, self.word_size])  # [batch_size, num_writes, word_size]

        write_gate = self.write_gate_layer(controller_output_t)  # [batch_size, num_writes]
        allocation_gate = self.allocation_gate_layer(controller_output_t)  # [batch_size, num_writes]
        free_gate = self.free_gate_layer(controller_output_t)  # [batch_size, num_reads]

        return write_vectors, erase_vectors, write_gate, allocation_gate, free_gate

    def _generate_read_mode(self, controller_output_t: tf.Tensor) -> tf.Tensor:
        read_mode = self.read_mode_layer(controller_output_t)  # [batch_size, num_reads * (1 + 2*num_writes)]
        read_mode = tf.reshape(read_mode, [tf.shape(read_mode)[0], self.num_reads, 1 + 2 * self.num_writes])  # [batch_size, num_reads, 1 + 2*num_writes]
        return read_mode

    def _generate_write_keys_strengths(self, controller_output_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        write_keys = self.write_keys_layer(controller_output_t)  # [batch_size, num_writes * word_size]
        write_keys = tf.reshape(write_keys, [tf.shape(write_keys)[0], self.num_writes, self.word_size])  # [batch_size, num_writes, word_size]

        write_strengths = self.write_strengths_layer(controller_output_t)  # [batch_size, num_writes]
        return write_keys, write_strengths

    def _generate_read_keys_strengths(self, controller_output_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        read_keys = self.read_keys_layer(controller_output_t)  # [batch_size, num_reads * word_size]
        read_keys = tf.reshape(read_keys, [tf.shape(read_keys)[0], self.num_reads, self.word_size])  # [batch_size, num_reads, word_size]

        read_strengths = self.read_strengths_layer(controller_output_t)  # [batch_size, num_reads]
        return read_keys, read_strengths

    def _compute_content_weights(self, keys: tf.Tensor, strengths: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        return self.content_weight_calculator.compute(
            keys=keys,
            strengths=strengths,
            memory=memory
        )

    def _compute_write_weights(self, write_content_weights: tf.Tensor, allocation_gate: tf.Tensor,
                               write_gate: tf.Tensor, usage: tf.Tensor, training: bool) -> tf.Tensor:
        return self.write_weight_calculator.compute(
            write_content_weights=write_content_weights,
            allocation_gate=allocation_gate,
            write_gate=write_gate,
            prev_usage=usage,
            training=training
        )

    def _update_memory(self, memory: tf.Tensor, write_weights: tf.Tensor, erase_vectors: tf.Tensor,
                       write_vectors: tf.Tensor) -> tf.Tensor:
        return self.memory_updater.update_memory(
            memory=memory,
            write_weights=write_weights,
            erase_vectors=erase_vectors,
            write_vectors=write_vectors
        )

    def _update_linkage(self, write_weights: tf.Tensor, prev_linkage: Dict[str, tf.Tensor], training: bool) -> Dict[str, tf.Tensor]:
        return self.temporal_linkage_updater.update_linkage(
            write_weights=write_weights,
            prev_linkage=prev_linkage,
            training=training
        )

    def _update_usage(self, write_weights: tf.Tensor, free_gate: tf.Tensor, read_weights_prev: tf.Tensor,
                      usage: tf.Tensor, training: bool) -> tf.Tensor:
        return self.usage_updater.update_usage(
            write_weights=write_weights,
            free_gate=free_gate,
            read_weights_prev=read_weights_prev,
            prev_usage=usage,
            training=training
        )

    def _compute_read_weights(self, read_content_weights: tf.Tensor, read_weights_prev: tf.Tensor, link: tf.Tensor,
                              read_mode: tf.Tensor, training: bool) -> tf.Tensor:
        return self.read_weight_calculator.compute(
            read_content_weights=read_content_weights,
            read_weights_prev=read_weights_prev,
            link=link,
            read_mode=read_mode,
            training=training
        )

    def _read_words(self, read_weights: tf.Tensor, memory_updated: tf.Tensor) -> tf.Tensor:
        return tf.matmul(read_weights, memory_updated)  # [batch_size, num_reads, word_size]
