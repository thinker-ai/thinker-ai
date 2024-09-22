import collections
import tensorflow as tf
import os

from thinker_ai.agent.memory.humanoid_memory.dnc_new import addressing
from thinker_ai.agent.memory.humanoid_memory.dnc_new import util
from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import CosineWeights

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 定义 AccessState，用于保存访问模块的状态
AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))

class MemoryAccess(tf.keras.layers.Layer):
    def __init__(self, memory_size=128, word_size=20, num_reads=1, num_writes=1, epsilon=1e-6, name='memory_access'):
        super(MemoryAccess, self).__init__(name=name)
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes

        # Define sub-layers
        self.write_vectors = tf.keras.layers.Dense(self._num_writes * self._word_size)
        self.erase_vectors = tf.keras.layers.Dense(self._num_writes * self._word_size)
        self.write_gate = tf.keras.layers.Dense(self._num_writes)
        self.allocation_gate = tf.keras.layers.Dense(self._num_writes)
        self.free_gate = tf.keras.layers.Dense(self._num_reads)
        self.read_mode = tf.keras.layers.Dense(self._num_reads * (1 + 2 * self._num_writes))
        self.write_keys = tf.keras.layers.Dense(self._num_writes * self._word_size)
        self.write_strengths = tf.keras.layers.Dense(self._num_writes)
        self.read_keys = tf.keras.layers.Dense(self._num_reads * self._word_size)
        self.read_strengths = tf.keras.layers.Dense(self._num_reads)
        # CosineWeights、TemporalLinkage 和 Freeness 模块
        self.write_content_weights = CosineWeights(num_heads=self._num_writes,word_size=self._word_size)
        self.read_content_weights = CosineWeights(num_heads=self._num_reads,word_size=self._word_size)
        self.temporal_linkage = addressing.TemporalLinkage(
            memory_size=memory_size, num_writes=num_writes, name='temporal_linkage')
        self.freeness = addressing.Freeness(
            memory_size=memory_size, epsilon=epsilon, name='freeness')

        # 定义初始化器
        bias_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
        kernel_init = tf.keras.initializers.GlorotUniform()

        # 这些层将在 build 方法中初始化，因为我们需要知道 input_size
        self._bias_init = bias_init
        self._kernel_init = kernel_init

    def build(self, input_shape):
        # 假设 input_shape 是一个字典，包含 'inputs' 和 'prev_state'
        controller_output_shape = input_shape['inputs']  # [batch_size, sequence_length, input_size]
        self.input_size = controller_output_shape[-1]  # 获取输入大小

        # 定义用于生成各个参数的 Dense 层
        self.write_vectors = tf.keras.layers.Dense(
            units=self._num_writes * self._word_size,
            activation=None,
            name='write_vectors',
            kernel_initializer=self._kernel_init,
            bias_initializer=self._bias_init,
            use_bias=True
        )
        self.erase_vectors = tf.keras.layers.Dense(
            units=self._num_writes * self._word_size,
            activation='sigmoid',
            name='erase_vectors',
            kernel_initializer=self._kernel_init,
            bias_initializer=self._bias_init,
            use_bias=True
        )
        self.free_gate = tf.keras.layers.Dense(
            units=self._num_reads,
            activation='sigmoid',
            name='free_gate',
            kernel_initializer=self._kernel_init,
            bias_initializer=self._bias_init,
            use_bias=True
        )
        self.allocation_gate = tf.keras.layers.Dense(
            units=self._num_writes,
            activation='sigmoid',
            name='allocation_gate',
            kernel_initializer=self._kernel_init,
            bias_initializer=self._bias_init,
            use_bias=True
        )
        self.write_gate = tf.keras.layers.Dense(
            units=self._num_writes,
            activation='sigmoid',
            name='write_gate',
            kernel_initializer=self._kernel_init,
            bias_initializer=self._bias_init,
            use_bias=True
        )
        self.read_mode = tf.keras.layers.Dense(
            units=self._num_reads * (1 + 2 * self._num_writes),
            activation=None,
            name='read_mode',
            kernel_initializer=self._kernel_init,
            bias_initializer=self._bias_init,
            use_bias=True
        )

        self.write_strengths = tf.keras.layers.Dense(
            units=self._num_writes,
            activation='softplus',
            name='write_strengths',
            kernel_initializer=self._kernel_init,
            bias_initializer=self._bias_init,
            use_bias=True
        )
        self.read_strengths = tf.keras.layers.Dense(
            units=self._num_reads,
            activation='softplus',
            name='read_strengths',
            kernel_initializer=self._kernel_init,
            bias_initializer=self._bias_init,
            use_bias=True
        )

        self.write_keys = tf.keras.layers.Dense(
            units=self._num_writes * self._word_size,
            activation=None,
            name='write_keys',
            kernel_initializer=self._kernel_init,
            bias_initializer=self._bias_init,
            use_bias=True
        )
        self.read_keys = tf.keras.layers.Dense(
            units=self._num_reads * self._word_size,
            activation=None,
            name='read_keys',
            kernel_initializer=self._kernel_init,
            bias_initializer=self._bias_init,
            use_bias=True
        )

        super(MemoryAccess, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        """
        Args:
            inputs: 字典，包含 'inputs' 和 'prev_state'
            training: Boolean, whether in training mode
        Returns:
            read_words: Tensor of shape [batch_size, sequence_length, num_reads, word_size]
            final_state: AccessState namedtuple containing the new state
        """
        # 从 inputs 字典中获取 controller_output 和 prev_state
        controller_output = inputs['inputs']  # [batch_size, sequence_length, input_size]
        prev_state = inputs['prev_state']

        # 获取 batch_size 和 sequence_length
        batch_size = tf.shape(controller_output)[0]
        sequence_length = tf.shape(controller_output)[1]
        input_size = tf.shape(controller_output)[2]

        # 将输入 reshape 为二维 (batch_size * sequence_length, input_size)
        reshaped_controller_output = tf.reshape(controller_output, [-1, input_size])  # [batch_size * sequence_length, input_size]

        # Tile memory to match sequence_length
        memory_tiled = tf.tile(prev_state.memory, [sequence_length, 1, 1])  # [batch_size * sequence_length, memory_size, word_size]

        # 处理输入，生成需要的参数
        processed_inputs = self._read_inputs(reshaped_controller_output, memory_tiled)  # dict with parameters

        # 将生成的参数 reshape 回三维 (batch_size, sequence_length, ...)
        for key in processed_inputs:
            # 处理形状
            processed_inputs[key] = tf.reshape(processed_inputs[key],
                                               [batch_size, sequence_length] + list(processed_inputs[key].shape[1:]))  # [batch_size, sequence_length, ...]

        # 执行一步计算
        next_state = self._step(processed_inputs, prev_state)

        read_words = next_state['read_words']  # [batch_size, sequence_length, num_reads, word_size]

        # 返回最终状态
        final_state = AccessState(
            memory=next_state['memory'],               # [batch_size, memory_size, word_size]
            read_weights=next_state['read_weights'],   # [batch_size, sequence_length, num_reads, memory_size]
            write_weights=next_state['write_weights'], # [batch_size, sequence_length, num_writes, memory_size]
            linkage=next_state['linkage'],             # {'link': ..., 'precedence_weights': ...}
            usage=next_state['usage']                  # [batch_size, memory_size]
        )

        return read_words, final_state

    def _step(self, inputs, prev_state):
        """
        执行 MemoryAccess 的一个时间步操作，更新状态。

        Args:
            inputs (dict): 处理后的输入参数
            prev_state (AccessState): 前一时刻的状态

        Returns:
            dict: 包含更新后的状态和输出
        """
        # 获取输入参数
        write_content_weights = inputs['write_content_weights']  # [batch_size, sequence_length, num_writes, memory_size]
        read_content_weights = inputs['read_content_weights']    # [batch_size, sequence_length, num_reads, memory_size]
        write_vectors = inputs['write_vectors']                  # [batch_size, sequence_length, num_writes, word_size]
        erase_vectors = inputs['erase_vectors']                  # [batch_size, sequence_length, num_writes, word_size]
        free_gate = inputs['free_gate']                          # [batch_size, sequence_length, num_reads]
        allocation_gate = inputs['allocation_gate']              # [batch_size, sequence_length, num_writes]
        write_gate = inputs['write_gate']                        # [batch_size, sequence_length, num_writes]
        read_mode = inputs['read_mode']                          # [batch_size, sequence_length, num_reads, 1 + 2*num_writes]

        # 定义 read_content_weights_sum，类似于 write_content_weights_sum
        read_content_weights_sum = tf.reduce_sum(read_content_weights, axis=1)  # [batch_size, num_reads, memory_size]

        # 定义 write_content_weights_sum
        write_content_weights_sum = tf.reduce_sum(write_content_weights, axis=1)  # [batch_size, num_writes, memory_size]

        # Aggregate write_gate and allocation_gate over sequence_length
        write_gate_sum = tf.reduce_sum(write_gate, axis=1)  # [batch_size, num_writes]
        allocation_gate_sum = tf.reduce_sum(allocation_gate, axis=1)  # [batch_size, num_writes]

        # Sum write_vectors and erase_vectors over sequence_length
        write_vectors_sum = tf.reduce_sum(write_vectors, axis=1)    # [batch_size, num_writes, word_size]
        erase_vectors_sum = tf.reduce_sum(erase_vectors, axis=1)    # [batch_size, num_writes, word_size]

        # Compute write_allocation_weights via Freeness
        write_gates_sum = allocation_gate_sum * write_gate_sum  # [batch_size, num_writes]
        write_allocation_weights = self.freeness.write_allocation_weights(
            usage=prev_state.usage,          # [batch_size, memory_size]
            write_gates=write_gates_sum,     # [batch_size, num_writes]
            num_writes=self._num_writes
        )  # [batch_size, num_writes, memory_size]

        # Compute final_write_weights
        write_gate_sum_expanded = write_gate_sum[..., tf.newaxis]  # Shape: [batch_size, num_writes, 1]
        final_write_weights = write_gate_sum_expanded * (
                allocation_gate_sum[..., tf.newaxis] * write_allocation_weights +
                (1 - allocation_gate_sum[..., tf.newaxis]) * write_content_weights_sum
        )
        # 更新使用率 via Freeness layer
        usage = self.freeness.call({
            'write_weights': final_write_weights,    # [batch_size, num_writes, memory_size]
            'free_gate': tf.reduce_sum(free_gate, axis=1),  # [batch_size, num_reads]
            'read_weights': read_content_weights_sum,        # [batch_size, num_reads, memory_size]
            'prev_usage': prev_state.usage                   # [batch_size, memory_size]
        }, training=False)  # [batch_size, memory_size]

        # 打印使用率
        tf.print("Usage after update:", usage)

        # 更新内存 via erase and write
        memory = self._erase_and_write(
            prev_state.memory,            # [batch_size, memory_size, word_size]
            address=final_write_weights,  # [batch_size, num_writes, memory_size]
            reset_weights=erase_vectors_sum,  # [batch_size, num_writes, word_size]
            values=write_vectors_sum          # [batch_size, num_writes, word_size]
        )  # [batch_size, memory_size, word_size]

        # 打印内存状态
        tf.print("Memory after write:", memory)

        # Update linkage via TemporalLinkage
        linkage_state = self.temporal_linkage.call({
            'write_weights': final_write_weights,  # [batch_size, num_writes, memory_size]
            'prev_linkage': prev_state.linkage      # {'link': ..., 'precedence_weights': ...}
        }, training=False)  # {'link': ..., 'precedence_weights': ...}

        # 打印链路状态
        tf.print("Linkage state after update:", linkage_state)

        # Compute directional read weights
        forward_weights = self.temporal_linkage.directional_read_weights(
            link=linkage_state['link'],              # [batch_size, num_writes, memory_size, memory_size]
            prev_read_weights=prev_state.read_weights[:, -1, :, :],  # [batch_size, num_reads, memory_size]
            forward=True
        )  # [batch_size, num_reads, num_writes, memory_size]

        backward_weights = self.temporal_linkage.directional_read_weights(
            link=linkage_state['link'],              # [batch_size, num_writes, memory_size, memory_size]
            prev_read_weights=prev_state.read_weights[:, -1, :, :],  # [batch_size, num_reads, memory_size]
            forward=False
        )  # [batch_size, num_reads, num_writes, memory_size]

        # Process read_mode to separate content, forward, and backward modes
        # read_mode: [batch_size, sequence_length, num_reads, 1 + 2*num_writes]
        # Aggregate read_mode over sequence_length by summing
        read_mode_sum = tf.reduce_sum(read_mode, axis=1)  # [batch_size, num_reads, 1 + 2*num_writes]

        # Split read_mode_sum into content_mode, forward_mode, and backward_mode
        content_mode = read_mode_sum[:, :, 0]  # [batch_size, num_reads]
        forward_mode = read_mode_sum[:, :, 1:1 + self._num_writes]  # [batch_size, num_reads, num_writes]
        backward_mode = read_mode_sum[:, :, 1 + self._num_writes:]  # [batch_size, num_reads, num_writes]

        # Expand modes for broadcasting
        content_mode_expanded = tf.expand_dims(content_mode, axis=-1)  # [batch_size, num_reads, 1]
        forward_mode_expanded = tf.expand_dims(forward_mode, axis=-1)  # [batch_size, num_reads, num_writes, 1]
        backward_mode_expanded = tf.expand_dims(backward_mode, axis=-1)  # [batch_size, num_reads, num_writes, 1]

        # 计算读取权重
        read_weights = (
            content_mode_expanded * read_content_weights_sum +  # [batch_size, num_reads, memory_size]
            tf.reduce_sum(forward_mode_expanded * forward_weights, axis=2) +  # [batch_size, num_reads, memory_size]
            tf.reduce_sum(backward_mode_expanded * backward_weights, axis=2)   # [batch_size, num_reads, memory_size]
        )  # [batch_size, num_reads, memory_size]

        # 打印读取权重
        tf.print("Read weights:", read_weights)

        # Read words from memory
        read_words = tf.matmul(read_weights, memory)  # [batch_size, num_reads, word_size]

        # 打印读取的词向量
        tf.print("Read words:", read_words)

        # Create next_state dictionary
        next_state = {
            'memory': memory,               # [batch_size, memory_size, word_size]
            'read_weights': tf.expand_dims(read_weights, axis=1),   # [batch_size, 1, num_reads, memory_size]
            'write_weights': tf.expand_dims(final_write_weights, axis=1),  # [batch_size, 1, num_writes, memory_size]
            'linkage': linkage_state,       # {'link': ..., 'precedence_weights': ...}
            'usage': usage                   # [batch_size, memory_size]
        }

        return next_state

    def _read_inputs(self, reshaped_controller_output, memory_tiled):
        """
        处理 controller 的输出，生成必要的参数。

        Args:
            reshaped_controller_output (tf.Tensor): [batch_size * sequence_length, input_size]
            memory_tiled (tf.Tensor): [batch_size * sequence_length, memory_size, word_size]

        Returns:
            dict: 包含生成的各个参数
        """
        batch_size_time = tf.shape(reshaped_controller_output)[0]  # batch_size * sequence_length

        # 使用子层生成各个参数
        write_vectors = tf.reshape(self.write_vectors(reshaped_controller_output),
                                   [batch_size_time, self._num_writes, self._word_size])  # [batch_size * sequence_length, num_writes, word_size]
        erase_vectors = tf.reshape(self.erase_vectors(reshaped_controller_output),
                                   [batch_size_time, self._num_writes, self._word_size])  # [batch_size * sequence_length, num_writes, word_size]
        free_gate = self.free_gate(reshaped_controller_output)  # [batch_size * sequence_length, num_reads]
        allocation_gate = self.allocation_gate(reshaped_controller_output)  # [batch_size * sequence_length, num_writes]
        write_gate = self.write_gate(reshaped_controller_output)  # [batch_size * sequence_length, num_writes]
        read_mode = tf.nn.softmax(
            tf.reshape(self.read_mode(reshaped_controller_output),
                       [batch_size_time, self._num_reads, 1 + 2 * self._num_writes]),
            axis=-1
        )  # [batch_size * sequence_length, num_reads, 1 + 2*num_writes]
        write_keys = tf.reshape(self.write_keys(reshaped_controller_output),
                                [batch_size_time, self._num_writes, self._word_size])  # [batch_size * sequence_length, num_writes, word_size]
        write_strengths = self.write_strengths(reshaped_controller_output)  # [batch_size * sequence_length, num_writes]
        read_keys = tf.reshape(self.read_keys(reshaped_controller_output),
                               [batch_size_time, self._num_reads, self._word_size])  # [batch_size * sequence_length, num_reads, word_size]
        read_strengths = self.read_strengths(reshaped_controller_output)  # [batch_size * sequence_length, num_reads]

        # 计算 CosineWeights
        write_content_weights = self.write_content_weights.call({
            'memory': memory_tiled,      # [batch_size * sequence_length, memory_size, word_size]
            'keys': write_keys,          # [batch_size * sequence_length, num_writes, word_size]
            'strengths': write_strengths # [batch_size * sequence_length, num_writes]
        })  # [batch_size * sequence_length, num_writes, memory_size]

        read_content_weights = self.read_content_weights.call({
            'memory': memory_tiled,      # [batch_size * sequence_length, memory_size, word_size]
            'keys': read_keys,            # [batch_size * sequence_length, num_reads, word_size]
            'strengths': read_strengths   # [batch_size * sequence_length, num_reads]
        })  # [batch_size * sequence_length, num_reads, memory_size]

        return {
            'write_content_weights': write_content_weights,  # [batch_size * sequence_length, num_writes, memory_size]
            'read_content_weights': read_content_weights,    # [batch_size * sequence_length, num_reads, memory_size]
            'write_vectors': write_vectors,                  # [batch_size * sequence_length, num_writes, word_size]
            'erase_vectors': erase_vectors,                  # [batch_size * sequence_length, num_writes, word_size]
            'free_gate': free_gate,                          # [batch_size * sequence_length, num_reads]
            'allocation_gate': allocation_gate,              # [batch_size * sequence_length, num_writes]
            'write_gate': write_gate,                        # [batch_size * sequence_length, num_writes]
            'read_mode': read_mode                           # [batch_size * sequence_length, num_reads, 1 + 2*num_writes]
        }

    def _erase_and_write(self, memory, address, reset_weights, values):
        """
        执行擦除和写入操作，更新内存。

        Args:
            memory (tf.Tensor): [batch_size, memory_size, word_size]
            address (tf.Tensor): [batch_size, num_writes, memory_size]
            reset_weights (tf.Tensor): [batch_size, num_writes, word_size]
            values (tf.Tensor): [batch_size, num_writes, word_size]

        Returns:
            tf.Tensor: 更新后的内存 [batch_size, memory_size, word_size]
        """
        with tf.name_scope('erase_memory'):
            # reset_weights: [batch_size, num_writes, word_size]
            # address: [batch_size, num_writes, memory_size]
            # Expand dimensions to [batch_size, num_writes, memory_size, word_size]
            reset_weights_expanded = tf.expand_dims(reset_weights, axis=2)  # [batch_size, num_writes, 1, word_size]
            reset_weights_broadcast = tf.tile(reset_weights_expanded, [1, 1, self._memory_size, 1])  # [batch_size, num_writes, memory_size, word_size]

            # address: [batch_size, num_writes, memory_size]
            address_expanded = tf.expand_dims(address, axis=-1)  # [batch_size, num_writes, memory_size, 1]
            address_broadcast = tf.tile(address_expanded, [1, 1, 1, self._word_size])  # [batch_size, num_writes, memory_size, word_size]

            # Compute erase_gate: 1 - address * reset_weights
            erase_gate = 1 - address_broadcast * reset_weights_broadcast  # [batch_size, num_writes, memory_size, word_size]

            # Compute total erase_gate: product over writes
            total_erase_gate = tf.reduce_prod(erase_gate, axis=1)  # [batch_size, memory_size, word_size]

            # Update memory by applying erase
            memory_erased = memory * total_erase_gate  # [batch_size, memory_size, word_size]

        with tf.name_scope('additive_write'):
            # Compute additive write
            # address: [batch_size, num_writes, memory_size]
            # values: [batch_size, num_writes, word_size]
            # Use einsum to compute sum over writes: [batch_size, memory_size, word_size]
            add_matrix = tf.einsum('bnm,bnw->bmw', address, values)  # [batch_size, memory_size, word_size]

            # Update memory
            memory_updated = memory_erased + add_matrix  # [batch_size, memory_size, word_size]

        return memory_updated

    def get_initial_state(self, batch_size):
        """
        返回 MemoryAccess 模块的初始状态

        Args:
            batch_size (int): 批次大小。

        Returns:
            AccessState: 包含初始化的 memory、read_weights、write_weights、linkage、usage。
        """
        memory = tf.zeros([batch_size, self._memory_size, self._word_size], dtype=tf.float32)
        read_weights = tf.zeros([batch_size, self._num_reads, self._memory_size], dtype=tf.float32)
        write_weights = tf.zeros([batch_size, self._num_writes, self._memory_size], dtype=tf.float32)

        linkage = self.temporal_linkage.get_initial_state(batch_size)
        usage = self.freeness.get_initial_state([batch_size])  # Freeness expects batch_shape as a list/tuple

        return AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage,
            usage=usage
        )
    def _write_weights(self, inputs, memory, usage):
        with tf.name_scope('write_weights'):
            # 使用修改后的属性名
            write_content_weights = self.write_content_weights({
                'memory': memory,
                'keys': inputs['write_content_keys'],
                'strengths': inputs['write_content_strengths']
            })

            write_allocation_weights = self.freeness.write_allocation_weights(
                usage=usage,
                write_gates=inputs['allocation_gate'] * inputs['write_gate'],
                num_writes=self._num_writes)

            tf.print("Write Allocation Weights:", write_allocation_weights)

            allocation_gate = tf.expand_dims(inputs['allocation_gate'], -1)
            write_gate = tf.expand_dims(inputs['write_gate'], -1)

            final_write_weights = write_gate * (
                    allocation_gate * write_allocation_weights +
                    (1 - allocation_gate) * write_content_weights)

            return final_write_weights
    def _read_weights(self, inputs, memory, prev_read_weights, link):
        """
        计算读权重。

        Args:
            inputs (dict): 处理后的输入参数
            memory (tf.Tensor): [batch_size, memory_size, word_size]
            prev_read_weights (tf.Tensor): [batch_size, num_reads, memory_size]
            link (tf.Tensor): [batch_size, num_writes, memory_size, memory_size]

        Returns:
            tf.Tensor: 读权重 [batch_size, num_reads, memory_size]
        """
        # 获取 content_weights
        content_weights = inputs['read_content_weights']  # [batch_size, sequence_length, num_reads, memory_size]
        content_weights_sum = tf.reduce_sum(content_weights, axis=1)  # [batch_size, num_reads, memory_size]

        # 计算前向和后向权重
        forward_weights = self.temporal_linkage.directional_read_weights(
            link=link,              # [batch_size, num_writes, memory_size, memory_size]
            prev_read_weights=prev_read_weights,  # [batch_size, num_reads, memory_size]
            forward=True
        )  # [batch_size, num_reads, num_writes, memory_size]

        backward_weights = self.temporal_linkage.directional_read_weights(
            link=link,              # [batch_size, num_writes, memory_size, memory_size]
            prev_read_weights=prev_read_weights,  # [batch_size, num_reads, memory_size]
            forward=False
        )  # [batch_size, num_reads, num_writes, memory_size]

        # 获取 read_mode 的各个部分
        read_mode = inputs['read_mode']  # [batch_size, num_reads, 1 + 2*num_writes]
        content_mode = read_mode[:, :, 0]  # [batch_size, num_reads]
        forward_mode = read_mode[:, :, 1:1 + self._num_writes]  # [batch_size, num_reads, num_writes]
        backward_mode = read_mode[:, :, 1 + self._num_writes:]  # [batch_size, num_reads, num_writes]

        # Expand modes for broadcasting
        content_mode_expanded = tf.expand_dims(content_mode, axis=-1)  # [batch_size, num_reads, 1]
        forward_mode_expanded = tf.expand_dims(forward_mode, axis=-1)  # [batch_size, num_reads, num_writes, 1]
        backward_mode_expanded = tf.expand_dims(backward_mode, axis=-1)  # [batch_size, num_reads, num_writes, 1]

        # 计算最终的读权重
        read_weights = (
            content_mode_expanded * content_weights_sum +  # [batch_size, num_reads, memory_size]
            tf.reduce_sum(forward_mode_expanded * forward_weights, axis=2) +  # [batch_size, num_reads, memory_size]
            tf.reduce_sum(backward_mode_expanded * backward_weights, axis=2)   # [batch_size, num_reads, memory_size]
        )  # [batch_size, num_reads, memory_size]

        return read_weights

    @property
    def state_size(self):
        return AccessState(
            memory=tf.TensorShape([self._memory_size, self._word_size]),
            read_weights=tf.TensorShape([self._num_reads, self._memory_size]),
            write_weights=tf.TensorShape([self._num_writes, self._memory_size]),
            linkage=self.temporal_linkage.state_size,
            usage=self.freeness.state_size
        )