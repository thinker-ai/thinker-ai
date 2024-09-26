import collections
import tensorflow as tf
import os

from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import (
    CosineWeights, TemporalLinkage, WriteAllocation, UsageUpdate
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 定义 AccessState，用于保存访问模块的状态
AccessState = collections.namedtuple('AccessState', [
    'memory',  # [batch_size, memory_size, word_size]
    'read_weights',  # [batch_size, time_steps, num_reads, memory_size]
    'write_weights',  # [batch_size, time_steps, num_writes, memory_size]
    'linkage',
    # {'link': [batch_size, num_writes, memory_size, memory_size], 'precedence_weights': [batch_size, num_writes, memory_size]}
    'usage',  # [batch_size, memory_size]
    'read_words'  # [batch_size, num_reads, word_size] 或 None
])


# 以下是 MemoryAccess 类的定义，请确保其存在于您的代码库中
class MemoryAccess(tf.keras.layers.Layer):
    def __init__(self, memory_size, word_size, num_reads, num_writes,
                 epsilon=1e-6, write_content_weights_fn=None, name='memory_access'):
        super(MemoryAccess, self).__init__(name=name)
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.epsilon = epsilon

        # 定义子层
        self.write_vectors = tf.keras.layers.Dense(
            units=self.num_writes * self.word_size,
            activation=None,
            name='write_vectors',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.erase_vectors = tf.keras.layers.Dense(
            units=self.num_writes * self.word_size,
            activation='sigmoid',
            name='erase_vectors',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.write_gate = tf.keras.layers.Dense(
            units=self.num_writes,
            activation='sigmoid',
            name='write_gate',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.allocation_gate = tf.keras.layers.Dense(
            units=self.num_writes,
            activation='sigmoid',
            name='allocation_gate',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.free_gate = tf.keras.layers.Dense(
            units=self.num_reads,
            activation='sigmoid',
            name='free_gate',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.read_mode = tf.keras.layers.Dense(
            units=self.num_reads * (1 + 2 * self.num_writes),
            activation=None,
            name='read_mode',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.write_keys = tf.keras.layers.Dense(
            units=self.num_writes * self.word_size,
            activation=None,
            name='write_keys',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.write_strengths = tf.keras.layers.Dense(
            units=self.num_writes,
            activation='softplus',
            name='write_strengths',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.read_keys = tf.keras.layers.Dense(
            units=self.num_reads * self.word_size,
            activation=None,
            name='read_keys',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )
        self.read_strengths = tf.keras.layers.Dense(
            units=self.num_reads,
            activation='softplus',
            name='read_strengths',
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            use_bias=True
        )

        # 初始化其他子层
        self.write_content_weights = CosineWeights(num_heads=self.num_writes, word_size=self.word_size)
        self.read_content_weights = CosineWeights(num_heads=self.num_reads, word_size=self.word_size)
        # 初始化 TemporalLinkage
        self.temporal_linkage = TemporalLinkage(memory_size, num_writes, name='temporal_linkage')
        # 初始化 WriteAllocation，传入 write_content_weights_fn
        self.write_allocation = WriteAllocation(
            num_writes=num_writes,
            memory_size=memory_size,
            write_content_weights_fn=write_content_weights_fn,  # 传入函数
            name='write_allocation'
        )
        self.usage_update = UsageUpdate(memory_size=self.memory_size, num_writes=self.num_writes,
                                        num_reads=self.num_reads, epsilon=self.epsilon, name='usage_update')

    def build(self, input_shape):
        # 让Keras自动处理子层的构建
        super(MemoryAccess, self).build(input_shape)

    def call(self, inputs, training=False):
        """
        处理输入并执行 MemoryAccess 操作。

        Args:
            inputs (dict): 包含 'inputs' 和 'prev_state'
                - 'inputs': [batch_size, sequence_length, input_size]
                - 'prev_state': AccessState namedtuple
            training (bool): 是否在训练模式

        Returns:
            dict: 包含 'read_words' 和 'final_state'
        """
        controller_output = inputs['inputs']  # [batch_size, sequence_length, input_size]
        prev_state = inputs['prev_state']  # AccessState

        batch_size = tf.shape(controller_output)[0]
        sequence_length = tf.shape(controller_output)[1]
        input_size = tf.shape(controller_output)[2]

        # Reshape controller_output to [batch_size * sequence_length, input_size]
        reshaped_controller_output = tf.reshape(controller_output,
                                                [-1, input_size])  # [batch_size * sequence_length, input_size]

        # Tile memory to match sequence_length
        memory_tiled = tf.tile(prev_state.memory,
                               [1, sequence_length, 1])  # [batch_size, sequence_length, memory_size, word_size]
        memory_tiled = tf.reshape(memory_tiled, [batch_size * sequence_length, self.memory_size,
                                                 self.word_size])  # [batch_size * sequence_length, memory_size, word_size]

        # Process inputs to generate required parameters
        processed_inputs = self._read_inputs(reshaped_controller_output, memory_tiled)  # dict with parameters

        # Reshape generated parameters back to [batch_size, sequence_length, ...]
        for key in processed_inputs:
            processed_inputs[key] = tf.reshape(
                processed_inputs[key],
                [batch_size, sequence_length] + processed_inputs[key].shape.as_list()[1:]
            )  # [batch_size, sequence_length, ...]

        # Perform one step computation for each time step
        next_state = prev_state
        read_words_all = []
        for t in range(sequence_length):
            # Extract parameters for the current time step
            current_inputs = {key: processed_inputs[key][:, t] for key in processed_inputs}

            # Perform one step
            next_state = self._step(current_inputs, next_state, training)

            # Collect read_words
            read_words_all.append(next_state.read_words)  # [batch_size, num_reads, word_size]

        # Concatenate read_words over time steps
        read_words = tf.stack(read_words_all, axis=1)  # [batch_size, sequence_length, num_reads, word_size]

        return {'read_words': read_words, 'final_state': next_state}

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
                                   [batch_size_time, self.num_writes,
                                    self.word_size])  # [batch_size * sequence_length, num_writes, word_size]
        erase_vectors = tf.reshape(self.erase_vectors(reshaped_controller_output),
                                   [batch_size_time, self.num_writes,
                                    self.word_size])  # [batch_size * sequence_length, num_writes, word_size]
        write_gate = tf.reshape(self.write_gate(reshaped_controller_output),
                                [batch_size_time, self.num_writes])  # [batch_size * sequence_length, num_writes]
        allocation_gate = tf.reshape(self.allocation_gate(reshaped_controller_output),
                                     [batch_size_time, self.num_writes])  # [batch_size * sequence_length, num_writes]
        free_gate = tf.reshape(self.free_gate(reshaped_controller_output),
                               [batch_size_time, self.num_reads])  # [batch_size * sequence_length, num_reads]
        read_mode = tf.nn.softmax(
            tf.reshape(self.read_mode(reshaped_controller_output),
                       [batch_size_time, self.num_reads, 1 + 2 * self.num_writes]),
            axis=-1
        )  # [batch_size * sequence_length, num_reads, 1 + 2*num_writes]
        write_keys = tf.reshape(self.write_keys(reshaped_controller_output),
                                [batch_size_time, self.num_writes,
                                 self.word_size])  # [batch_size * sequence_length, num_writes, word_size]
        write_strengths = self.write_strengths(reshaped_controller_output)  # [batch_size * sequence_length, num_writes]
        read_keys = tf.reshape(self.read_keys(reshaped_controller_output),
                               [batch_size_time, self.num_reads,
                                self.word_size])  # [batch_size * sequence_length, num_reads, word_size]
        read_strengths = self.read_strengths(reshaped_controller_output)  # [batch_size * sequence_length, num_reads]

        # 计算 CosineWeights
        write_content_weights = self.write_content_weights({
            'memory': memory_tiled,  # [batch_size * sequence_length, memory_size, word_size]
            'keys': write_keys,  # [batch_size * sequence_length, num_writes, word_size]
            'strengths': write_strengths  # [batch_size * sequence_length, num_writes]
        })  # [batch_size * sequence_length, num_writes, memory_size]

        read_content_weights = self.read_content_weights({
            'memory': memory_tiled,  # [batch_size * sequence_length, memory_size, word_size]
            'keys': read_keys,  # [batch_size * sequence_length, num_reads, word_size]
            'strengths': read_strengths  # [batch_size * sequence_length, num_reads]
        })  # [batch_size * sequence_length, num_reads, memory_size]

        return {
            'write_content_weights': write_content_weights,  # [batch_size * sequence_length, num_writes, memory_size]
            'read_content_weights': read_content_weights,  # [batch_size * sequence_length, num_reads, memory_size]
            'write_vectors': write_vectors,  # [batch_size * sequence_length, num_writes, word_size]
            'erase_vectors': erase_vectors,  # [batch_size * sequence_length, num_writes, word_size]
            'free_gate': free_gate,  # [batch_size * sequence_length, num_reads]
            'allocation_gate': allocation_gate,  # [batch_size * sequence_length, num_writes]
            'write_gate': write_gate,  # [batch_size * sequence_length, num_writes]
            'read_mode': read_mode  # [batch_size * sequence_length, num_reads, 1 + 2*num_writes]
        }

    def _compute_write_weights(self, write_content_weights, allocation_gate, write_gate, prev_usage, training):
        """
        计算写入权重（write_weights）
        """
        # 计算 write_gates_sum
        write_gates_sum = allocation_gate * write_gate  # [batch_size, num_writes]

        # 计算 write_allocation_weights via WriteAllocation layer
        write_allocation_weights = self.write_allocation({
            'usage': prev_usage,  # [batch_size, memory_size]
            'write_gates_sum': write_gates_sum  # [batch_size, num_writes]
        }, training=training)  # [batch_size, num_writes, memory_size]

        # 计算 final_write_weights
        write_gates_sum_expanded = tf.expand_dims(write_gates_sum, axis=-1)  # [batch_size, num_writes, 1]
        allocation_gate_expanded = tf.expand_dims(allocation_gate, axis=-1)  # [batch_size, num_writes, 1]
        final_write_weights = write_gates_sum_expanded * (
                allocation_gate_expanded * write_allocation_weights +
                (1 - allocation_gate_expanded) * write_content_weights
        )  # [batch_size, num_writes, memory_size]

        return final_write_weights

    def _step(self, inputs, prev_state, training):
        """
        执行 MemoryAccess 的一个时间步操作，更新状态。
        """
        # 获取输入参数
        write_content_weights = inputs['write_content_weights']  # [batch_size, num_writes, memory_size]
        read_content_weights = inputs['read_content_weights']  # [batch_size, num_reads, memory_size]
        write_vectors = inputs['write_vectors']  # [batch_size, num_writes, word_size]
        erase_vectors = inputs['erase_vectors']  # [batch_size, num_writes, word_size]
        free_gate = inputs['free_gate']  # [batch_size, num_reads]
        allocation_gate = inputs['allocation_gate']  # [batch_size, num_writes]
        write_gate = inputs['write_gate']  # [batch_size, num_writes]
        read_mode = inputs['read_mode']  # [batch_size, num_reads, 1 + 2*num_writes]

        # 使用提取出来的辅助函数来计算写入权重
        final_write_weights = self._compute_write_weights(
            write_content_weights, allocation_gate, write_gate, prev_state.usage, training
        )  # [batch_size, num_writes, memory_size]

        # 更新使用率 via UsageUpdate layer
        usage = self.usage_update({
            'write_weights': final_write_weights,  # [batch_size, num_writes, memory_size]
            'free_gate': free_gate,  # [batch_size, num_reads]
            'read_weights': read_content_weights,  # [batch_size, num_reads, memory_size]
            'prev_usage': prev_state.usage  # [batch_size, memory_size]
        }, training=training)  # [batch_size, memory_size]

        # 更新内存 via erase and write
        memory = self._erase_and_write(
            prev_state.memory,  # [batch_size, memory_size, word_size]
            address=final_write_weights,  # [batch_size, num_writes, memory_size]
            reset_weights=erase_vectors,  # [batch_size, num_writes, word_size]
            values=write_vectors  # [batch_size, num_writes, word_size]
        )  # [batch_size, memory_size, word_size]

        # 更新链接 via TemporalLinkage layer
        linkage_state = self.temporal_linkage({
            'write_weights': final_write_weights,  # [batch_size, num_writes, memory_size]
            'prev_linkage': prev_state.linkage  # {'link': ..., 'precedence_weights': ...}
        }, training=training)

        # 计算前向和后向读取权重
        link = linkage_state['link']  # [batch_size, num_writes, memory_size, memory_size]
        prev_read_weights = prev_state.read_weights[:, -1, :, :]  # [batch_size, num_reads, memory_size]
        forward_weights = self.temporal_linkage.directional_read_weights(
            link=link,
            prev_read_weights=prev_read_weights,
            forward=True
        )  # [batch_size, num_reads, num_writes, memory_size]

        backward_weights = self.temporal_linkage.directional_read_weights(
            link=link,
            prev_read_weights=prev_read_weights,
            forward=False
        )  # [batch_size, num_reads, num_writes, memory_size]

        # 处理 read_mode，将其分为 content, forward 和 backward 模式
        content_mode = read_mode[:, :, 0]  # [batch_size, num_reads]
        forward_mode = read_mode[:, :, 1:1 + self.num_writes]  # [batch_size, num_reads, num_writes]
        backward_mode = read_mode[:, :, 1 + self.num_writes:]  # [batch_size, num_reads, num_writes]

        # Expand modes for broadcasting
        content_mode_expanded = tf.expand_dims(content_mode, axis=-1)  # [batch_size, num_reads, 1]
        forward_mode_expanded = tf.expand_dims(forward_mode, axis=-1)  # [batch_size, num_reads, num_writes, 1]
        backward_mode_expanded = tf.expand_dims(backward_mode, axis=-1)  # [batch_size, num_reads, num_writes, 1]

        # 计算读取权重
        read_weights = (
                content_mode_expanded * read_content_weights +  # [batch_size, num_reads, memory_size]
                tf.reduce_sum(forward_mode_expanded * forward_weights, axis=2) +  # [batch_size, num_reads, memory_size]
                tf.reduce_sum(backward_mode_expanded * backward_weights, axis=2)  # [batch_size, num_reads, memory_size]
        )  # [batch_size, num_reads, memory_size]

        # 读取词向量
        read_words = tf.matmul(read_weights, memory)  # [batch_size, num_reads, word_size]

        # 创建下一个状态
        next_state = AccessState(
            memory=memory,  # [batch_size, memory_size, word_size]
            read_weights=tf.concat(
                [prev_state.read_weights, tf.expand_dims(read_weights, axis=1)],
                axis=1
            ),  # [batch_size, time_steps+1, num_reads, memory_size]
            write_weights=tf.concat(
                [prev_state.write_weights, tf.expand_dims(final_write_weights, axis=1)],
                axis=1
            ),  # [batch_size, time_steps+1, num_writes, memory_size]
            linkage=linkage_state,  # {'link': ..., 'precedence_weights': ...}
            usage=usage,  # [batch_size, memory_size]
            read_words=read_words  # [batch_size, num_reads, word_size]
        )

        return next_state

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

            # Compute erase_gate: 1 - address * reset_weights
            # Expand dimensions to [batch_size, num_writes, memory_size, word_size]
            reset_weights_expanded = tf.expand_dims(reset_weights, axis=2)  # [batch_size, num_writes, 1, word_size]
            address_expanded = tf.expand_dims(address, axis=-1)  # [batch_size, num_writes, memory_size, 1]
            erase_gate = 1 - tf.multiply(address_expanded,
                                         reset_weights_expanded)  # [batch_size, num_writes, memory_size, word_size]

            # Compute total erase_gate: product over writes
            total_erase_gate = tf.reduce_prod(erase_gate, axis=1)  # [batch_size, memory_size, word_size]

            # Update memory by applying erase
            memory_erased = tf.multiply(memory, total_erase_gate)  # [batch_size, memory_size, word_size]

        with tf.name_scope('additive_write'):
            # Compute additive write
            # address: [batch_size, num_writes, memory_size]
            # values: [batch_size, num_writes, word_size]

            # Expand dimensions for broadcasting
            address_expanded = tf.expand_dims(address, axis=-1)  # [batch_size, num_writes, memory_size, 1]
            values_expanded = tf.expand_dims(values, axis=2)  # [batch_size, num_writes, 1, word_size]

            # Compute additive writes: [batch_size, memory_size, word_size]
            add_matrix = tf.reduce_sum(tf.multiply(address_expanded, values_expanded),
                                       axis=1)  # [batch_size, memory_size, word_size]

            # Update memory
            memory_updated = tf.add(memory_erased, add_matrix)  # [batch_size, memory_size, word_size]

        return memory_updated

    def get_initial_state(self, batch_shape, initial_time_steps=0):
        """
        返回 MemoryAccess 模块的初始状态。

        Args:
            batch_shape (tf.Tensor): 批次形状，例如 tf.constant(BATCH_SIZE, dtype=tf.int32)
            initial_time_steps (int, optional): 初始时间步数。默认值为 0。

        Returns:
            AccessState: 包含初始化的 memory、read_weights、write_weights、linkage、usage、read_words。
        """
        memory = tf.zeros([batch_shape, self.memory_size, self.word_size], dtype=tf.float32)
        read_weights = tf.zeros([batch_shape, initial_time_steps, self.num_reads, self.memory_size], dtype=tf.float32)
        write_weights = tf.zeros([batch_shape, initial_time_steps, self.num_writes, self.memory_size], dtype=tf.float32)

        linkage = self.temporal_linkage.get_initial_state(batch_shape=batch_shape)
        usage = tf.zeros([batch_shape, self.memory_size], dtype=tf.float32)
        read_words = tf.zeros([batch_shape, self.num_reads, self.word_size], dtype=tf.float32)
        return AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage,
            usage=usage,
            read_words=read_words,
        )

    def state_size(self):
        return AccessState(
            memory=tf.TensorShape([self.memory_size, self.word_size]),
            read_weights=tf.TensorShape([None, self.num_reads, self.memory_size]),
            write_weights=tf.TensorShape([None, self.num_writes, self.memory_size]),
            linkage=self.temporal_linkage.state_size,
            usage=self.usage_update.state_size,
            read_words=tf.TensorShape([self.num_reads, self.word_size])
        )
