# thinker_ai/agent/memory/humanoid_memory/dnc/access.py
import collections
import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc_new import addressing
from thinker_ai.agent.memory.humanoid_memory.dnc_new import util
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 保持 AccessState 为 namedtuple
AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))

class MemoryAccess(tf.keras.layers.Layer):

    def __init__(self, memory_size=128, word_size=20, num_reads=1, num_writes=1, name='memory_access'):
        super(MemoryAccess, self).__init__(name=name)
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes

        # 修改属性名，使其与子层的 name 参数一致
        self.write_content_weights = addressing.CosineWeights(
            num_heads=num_writes, word_size=word_size, name='write_content_weights')
        self.read_content_weights = addressing.CosineWeights(
            num_heads=num_reads, word_size=word_size, name='read_content_weights')

        # TemporalLinkage 和 Freeness 模块
        self.temporal_linkage = addressing.TemporalLinkage(
            memory_size=memory_size, num_writes=num_writes, name='temporal_linkage')
        self.freeness = addressing.Freeness(
            memory_size=memory_size, name='freeness')

        # 定义初始化器
        bias_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

        # 定义用于生成各个参数的 Dense 层，确保属性名与子层名称一致
        self.write_vectors = tf.keras.layers.Dense(
            units=self._num_writes * self._word_size,
            activation=None,
            name='write_vectors',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=True
        )
        self.erase_vectors = tf.keras.layers.Dense(
            units=self._num_writes * self._word_size,
            activation='sigmoid',
            name='erase_vectors',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=True
        )
        self.free_gate = tf.keras.layers.Dense(
            units=self._num_reads,
            activation='sigmoid',
            name='free_gate',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=True
        )
        self.allocation_gate = tf.keras.layers.Dense(
            units=self._num_writes,
            activation='sigmoid',
            name='allocation_gate',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=True
        )
        self.write_gate = tf.keras.layers.Dense(
            units=self._num_writes,
            activation='sigmoid',
            name='write_gate',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=True
        )
        self.read_mode = tf.keras.layers.Dense(
            units=self._num_reads * (1 + 2 * self._num_writes),
            activation=None,
            name='read_mode',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=True
        )

        self.write_strengths = tf.keras.layers.Dense(
            units=self._num_writes,
            activation='softplus',
            name='write_strengths',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=True
        )
        self.read_strengths = tf.keras.layers.Dense(
            units=self._num_reads,
            activation='softplus',
            name='read_strengths',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=True
        )

        self.write_keys = tf.keras.layers.Dense(
            units=self._num_writes * self._word_size,
            activation=None,
            name='write_keys',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=True
        )
        self.read_keys = tf.keras.layers.Dense(
            units=self._num_reads * self._word_size,
            activation=None,
            name='read_keys',
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=True
        )
    def _print_inputs(self, processed_inputs):
        for key, value in processed_inputs.items():
            tf.print(key + ":", value)
        return processed_inputs
    # 确保在 call 方法中调用了所有子层
    def call(self, inputs, training=False):
        prev_state = inputs['prev_state']
        processed_inputs = self._read_inputs(inputs)

        # 打印处理后的输入
        tf.print("MemoryAccess Layer: Processing inputs.")
        self._print_inputs(processed_inputs)

        # 更新使用率
        usage = self.freeness({
            'write_weights': prev_state.write_weights,
            'free_gate': processed_inputs['free_gate'],
            'read_weights': prev_state.read_weights,
            'prev_usage': prev_state.usage
        })

        # 计算写权重
        write_weights = self._write_weights(processed_inputs, prev_state.memory, usage)

        # 更新内存
        memory = self._erase_and_write(
            prev_state.memory,
            address=write_weights,
            reset_weights=processed_inputs['erase_vectors'],
            values=processed_inputs['write_vectors']
        )

        # 更新时序链路
        linkage_state = self.temporal_linkage({
            'write_weights': write_weights,
            'prev_linkage': prev_state.linkage
        })

        # 计算读取权重
        read_weights = self._read_weights(
            processed_inputs,
            memory=memory,
            prev_read_weights=prev_state.read_weights,
            link=linkage_state.link
        )
        read_words = tf.matmul(read_weights, memory)

        # 打印输出值的部分样本
        tf.print("MemoryAccess Layer: Output values (sample):", read_words[:2, :])

        # 检查 read_words 是否有 NaN 或 Inf
        tf.debugging.check_numerics(read_words, "read_words contains NaN or Inf")

        return read_words, AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage_state,
            usage=usage
        )

    # 更新 _read_inputs 方法，使用修改后的子层属性名
    def _read_inputs(self, inputs):
        controller_output = inputs['inputs']
        batch_size = tf.shape(controller_output)[0]

        # 使用修改后的子层属性名
        write_vectors = tf.reshape(self.write_vectors(controller_output),
                                   [batch_size, self._num_writes, self._word_size])
        erase_vectors = tf.reshape(self.erase_vectors(controller_output),
                                   [batch_size, self._num_writes, self._word_size])

        free_gate = self.free_gate(controller_output)
        allocation_gate = self.allocation_gate(controller_output)
        write_gate = self.write_gate(controller_output)

        read_mode = tf.nn.softmax(
            tf.reshape(self.read_mode(controller_output), [batch_size, self._num_reads, 1 + 2 * self._num_writes])
        )

        write_keys = tf.reshape(self.write_keys(controller_output),
                                [batch_size, self._num_writes, self._word_size])
        write_strengths = self.write_strengths(controller_output)

        read_keys = tf.reshape(self.read_keys(controller_output),
                               [batch_size, self._num_reads, self._word_size])
        read_strengths = self.read_strengths(controller_output)

        return {
            'read_content_keys': read_keys,
            'read_content_strengths': read_strengths,
            'write_content_keys': write_keys,
            'write_content_strengths': write_strengths,
            'write_vectors': write_vectors,
            'erase_vectors': erase_vectors,
            'free_gate': free_gate,
            'allocation_gate': allocation_gate,
            'write_gate': write_gate,
            'read_mode': read_mode,
        }

    def _erase_and_write(self, memory, address, reset_weights, values):
        with tf.name_scope('erase_memory'):
            # 擦除操作：
            # 1. 首先将写入权重 `address` 和擦除权重 `reset_weights` 进行维度扩展，以便进行广播操作。
            expand_address = tf.expand_dims(address, 3)  # [batch_size, num_writes, memory_size, 1]
            reset_weights = tf.expand_dims(reset_weights, 2)  # [batch_size, num_writes, 1, word_size]

            # 2. 计算加权的擦除权重，通过乘法将写入权重 `address` 和擦除权重 `reset_weights` 相乘，
            #    得到每个存储器位置的擦除比例。
            weighted_resets = expand_address * reset_weights  # [batch_size, num_writes, memory_size, word_size]

            # 3. 通过累积乘积计算擦除门 `reset_gate`，表示要擦除的内容。擦除门控制存储器的部分内容会被保留。
            #    `reduce_prod` 是一种自定义操作，用于高效地计算张量沿指定轴的乘积。
            reset_gate = util.reduce_prod(1 - weighted_resets, axis=1)  # [batch_size, memory_size, word_size]

            # 4. 更新内存，按元素乘法将擦除门应用到内存上，抹去部分内容。
            memory = memory * reset_gate  # [batch_size, memory_size, word_size]

        with tf.name_scope('additive_write'):
            # 写入操作：
            # 1. 使用 `tf.matmul`（矩阵乘法）将写入权重 `address` 与要写入的值 `values` 结合，
            #    生成写入矩阵 `add_matrix`。
            #    矩阵乘法中，`adjoint_a=True` 表示第一个输入张量要进行转置操作。
            add_matrix = tf.matmul(address, values, adjoint_a=True)  # [batch_size, memory_size, word_size]

            # 2. 将计算得到的写入矩阵加到当前内存上，完成写入操作。
            memory = memory + add_matrix  # [batch_size, memory_size, word_size]

        # 返回更新后的内存状态
        return memory
    # 更新 _write_weights 方法，使用修改后的子层属性名
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

    # 更新 _read_weights 方法，使用修改后的子层属性名
    def _read_weights(self, inputs, memory, prev_read_weights, link):
        with tf.name_scope('read_weights'):
            # 使用修改后的属性名
            content_weights = self.read_content_weights({
                'memory': memory,
                'keys': inputs['read_content_keys'],
                'strengths': inputs['read_content_strengths']
            })

            forward_weights = self.temporal_linkage.directional_read_weights(
                link, prev_read_weights, forward=True)
            backward_weights = self.temporal_linkage.directional_read_weights(
                link, prev_read_weights, forward=False)

            backward_mode = inputs['read_mode'][:, :, :self._num_writes]
            forward_mode = inputs['read_mode'][:, :, self._num_writes:2 * self._num_writes]
            content_mode = inputs['read_mode'][:, :, 2 * self._num_writes]

            tf.debugging.assert_shapes([
                (content_weights, ['batch_size', 'num_reads', 'memory_size']),
                (forward_weights, ['batch_size', 'num_reads', 'num_writes', 'memory_size']),
                (backward_weights, ['batch_size', 'num_reads', 'num_writes', 'memory_size']),
                (backward_mode, ['batch_size', 'num_reads', 'num_writes']),
                (forward_mode, ['batch_size', 'num_reads', 'num_writes']),
                (content_mode, ['batch_size', 'num_reads'])
            ])

            read_weights = (
                    tf.expand_dims(content_mode, -1) * content_weights +
                    tf.reduce_sum(tf.expand_dims(forward_mode, -1) * forward_weights, axis=2) +
                    tf.reduce_sum(tf.expand_dims(backward_mode, -1) * backward_weights, axis=2)
            )

            return read_weights

    @property
    def state_size(self):
        return AccessState(
            memory=tf.TensorShape([self._memory_size, self._word_size]),
            read_weights=tf.TensorShape([self._num_reads, self._memory_size]),
            write_weights=tf.TensorShape([self._num_writes, self._memory_size]),
            linkage=self.linkage.state_size,
            usage=self.freeness.state_size)

    @property
    def output_size(self):
        return tf.TensorShape([self._num_reads, self._word_size])