import collections
import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc import addressing
from thinker_ai.agent.memory.humanoid_memory.dnc import util
AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))



class MemoryAccess(tf.keras.layers.Layer):
    """
    该模块允许多个读取和写入头对外部存储器进行操作。DNC 是一种增强型神经网络模型，
    它通过外部存储器来扩展其记忆能力。这个 `MemoryAccess` 模块负责处理外部存储器
    的读取、写入和管理，确保多个读取和写入操作能够高效地并发运行。该模块使用了以下
    机制：

    - `addressing.TemporalLinkage`：用于追踪每个写入头在存储器中的写入顺序，确保写入
      的内容可以根据时间顺序进行管理。
    - `addressing.FreenessAllocator`：用于追踪存储器的使用情况，保证在控制器允许的情况下，
      存储器可以被释放，避免不必要占用。

    动态神经计算机（DNC）不仅能像传统神经网络一样学习数据的模式，还可以借助外部存储器记住大量数据，
    并对其进行复杂的操作，如读写和跟踪时间顺序。
    """

    def __init__(self, memory_size=128, word_size=20, num_reads=1, num_writes=1, name='memory_access'):
        """初始化 MemoryAccess 模块。

        Args:
          memory_size: 存储器的槽数量，即存储器中的地址数。
          word_size: 每个存储器槽的宽度（存储器中每个位置存储的值的大小）。
          num_reads: 读取头的数量，表示同时可以有多少个读取操作。
          num_writes: 写入头的数量，表示同时可以有多少个写入操作。
          name: 模块的名称（可选）。
        """
        super(MemoryAccess, self).__init__(name=name)
        # self.dense_read_keys = tf.keras.layers.Dense(units=..., activation=...)
        # self.dense_read_strengths = tf.keras.layers.Dense(units=..., activation=...)
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes

        # 定义用于计算内容权重的 CosineWeights 模块
        self._write_content_weights_mod = addressing.CosineWeights(num_writes, word_size, name='write_content_weights')
        self._read_content_weights_mod = addressing.CosineWeights(num_reads, word_size, name='read_content_weights')

        # TemporalLinkage 和 Freeness 模块
        self._linkage = addressing.TemporalLinkage(memory_size, num_writes)
        self._freeness = addressing.Freeness(memory_size)

        # 定义用于生成各个参数的 Dense 层
        self.write_vector_dense = tf.keras.layers.Dense(self._num_writes * self._word_size, activation=None,
                                                        name='write_vectors')
        self.erase_vector_dense = tf.keras.layers.Dense(self._num_writes * self._word_size, activation=tf.sigmoid,
                                                        name='erase_vectors')
        self.free_gate_dense = tf.keras.layers.Dense(self._num_reads, activation=tf.sigmoid, name='free_gate')
        self.allocation_gate_dense = tf.keras.layers.Dense(self._num_writes, activation=tf.sigmoid,
                                                           name='allocation_gate')
        self.write_gate_dense = tf.keras.layers.Dense(self._num_writes, activation=tf.sigmoid, name='write_gate')
        self.read_mode_dense = tf.keras.layers.Dense(self._num_reads * (1 + 2 * self._num_writes), activation=None,
                                                     name='read_mode')

        # **添加用于生成写入和读取强度的 Dense 层**
        self.write_strengths_dense = tf.keras.layers.Dense(self._num_writes, activation=tf.nn.softplus,
                                                           name='write_strengths')
        self.read_strengths_dense = tf.keras.layers.Dense(self._num_reads, activation=tf.nn.softplus,
                                                          name='read_strengths')

        # **添加用于生成写入和读取键的 Dense 层**
        self.write_keys_dense = tf.keras.layers.Dense(self._num_writes * self._word_size, activation=None,
                                                      name='write_keys')
        self.read_keys_dense = tf.keras.layers.Dense(self._num_reads * self._word_size, activation=None,
                                                     name='read_keys')

    def call(self, inputs, training=False):
        """将 MemoryAccess 模块连接到计算图中，执行内存操作。
        该方法将当前输入、存储器状态和读写操作整合在一起，执行存储器的读取和写入，并返回读取的内容和更新后的状态。主要步骤包括：
        •	解析输入数据（如写入向量、擦除向量等）。
        •	通过 Freeness 计算当前存储器的使用情况。
        •	调用 TemporalLinkage 更新写入顺序，并通过 CosineWeights 计算写入权重。
        •	根据输入执行存储器的擦除和写入操作。
        •	计算读取权重，并读取存储器中的内容。

        Args:
          inputs: 输入字典，包含控制器的输入、前一时间步的状态等信息。
          training: 是否在训练模式下执行（默认值为 False）。
        """
        # 从输入中提取前一时间步的状态
        prev_state = inputs['prev_state']
        # 解析输入并进行预处理
        processed_inputs = self._read_inputs(inputs)
        # 仅保留关键信息的打印
        print("MemoryAccess Layer: Processing inputs.")
        # 解析输入并进行预处理
        inputs = self._read_inputs(inputs)

        # 根据自由门、读取权重和写入权重更新存储器的使用情况
        usage = self._freeness({
            'write_weights': prev_state.write_weights,
            'free_gate': inputs['free_gate'],
            'read_weights': prev_state.read_weights,
            'prev_usage': prev_state.usage
        })

        # 写入存储器，并处理更新后的存储器状态
        write_weights = self._write_weights(inputs, prev_state.memory, usage)
        memory = self._erase_and_write(
            prev_state.memory,
            address=write_weights,
            reset_weights=inputs['erase_vectors'],
            values=inputs['write_vectors']
        )
        # 根据新的写入权重更新时间顺序链路状态
        linkage_state = self._linkage({
            'write_weights': write_weights,
            'prev_linkage': prev_state.linkage
        })

        # 根据更新后的存储器状态和链路状态进行读取
        read_weights = self._read_weights(
            inputs,
            memory=memory,
            prev_read_weights=prev_state.read_weights,
            link=linkage_state.link
        )
        read_words = tf.matmul(read_weights, memory)
        # 添加输出值的打印
        print("MemoryAccess Layer: Output values (sample):", read_words.numpy()[0, :2, :])
        # 返回读取结果及更新后的状态
        return read_words, AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage_state,
            usage=usage
        )

    def _read_inputs(self, inputs):
        """对输入进行转换和处理以适配 MemoryAccess 模块。

        Args:
          inputs: 原始输入字典，包含控制器的输出。

        Returns:
          处理后的输入字典，包含写入向量、擦除向量、自由门等信息。
        """
        # 从 inputs 字典中提取控制器的输出
        controller_output = inputs['inputs']
        batch_size = tf.shape(controller_output)[0]

        # 生成写入向量和擦除向量
        write_vectors = tf.reshape(self.write_vector_dense(controller_output),
                                   [batch_size, self._num_writes, self._word_size])
        erase_vectors = tf.reshape(self.erase_vector_dense(controller_output),
                                   [batch_size, self._num_writes, self._word_size])

        # 生成自由门、分配门和写入门
        free_gate = self.free_gate_dense(controller_output)
        allocation_gate = self.allocation_gate_dense(controller_output)
        write_gate = self.write_gate_dense(controller_output)

        # 计算读取模式的 Softmax
        num_read_modes = 1 + 2 * self._num_writes
        read_mode = tf.nn.softmax(
            tf.reshape(self.read_mode_dense(controller_output), [batch_size, self._num_reads, num_read_modes]))

        # 生成写入键和写入强度
        write_keys = tf.reshape(self.write_keys_dense(controller_output),
                                [batch_size, self._num_writes, self._word_size])
        write_strengths = self.write_strengths_dense(controller_output)

        # 生成读取键和读取强度
        read_keys = tf.reshape(self.read_keys_dense(controller_output), [batch_size, self._num_reads, self._word_size])
        read_strengths = self.read_strengths_dense(controller_output)

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

    def _erase_and_write(self,memory, address, reset_weights, values):
        """在外部存储器上执行擦除和写入操作的模块。

        在动态神经计算机（DNC）等神经网络架构中，存储器用于存储和更新模型的外部状态。
        这个函数的核心是根据输入的“擦除权重”（reset_weights）和“写入权重”（address）来更新
        存储器中的信息。它由两个主要步骤组成：擦除操作和写入操作。

        数学背景：
        1. 擦除操作：
           M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)
           其中：
             - M_t'(i)：表示擦除后的存储器值。
             - M_{t-1}(i)：表示之前的存储器状态。
             - w_t(i)：是写入权重，表示每个写头写入到存储器中每个位置的权重。
             - e_t：是擦除权重，表示要擦除存储器位置的程度。
           这一步的作用是根据写入权重和擦除权重选择性地抹去存储器中的部分信息，确保即将写入的新信息不会与之前的内容冲突。

        2. 写入操作：
           M_t(i) = M_t'(i) + w_t(i) * a_t
           其中：
             - M_t(i)：表示写入后的存储器状态。
             - a_t：表示要写入的新信息。
           写入操作在擦除后的存储器中写入新信息。写入的内容根据写入权重的不同，可以同时写入多个存储器位置。

        参数说明：
          memory: 3维张量，形状为 `[batch_size, memory_size, word_size]`，表示当前的存储器状态。
          address: 3维张量，形状为 `[batch_size, num_writes, memory_size]`，表示写入权重。
          reset_weights: 3维张量，形状为 `[batch_size, num_writes, word_size]`，表示擦除权重。
          values: 3维张量，形状为 `[batch_size, num_writes, word_size]`，表示要写入的值。

        返回值：
          更新后的存储器状态，形状为 `[batch_size, memory_size, word_size]`。
        """
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

            # 4. 更新存储器，按元素乘法将擦除门应用到存储器上，抹去部分内容。
            memory = memory * reset_gate  # [batch_size, memory_size, word_size]

        with tf.name_scope('additive_write'):
            # 写入操作：
            # 1. 使用 `tf.matmul`（矩阵乘法）将写入权重 `address` 与要写入的值 `values` 结合，
            #    生成写入矩阵 `add_matrix`。
            #    矩阵乘法中，`adjoint_a=True` 表示第一个输入张量要进行转置操作。
            add_matrix = tf.matmul(address, values, adjoint_a=True)  # [batch_size, memory_size, word_size]

            # 2. 将计算得到的写入矩阵加到当前存储器上，完成写入操作。
            memory = memory + add_matrix  # [batch_size, memory_size, word_size]

        # 返回更新后的存储器状态
        return memory

    def _write_weights(self, inputs, memory, usage):
        """计算写入权重，决定写入哪些存储器位置。

        Args:
          inputs: 处理后的输入字典。
          memory: 当前存储器状态。
          usage: 当前存储器使用情况。

        Returns:
          写入权重，表示每个写入头向哪些存储器位置写入。
        """
        with tf.name_scope('write_weights'):
            write_content_weights = self._write_content_weights_mod({
                'memory': memory,
                'keys': inputs['write_content_keys'],
                'strengths': inputs['write_content_strengths']
            })

            write_allocation_weights = self._freeness.write_allocation_weights(
                usage=usage,
                write_gates=inputs['allocation_gate'] * inputs['write_gate'],
                num_writes=self._num_writes)

            allocation_gate = tf.expand_dims(inputs['allocation_gate'], -1)
            write_gate = tf.expand_dims(inputs['write_gate'], -1)

            return write_gate * (
                    allocation_gate * write_allocation_weights +
                    (1 - allocation_gate) * write_content_weights)

    def _read_weights(self, inputs, memory, prev_read_weights, link):
        """计算每个读取头的读取权重。

        Args:
          inputs: 处理后的输入字典，包括读取模式、读取键和强度。
          memory: 当前的存储器状态，用于根据内容权重进行读取。
          prev_read_weights: 上一时间步的读取权重，用于计算时间顺序的权重。
          link: 由 `TemporalLinkage` 模块计算的时间链路，用于前向和后向读取权重。

        Returns:
          read_weights: 每个读取头的最终读取权重，表示从哪些存储器位置读取内容。
        """
        with tf.name_scope('read_weights'):
            # 计算内容权重
            content_weights = self._read_content_weights_mod({
                'memory': memory,
                'keys': inputs['read_content_keys'],
                'strengths': inputs['read_content_strengths']
            })  # 形状: [batch_size, num_reads, memory_size]

            # 计算前向和后向权重
            forward_weights = self._linkage.directional_read_weights(
                link, prev_read_weights, forward=True)  # 形状: [batch_size, num_reads, num_writes, memory_size]
            backward_weights = self._linkage.directional_read_weights(
                link, prev_read_weights, forward=False)  # 同上

            # 获取读取模式
            backward_mode = inputs['read_mode'][:, :, :self._num_writes]  # 形状: [batch_size, num_reads, num_writes]
            forward_mode = inputs['read_mode'][:, :, self._num_writes:2 * self._num_writes]  # 同上
            content_mode = inputs['read_mode'][:, :, 2 * self._num_writes]  # 形状: [batch_size, num_reads]

            # 检查张量形状
            tf.debugging.assert_shapes([
                (content_weights, ['batch_size', 'num_reads', 'memory_size']),
                (forward_weights, ['batch_size', 'num_reads', 'num_writes', 'memory_size']),
                (backward_weights, ['batch_size', 'num_reads', 'num_writes', 'memory_size']),
                (backward_mode, ['batch_size', 'num_reads', 'num_writes']),
                (forward_mode, ['batch_size', 'num_reads', 'num_writes']),
                (content_mode, ['batch_size', 'num_reads'])
            ])

            # 计算读取权重
            read_weights = (
                    tf.expand_dims(content_mode, -1) * content_weights +
                    tf.reduce_sum(tf.expand_dims(forward_mode, -1) * forward_weights, axis=2) +
                    tf.reduce_sum(tf.expand_dims(backward_mode, -1) * backward_weights, axis=2)
            )

            return read_weights  # 形状: [batch_size, num_reads, memory_size]

    @property
    def state_size(self):
        """返回状态张量的形状元组。

        该属性用于定义 `MemoryAccess` 模块中每个状态变量的大小。状态包括存储器、读取权重、写入权重、
        链接状态（由 TemporalLinkage 管理）和使用情况（由 Freeness 管理）。

        Returns:
          AccessState: 包含存储器状态、读取权重、写入权重、链路状态和使用情况的元组。
        """
        return AccessState(
            memory=tf.TensorShape([self._memory_size, self._word_size]),
            read_weights=tf.TensorShape([self._num_reads, self._memory_size]),
            write_weights=tf.TensorShape([self._num_writes, self._memory_size]),
            linkage=self._linkage.state_size,
            usage=self._freeness.state_size)

    @property
    def output_size(self):
        """返回模块的输出大小。

        输出是从存储器读取的单词，形状为 `[num_reads, word_size]`，表示每个读取头读取的值。

        Returns:
          TensorShape: 表示输出的形状。
        """
        return tf.TensorShape([self._num_reads, self._word_size])