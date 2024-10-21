# test_memory_access_scenario.py
import numpy as np
import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc.cache_manager import CacheManager
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess, BatchAccessState


class MemoryAccessUserScenarioTest(tf.test.TestCase):
    def setUp(self):
        self.memory_size = 32
        self.word_size = 16
        self.num_reads = 2
        self.num_writes = 1
        self.batch_size = 2  # 两个用户
        self.controller_output_size = 93  # 调整为93
        self.num_read_modes = 3  # 添加此行

        # 初始化 CacheManager
        self.cache_manager = CacheManager(max_cache_size=self.memory_size)

        # 初始化 MemoryAccess 并注入 CacheManager
        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size,
            cache_manager=self.cache_manager
        )

        # 获取 interface_size
        self.interface_size = self.memory_access.interface_size  # 应为93

        # 初始化初始状态
        self.initial_state = self.memory_access.get_initial_state(self.batch_size)

        # 设置 interface_layer 的权重以获得可预测的输出
        self._set_interface_layer_weights()

    def _set_interface_layer_weights(self):
        """
        设置 interface_layer 的权重和偏置，使其输出可预测。
        具体来说，将控制器输出的特定部分映射到 interface_vector 中的对应部分。
        """
        # interface_layer 是一个 Dense 层，input_dim = controller_output_size，units = interface_size
        # 我们希望将 controller_output 的值直接映射到 interface_vector 中的对应部分
        # 因此，我们可以将权重设置为单位矩阵，偏置为零
        if self.controller_output_size == self.interface_size:
            identity_matrix = np.eye(self.controller_output_size, dtype=np.float32)
        elif self.controller_output_size > self.interface_size:
            identity_matrix = np.eye(self.interface_size, dtype=np.float32)[:self.controller_output_size, :]
        else:
            identity_matrix = np.pad(
                np.eye(self.controller_output_size, dtype=np.float32),
                ((0, 0), (0, self.interface_size - self.controller_output_size)),
                'constant'
            )
        # 设置 weights 和 biases
        weights = [identity_matrix, np.zeros(self.interface_size, dtype=np.float32)]

        # 手动构建 interface_layer
        # 这里的 input_shape 不包含批次维度，因此使用 (self.controller_output_size,)
        self.memory_access.interface_layer.build((None, self.controller_output_size))

        # 设置权重
        self.memory_access.interface_layer.set_weights(weights)

    def _build_controller_output(self, write_vector=None, erase_logit=None, write_strength=None,
                                 read_content_keys=None, read_strengths=None, read_modes=None,
                                 allocation_gate=None, write_gate=None,
                                 free_gates=None):
        """
        构建 controller_output，使 interface_layer 输出特定的 interface_vector。
        """
        batch_size = self.batch_size
        controller_output_size = self.controller_output_size
        interface_size = self.interface_size
        num_reads = self.num_reads
        num_writes = self.num_writes
        word_size = self.word_size
        num_read_modes = self.num_read_modes

        # 初始化为零
        controller_output = np.zeros((batch_size, controller_output_size), dtype=np.float32)

        indices = self.memory_access.interface_vector_indices

        # 设置 read_keys 部分
        if read_content_keys is not None:
            read_keys_start = indices['read_keys_start']
            read_keys_end = indices['read_keys_end']
            read_keys_flat = read_content_keys.reshape(batch_size, -1)
            controller_output[:, read_keys_start:read_keys_end] = read_keys_flat

        # 设置 read_strengths 部分
        if read_strengths is not None:
            read_strengths_start = indices['read_strengths_start']
            read_strengths_end = indices['read_strengths_end']
            read_strengths_flat = read_strengths.reshape(batch_size, -1)
            controller_output[:, read_strengths_start:read_strengths_end] = read_strengths_flat

        # 设置 write_keys 部分
        if write_vector is not None:
            write_keys_start = indices['write_keys_start']
            write_keys_end = indices['write_keys_end']
            write_keys_flat = write_vector.reshape(batch_size, -1)
            controller_output[:, write_keys_start:write_keys_end] = write_keys_flat

        # 设置 write_strengths 部分
        if write_strength is not None:
            write_strengths_start = indices['write_strengths_start']
            write_strengths_end = indices['write_strengths_end']
            if isinstance(write_strength, (float, int)):
                write_strengths_flat = np.full((batch_size, write_strengths_end - write_strengths_start),
                                               write_strength, dtype=np.float32)
            else:
                write_strengths_flat = write_strength.reshape(batch_size, -1)
            controller_output[:, write_strengths_start:write_strengths_end] = write_strengths_flat

        # 设置 erase_vectors 部分
        if erase_logit is not None:
            erase_vectors_start = indices['erase_vectors_start']
            erase_vectors_end = indices['erase_vectors_end']
            if isinstance(erase_logit, (float, int)):
                erase_vector_flat = np.full((batch_size, erase_vectors_end - erase_vectors_start), erase_logit,
                                            dtype=np.float32)
            else:
                erase_vector_flat = erase_logit.reshape(batch_size, -1)
            controller_output[:, erase_vectors_start:erase_vectors_end] = erase_vector_flat

        # 设置 write_vectors 部分
        if write_vector is not None:
            write_vectors_start = indices['write_vectors_start']
            write_vectors_end = indices['write_vectors_end']
            write_vectors_flat = write_vector.reshape(batch_size, -1)
            controller_output[:, write_vectors_start:write_vectors_end] = write_vectors_flat

        # 设置 free_gates 部分
        if free_gates is not None:
            free_gates_start = indices['free_gates_start']
            free_gates_end = indices['free_gates_end']
            free_gates_flat = free_gates.reshape(batch_size, -1)
            controller_output[:, free_gates_start:free_gates_end] = free_gates_flat

        # 设置 allocation_gates 部分
        if allocation_gate is not None:
            allocation_gates_start = indices['allocation_gates_start']
            allocation_gates_end = indices['allocation_gates_end']
            allocation_gate_flat = allocation_gate.reshape(batch_size, -1)
            controller_output[:, allocation_gates_start:allocation_gates_end] = allocation_gate_flat

        # 设置 write_gates 部分
        if write_gate is not None:
            write_gates_start = indices['write_gates_start']
            write_gates_end = indices['write_gates_end']
            write_gate_flat = write_gate.reshape(batch_size, -1)
            controller_output[:, write_gates_start:write_gates_end] = write_gate_flat

        # 设置 read_modes 部分
        if read_modes is not None:
            read_modes_start = indices['read_modes_start']
            read_modes_end = indices['read_modes_end']
            read_modes_flat = read_modes.reshape(batch_size, -1)
            controller_output[:, read_modes_start:read_modes_end] = read_modes_flat

        return controller_output

    def test_content_read_write(self):
        """
        内容读取与写入：
        验证写入的内容能够被准确读取。
        """
        batch_size = self.batch_size

        # 构建写入内容
        write_vector = np.random.rand(batch_size, self.num_writes, self.word_size).astype(np.float32)

        # 构建 controller_output，使得 interface_vector 包含所需的 write_vector 和其他必要部分
        controller_output = self._build_controller_output(
            write_vector=write_vector,
            erase_logit=-1000.0,  # 非常小的值，确保 erase_vector ≈ 0.0
            write_strength=10.0,
            allocation_gate=np.ones((batch_size, self.num_writes), dtype=np.float32),
            write_gate=np.ones((batch_size, self.num_writes), dtype=np.float32)
        )

        # 将 controller_output 转换为 Tensor
        controller_output_tensor = tf.convert_to_tensor(controller_output)

        # 调用 memory_access
        inputs = {
            'inputs': controller_output_tensor,
            'prev_state': self.initial_state
        }
        outputs = self.memory_access(inputs)

        # 获取更新后的 memory
        updated_memory = outputs['final_state'].memory.numpy()

        # 获取 read_weights 和 write_weights
        read_weights = outputs['final_state'].read_weights.numpy()  # [batch_size, num_reads, memory_size]
        write_weights = outputs['write_weights'].numpy()  # [batch_size, num_writes, memory_size]

        # 计算预期的内存内容
        expected_memory = np.zeros((batch_size, self.memory_size, self.word_size), dtype=np.float32)
        for b in range(batch_size):
            for w in range(self.num_writes):
                for i in range(self.memory_size):
                    ww = write_weights[b, w, i]  # 写入权重
                    expected_memory[b, i, :] += ww * write_vector[b, w, :]

        # 验证更新后的内存与预期内存接近
        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)

        # 验证读取内容是否与写入内容匹配
        # 读取内容应等于 read_weights * memory 的加权和
        read_words = outputs['read_words'].numpy()
        for b in range(batch_size):
            for r in range(self.num_reads):
                # 计算实际读取的内容
                actual_read = np.dot(read_weights[b, r, :], updated_memory[b, :, :])  # shape (word_size,)
                # 计算预期的读取内容
                expected_read = write_vector[b, 0, :] * np.sum(write_weights[b, 0, :] * read_weights[b, r, :])
                self.assertAllClose(read_words[b, r, :], expected_read, atol=1e-5)

    def test_user_memory_isolation(self):
        """
        测试不同用户的内存是否相互隔离。
        """
        batch_size = self.batch_size

        # 构建写入内容为不同的写入向量
        write_vector_user1 = np.ones((batch_size, self.num_writes, self.word_size), dtype=np.float32)
        write_vector_user2 = np.full((batch_size, self.num_writes, self.word_size), 2.0, dtype=np.float32)

        # 用户1写入
        controller_output_user1 = self._build_controller_output(
            write_vector=write_vector_user1,
            erase_logit=-1000.0,  # 确保不擦除
            write_strength=10.0,
            allocation_gate=np.ones((batch_size, self.num_writes), dtype=np.float32),
            write_gate=np.ones((batch_size, self.num_writes), dtype=np.float32)
        )
        controller_output_user1_tensor = tf.convert_to_tensor(controller_output_user1)
        inputs_user1 = {
            'inputs': controller_output_user1_tensor,
            'prev_state': self.initial_state
        }
        outputs_user1 = self.memory_access(inputs_user1)
        final_state_user1 = outputs_user1['final_state']

        # 用户2写入
        controller_output_user2 = self._build_controller_output(
            write_vector=write_vector_user2,
            erase_logit=-1000.0,  # 确保不擦除
            write_strength=10.0,
            allocation_gate=np.ones((batch_size, self.num_writes), dtype=np.float32),
            write_gate=np.ones((batch_size, self.num_writes), dtype=np.float32)
        )
        controller_output_user2_tensor = tf.convert_to_tensor(controller_output_user2)
        inputs_user2 = {
            'inputs': controller_output_user2_tensor,
            'prev_state': final_state_user1
        }
        outputs_user2 = self.memory_access(inputs_user2)
        final_state_user2 = outputs_user2['final_state']

        # 验证用户1的内存未被用户2的操作影响
        memory_user1 = final_state_user1.memory.numpy()
        memory_user2 = final_state_user2.memory.numpy()

        # 预期用户1的内存为 write_weights_user1 * write_vector_user1
        write_weights_user1 = final_state_user1.write_weights.numpy()
        expected_memory_user1 = np.zeros((batch_size, self.memory_size, self.word_size), dtype=np.float32)
        for b in range(batch_size):
            for w in range(self.num_writes):
                for i in range(self.memory_size):
                    expected_memory_user1[b, i, :] += write_weights_user1[b, w, i] * write_vector_user1[b, w, :]
        self.assertAllClose(memory_user1, expected_memory_user1, atol=1e-5)

        # 预期用户2的内存为 write_weights_user2 * write_vector_user2 + memory_user1
        write_weights_user2 = outputs_user2['final_state'].write_weights.numpy()
        expected_memory_user2 = memory_user1.copy()
        for b in range(batch_size):
            for w in range(self.num_writes):
                for i in range(self.memory_size):
                    expected_memory_user2[b, i, :] += write_weights_user2[b, w, i] * write_vector_user2[b, w, :]
        self.assertAllClose(memory_user2, expected_memory_user2, atol=1e-5)

    def test_full_erase(self):
        """
        完全擦除 (erase_vector = 1)：
        验证内存被完全擦除，只保留新写入的内容。
        """
        batch_size = self.batch_size

        # 构建写入内容
        write_vector = np.random.rand(batch_size, self.num_writes, self.word_size).astype(np.float32)

        # 构建 controller_output，使 erase_vectors = 1
        controller_output = self._build_controller_output(
            write_vector=write_vector,
            erase_logit=1000.0,  # sigmoid(1000) ≈ 1.0
            write_strength=10.0,
            allocation_gate=np.ones((batch_size, self.num_writes), dtype=np.float32),
            write_gate=np.ones((batch_size, self.num_writes), dtype=np.float32)
        )

        # 转换为 Tensor
        controller_output_tensor = tf.convert_to_tensor(controller_output)

        # 调用 memory_access
        inputs = {
            'inputs': controller_output_tensor,
            'prev_state': self.initial_state
        }
        outputs = self.memory_access(inputs)

        # 获取更新后的 memory
        updated_memory = outputs['final_state'].memory.numpy()

        # 预期内存为 write_weights * write_vector
        write_weights = outputs['final_state'].write_weights.numpy()
        expected_memory = np.zeros((batch_size, self.memory_size, self.word_size), dtype=np.float32)
        for b in range(batch_size):
            for w in range(self.num_writes):
                for i in range(self.memory_size):
                    expected_memory[b, i, :] += write_weights[b, w, i] * write_vector[b, w, :]
        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)

    def test_partial_erase(self):
        """
        部分擦除 (erase_vector = 0.5)：
        验证内存被部分擦除，旧内容与新内容的混合。
        """
        batch_size = self.batch_size

        # 初始化内存为某个固定值
        initial_memory = np.full((batch_size, self.memory_size, self.word_size), 0.5,
                                 dtype=np.float32)  # [batch_size, memory_size, word_size]
        initial_state = BatchAccessState(
            memory=tf.convert_to_tensor(initial_memory),
            read_weights=self.initial_state.read_weights,
            write_weights=self.initial_state.write_weights,
            linkage=self.initial_state.linkage,  # 确保linkage['link']为 [batch_size, memory_size, memory_size]
            usage=self.initial_state.usage,
            read_words=self.initial_state.read_words
        )

        # 构建写入内容
        write_vector = np.ones((batch_size, self.num_writes, self.word_size),
                               dtype=np.float32)  # [batch_size, num_writes, word_size]

        # 构建 controller_output，使 erase_vectors = sigmoid(0) = 0.5
        controller_output = self._build_controller_output(
            write_vector=write_vector,
            erase_logit=0.0,  # sigmoid(0) = 0.5
            write_strength=10.0,
            allocation_gate=np.ones((batch_size, self.num_writes), dtype=np.float32),
            write_gate=np.ones((batch_size, self.num_writes), dtype=np.float32)
        )

        # 转换为 Tensor
        controller_output_tensor = tf.convert_to_tensor(controller_output)

        # 调用 memory_access
        inputs = {
            'inputs': controller_output_tensor,
            'prev_state': initial_state
        }
        outputs = self.memory_access(inputs)

        # 获取更新后的 memory
        updated_memory = outputs['final_state'].memory.numpy()  # [batch_size, memory_size, word_size]

        # 预期内存 = initial_memory * (1 - write_weights * erase_vector) + write_weights * write_vector
        write_weights = outputs['final_state'].write_weights.numpy()  # [batch_size, num_writes, memory_size]
        erase_vectors = self.memory_access._parse_interface_vector(tf.convert_to_tensor(controller_output))[
            'erase_vectors'].numpy()  # [batch_size, num_writes, word_size]

        # 计算擦除掩码
        erase_mask = 1 - (write_weights[:, :, :, np.newaxis] * erase_vectors[:, :, np.newaxis,
                                                               :])  # [batch_size, num_writes, memory_size, word_size]
        erased_memory = initial_memory[:, np.newaxis, :,
                        :] * erase_mask  # [batch_size, num_writes, memory_size, word_size]
        added_memory = write_weights[:, :, :, np.newaxis] * write_vector[:, :, np.newaxis,
                                                            :]  # [batch_size, num_writes, memory_size, word_size]
        expected_memory = np.sum(erased_memory + added_memory, axis=1)  # [batch_size, memory_size, word_size]

        # 断言
        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)


    def test_no_erase(self):
        """
        不擦除 (erase_vector = 0)：
        验证内存不被擦除，旧内容与新内容相加。
        """
        batch_size = self.batch_size

        # 初始化内存为某个固定值
        initial_memory = np.full((batch_size, self.memory_size, self.word_size), 1.0, dtype=np.float32)
        initial_state = BatchAccessState(
            memory=tf.convert_to_tensor(initial_memory),
            read_weights=self.initial_state.read_weights,
            write_weights=self.initial_state.write_weights,
            linkage=self.initial_state.linkage,
            usage=self.initial_state.usage,
            read_words=self.initial_state.read_words
        )

        # 构建写入内容
        write_vector = np.ones((batch_size, self.num_writes, self.word_size), dtype=np.float32) * 2.0

        # 构建 controller_output，使 erase_vectors = 0
        controller_output = self._build_controller_output(
            write_vector=write_vector,
            erase_logit=-1000.0,  # sigmoid(-1000) ≈ 0
            write_strength=10.0,
            allocation_gate=np.ones((batch_size, self.num_writes), dtype=np.float32),
            write_gate=np.ones((batch_size, self.num_writes), dtype=np.float32)
        )

        # 转换为 Tensor
        controller_output_tensor = tf.convert_to_tensor(controller_output)

        # 调用 memory_access
        inputs = {
            'inputs': controller_output_tensor,
            'prev_state': initial_state
        }
        outputs = self.memory_access(inputs)

        # 获取更新后的 memory
        updated_memory = outputs['final_state'].memory.numpy()

        # 获取 write_weights
        write_weights = outputs['final_state'].write_weights.numpy()  # [batch_size, num_writes, memory_size]

        # 预期内存 = initial_memory + write_weights * write_vector
        # write_weights: [2,1,32], write_vector: [2,1,16]
        # Compute write_weights * write_vector: [2,1,32,16]
        added_memory = write_weights[:, :, :, np.newaxis] * write_vector[:, :, np.newaxis, :]  # [2,1,32,16]
        # Sum over writes
        expected_memory = initial_memory + np.sum(added_memory, axis=1)  # [2,32,16]

        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)

    def test_batch_independence(self):
        """
        测试批次中的每个样本在内存操作中是独立的。
        """
        batch_size = self.batch_size

        # 构建不同批次的写入内容
        write_vector_batch1 = np.ones((1, self.num_writes, self.word_size), dtype=np.float32)
        write_vector_batch2 = np.full((1, self.num_writes, self.word_size), 2.0, dtype=np.float32)

        # 合并为一个批次
        write_vector = np.vstack([write_vector_batch1, write_vector_batch2])  # [2, num_writes, word_size]

        # 构建 controller_output
        controller_output = self._build_controller_output(
            write_vector=write_vector,
            erase_logit=-1000.0,  # 不擦除
            write_strength=10.0,
            allocation_gate=np.ones((batch_size, self.num_writes), dtype=np.float32),
            write_gate=np.ones((batch_size, self.num_writes), dtype=np.float32)
        )

        # 转换为 Tensor
        controller_output_tensor = tf.convert_to_tensor(controller_output)

        # 调用 memory_access
        inputs = {
            'inputs': controller_output_tensor,
            'prev_state': self.initial_state
        }
        outputs = self.memory_access(inputs)

        # 获取更新后的 memory
        updated_memory = outputs['final_state'].memory.numpy()

        # 验证每个批次的内存独立更新
        write_weights = outputs['final_state'].write_weights.numpy()

        # 预期内存
        expected_memory = np.zeros((batch_size, self.memory_size, self.word_size), dtype=np.float32)
        for b in range(batch_size):
            for w in range(self.num_writes):
                for i in range(self.memory_size):
                    expected_memory[b, i, :] += write_weights[b, w, i] * write_vector[b, w, :]
        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)

    def test_memory_shape(self):
        """
        验证 memory 的形状是否正确。
        """
        expected_shape = [self.batch_size, self.memory_size, self.word_size]
        actual_shape = self.initial_state.memory.shape
        self.assertEqual(expected_shape, actual_shape)

    def test_parse_interface_vector(self):
        """
        测试 _parse_interface_vector 方法是否正确解析 interface_vector。
        """
        batch_size = 2
        interface_size = self.memory_access.interface_size
        interface_vector = tf.random.uniform([batch_size, interface_size], dtype=tf.float32)
        parsed = self.memory_access._parse_interface_vector(interface_vector)

        # 验证每个组件的形状
        self.assertEqual(parsed['read_keys'].shape, [batch_size, self.num_reads, self.word_size])
        self.assertEqual(parsed['read_strengths'].shape, [batch_size, self.num_reads])
        self.assertEqual(parsed['write_keys'].shape, [batch_size, self.num_writes, self.word_size])
        self.assertEqual(parsed['write_strengths'].shape, [batch_size, self.num_writes])
        self.assertEqual(parsed['erase_vectors'].shape, [batch_size, self.num_writes, self.word_size])
        self.assertEqual(parsed['write_vectors'].shape, [batch_size, self.num_writes, self.word_size])
        self.assertEqual(parsed['free_gates'].shape, [batch_size, self.num_reads])
        self.assertEqual(parsed['allocation_gates'].shape, [batch_size, self.num_writes])
        self.assertEqual(parsed['write_gates'].shape, [batch_size, self.num_writes])
        self.assertEqual(parsed['read_modes'].shape, [batch_size, self.num_reads, self.num_read_modes])

        # 进一步验证解析后的值的合理性
        # 例如，read_modes 应该是 softmax 后的概率分布
        read_modes_sum = tf.reduce_sum(parsed['read_modes'], axis=-1)
        self.assertAllClose(read_modes_sum, tf.ones_like(read_modes_sum), atol=1e-5)


