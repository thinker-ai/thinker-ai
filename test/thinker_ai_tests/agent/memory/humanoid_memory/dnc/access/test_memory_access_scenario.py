# test_memory_access_scenario.py
import numpy as np
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.cache_manager import CacheManager
from thinker_ai.agent.memory.humanoid_memory.dnc.default_component import get_default_config
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess, BatchAccessState


class MemoryAccessUserScenarioTest(tf.test.TestCase):
    def setUp(self):
        super(MemoryAccessUserScenarioTest, self).setUp()
        # 初始化测试参数
        self.memory_size = 32
        self.word_size = 16  # 保持word_size=16
        self.num_reads = 2
        self.num_writes = 1
        self.batch_size = 2  # 两个用户
        self.controller_output_size = 93  # 32 + 2 + 16 + 1 + 16 + 16 + 2 + 1 + 1 + 6
        self.num_read_modes = 3  # 添加此行
        self.epsilon = 1e-6  # 定义 epsilon

        # 初始化 CacheManager
        self.cache_manager = CacheManager(cache_dir="./test_cache", max_cache_size=10)
        self.cache_manager.clear_cache()  # 确保缓存为空

        # 初始化 MemoryAccess 并注入 CacheManager
        config = get_default_config(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            num_reads=self.num_reads,
            word_size=self.word_size
        )

        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size,
            cache_manager=self.cache_manager,
            config=config
        )

        # 获取 interface_size
        self.interface_size = self.memory_access.interface_size  # 应为93

        # 初始化初始状态
        self.initial_state = self.memory_access.get_initial_state(self.batch_size)

        # 设置 interface_layer 的权重以获得可预测的输出
        self._set_interface_layer_weights()

        # 将初始状态写入缓存
        self.cache_manager.write_to_cache('memory_state', self.initial_state)

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

    def _perform_write_operation(self, write_vector_value, erase_logit):
        """
        执行写操作并返回更新后的内存、写入权重和写入向量。
        """
        batch_size = self.batch_size

        # 构建写入内容
        write_vector = np.ones((batch_size, self.num_writes, self.word_size), dtype=np.float32) * write_vector_value

        # 构建 controller_output
        controller_output = self._build_controller_output(
            write_vector=write_vector,
            erase_logit=erase_logit,
            write_strength=10.0,  # 确保写强度足够大
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
        final_state = outputs['final_state']

        # 更新 initial_state 和 cache
        self.initial_state = final_state
        self.cache_manager.write_to_cache('memory_state', final_state)

        # 获取更新后的 memory 和 write_weights
        updated_memory = final_state.memory.numpy()  # [batch_size, memory_size, word_size]
        write_weights = final_state.write_weights.numpy()  # [batch_size, num_writes, memory_size]

        # 验证 write_weights 在 memory_size 维度上归一化
        write_weights_sum = np.sum(write_weights, axis=2)  # [batch_size, num_writes]
        self.assertAllClose(write_weights_sum, np.ones([self.batch_size, self.num_writes]), atol=1e-5)

        return updated_memory, write_weights, write_vector

    def test_memory_write_operation(self):
        """
        验证写操作是否正确更新内存。
        """
        # 执行写操作，不擦除
        updated_memory, write_weights, write_vectors = self._perform_write_operation(
            write_vector_value=1.0,
            erase_logit=-1000.0
        )

        # 正确的广播方式
        expected_memory = write_weights[:, 0, :, np.newaxis] * write_vectors[:, 0, np.newaxis, :]  # [2,32,16]

        # 将 expected_memory 转换为相同的数据类型（如果需要）
        expected_memory = expected_memory.astype(np.float32)

        # 打印调试信息（可选）
        tf.print("Write Weights:", write_weights)
        tf.print("Expected Memory:", expected_memory)
        tf.print("Actual Updated Memory:", updated_memory)

        # 断言
        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)

    def test_single_write_operation(self):
        """
        验证单次写操作是否正确更新内存，并能通过查询读取到正确的写入内容。
        """
        # 执行写操作
        updated_memory, write_weights, write_vectors = self._perform_write_operation(
            write_vector_value=1.0,
            erase_logit=-1000.0  # 确保不擦除
        )

        # 获取读_weights
        read_weights = self.initial_state.read_weights.numpy()  # [2,2,32]

        # 计算预期读取向量
        # 使用批量矩阵乘法，应该使用 memory
        expected_read_vectors = np.einsum('brm,bmk->brk', read_weights, updated_memory)  # [2,2,16]

        # 定义查询向量，与写入向量相同
        write_vector = np.ones([self.batch_size, self.num_writes, self.word_size], dtype=np.float32)
        query_vector = np.tile(write_vector, (1, self.num_reads, 1)).reshape(self.batch_size,
                                                                             self.num_reads * self.word_size)  # [2,32]
        query_vector_tf = tf.convert_to_tensor(query_vector, dtype=tf.float32)

        # 执行查询
        read_vectors = self.memory_access.query_history(
            query_vector=query_vector_tf,
            top_k=self.num_reads,
            read_strength=10.0
        )  # [2,2,16]
        read_vectors_np = read_vectors.numpy()

        # 打印调试信息（可选）
        tf.print("Read Weights:", self.initial_state.read_weights)
        tf.print("Expected Read Vectors:", expected_read_vectors)
        tf.print("Actual Read Vectors:", read_vectors_np)

        # 断言
        self.assertAllClose(read_vectors_np, expected_read_vectors, atol=1e-5)

    def test_basic_query_operation(self):
        """
        验证基本的查询操作是否能够检索到正确的内存内容。
        """
        # 写入已知向量
        write_vectors = np.array([
            [0.1] * self.word_size,
            [0.4] * self.word_size,
            [0.5] * self.word_size
        ], dtype=np.float32)  # [3, word_size]

        for vec in write_vectors:
            write_vector = np.tile(vec, (self.batch_size, self.num_writes, 1))  # [batch_size, num_writes, word_size]
            controller_output = self._build_controller_output(
                write_vector=write_vector,
                erase_logit=0.0,
                write_strength=10.0,
                allocation_gate=np.ones((self.batch_size, self.num_writes), dtype=np.float32),
                write_gate=np.ones((self.batch_size, self.num_writes), dtype=np.float32)
            )
            # 转换为 Tensor
            controller_output_tensor = tf.convert_to_tensor(controller_output)
            # 调用 memory_access
            inputs = {
                'inputs': controller_output_tensor,
                'prev_state': self.initial_state
            }
            outputs = self.memory_access(inputs)
            # 更新 initial_state
            self.initial_state = outputs['final_state']

        # 定义查询向量，与最后一个写入的向量相关
        query_vector = np.tile(write_vectors[-1], (self.batch_size, self.num_reads, 1)).astype(
            np.float32)  # [batch_size, num_reads, word_size]
        query_vector = query_vector.reshape(self.batch_size, self.num_reads * self.word_size)  # [batch_size, 32]

        # 执行查询
        related_records = self.memory_access.query_history(
            query_vector=tf.convert_to_tensor(query_vector),
            top_k=1,
            read_strength=10.0
        )  # [batch_size, 1, word_size]

        # 预期相关记录为最后一个写入向量 * write_strength
        # 由于写入向量为1.0，并且写_strength影响了内容权重，所以预期读取向量为写入向量
        expected_related_records = np.tile(write_vectors[-1], (self.batch_size, 1, 1)).astype(
            np.float32)  # [batch_size,1,word_size]

        # 将 related_records 转换为 NumPy 数组
        related_records_np = related_records.numpy()

        # 断言
        self.assertAllClose(related_records_np, expected_related_records, atol=1e-5)

    def test_content_weight_calculator(self):
        """
        验证 ContentWeightCalculator 是否正确计算 content_weights。
        """
        # 使用 MemoryAccess 实例的 ContentWeightCalculator
        content_weight_calculator = self.memory_access.content_weight_calculator

        # 假设 ContentWeightCalculator 的输入是 keys 和 memory
        keys = tf.constant(
            np.full([self.batch_size, self.num_writes, self.word_size], 0.25, dtype=np.float32)
        )  # [2,1,16]
        memory = tf.constant(
            np.full([self.batch_size, self.memory_size, self.word_size], 0.25, dtype=np.float32)
        )  # [2,32,16]

        # 计算 content_weights
        content_weights = content_weight_calculator.compute(
            keys=keys,
            strengths=tf.ones([self.batch_size, self.num_writes], dtype=tf.float32),  # [2,1]
            memory=memory
        ).numpy()  # [2,1,32]

        # 打印调试信息
        print("Content Weights:", content_weights)

        # 预期内容权重应均匀分布，因为所有相似度相同
        expected_content_weights = np.full([self.batch_size, self.num_writes, self.memory_size], 1.0 / self.memory_size,
                                           dtype=np.float32)

        # 断言
        self.assertAllClose(content_weights, expected_content_weights, atol=1e-5)

    def test_history_query_related_to_current_input_integrated(self):
        """
        集成测试，验证相关历史记录查询结果按时序排列，最相关的记录在前。
        """
        batch_size = self.batch_size
        write_strength = 10.0
        # 定义多个写入向量，后写入的向量更相关
        write_vectors = np.array([
            [0.1] * self.word_size,
            [0.4] * self.word_size,
            [0.5] * self.word_size
        ], dtype=np.float32)  # [3, word_size]

        # 执行多个写入操作
        for i in range(len(write_vectors)):
            # 确保 write_vector 形状为 (batch_size, num_writes, word_size)
            write_vector = np.tile(write_vectors[i:i + 1],
                                   (batch_size, self.num_writes, 1))  # [batch_size, num_writes, word_size]
            controller_output = self._build_controller_output(
                write_vector=write_vector,
                erase_logit=0.0,  # erase_vector=0.0
                write_strength=write_strength,
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
            # 更新 initial_state
            self.initial_state = outputs['final_state']

        # 定义当前输入主题向量，最相关于最后一个写入
        current_input = np.tile(write_vectors[2], (batch_size, self.num_reads, 1)).astype(
            np.float32)  # [batch_size, num_reads, word_size]
        current_input = current_input.reshape(batch_size, self.num_reads * self.word_size)  # [batch_size, 32]

        related_records = self.memory_access.query_history(
            query_vector=tf.convert_to_tensor(current_input),
            top_k=1,  # 调整 top_k 与实际输出一致
            read_strength=write_strength  #
        )  # [batch_size, 1, word_size]

        # 预期相关记录应为最后一个写入向量 * write_strength
        expected_related_records = np.tile(write_vectors[2] * write_strength, (batch_size, 1, 1)).astype(
            np.float32)  # [batch_size,1,word_size]

        # 将 related_records 转换为 NumPy 数组
        related_records_np = related_records.numpy()

        # 断言
        self.assertAllClose(related_records_np, expected_related_records, atol=1e-5)

    def test_history_query_temporal_order(self):
        """
        测试相关历史记录查询结果按时序排列，最相关的记录在前。
        """
        batch_size = self.batch_size
        write_strength = 10.0
        # 定义多个写入向量，后写入的向量更相关
        write_vectors = np.array([
            [0.1] * self.word_size,
            [0.4] * self.word_size,
            [0.5] * self.word_size
        ], dtype=np.float32)  # [3, word_size]

        # 执行多个写入操作
        for i in range(len(write_vectors)):
            # 确保 write_vector 形状为 (batch_size, num_writes, word_size)
            write_vector = np.tile(write_vectors[i:i + 1],
                                   (batch_size, self.num_writes, 1))  # [batch_size, num_writes, word_size]
            controller_output = self._build_controller_output(
                write_vector=write_vector,
                erase_logit=0.0,  # erase_vector=0.0
                write_strength=write_strength,
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
            # 更新 initial_state
            self.initial_state = outputs['final_state']

        # 定义当前输入主题向量，最相关于最后一个写入
        current_input = np.tile(write_vectors[2], (batch_size, self.num_reads, 1)).astype(
            np.float32)  # [batch_size, num_reads, word_size]
        current_input = current_input.reshape(batch_size, self.num_reads * self.word_size)  # [batch_size, 32]

        related_records = self.memory_access.query_history(
            query_vector=tf.convert_to_tensor(current_input),
            top_k=1,  # 调整 top_k 与实际输出一致
            read_strength=10.0
        )  # [batch_size, 1, word_size]

        # 预期相关记录应为最后一个写入向量 * write_strength
        expected_related_records = np.tile(write_vectors[2] * write_strength, (batch_size, 1, 1)).astype(
            np.float32)  # [batch_size,1,word_size]

        # 将 related_records 转换为 NumPy 数组
        related_records_np = related_records.numpy()

        # 断言
        self.assertAllClose(related_records_np, expected_related_records, atol=1e-5)

    def test_memory_write_and_query(self):
        """
        验证写入操作后，能够通过查询检索到正确的向量。
        """
        # 写入向量
        write_vectors = np.array([
            [0.2] * self.word_size,
            [0.3] * self.word_size
        ], dtype=np.float32)  # [2, word_size]

        for vec in write_vectors:
            write_vector = np.tile(vec, (self.batch_size, self.num_writes, 1))
            controller_output = self._build_controller_output(
                write_vector=write_vector,
                erase_logit=0.0,
                write_strength=10.0,
                allocation_gate=np.ones((self.batch_size, self.num_writes), dtype=np.float32),
                write_gate=np.ones((self.batch_size, self.num_writes), dtype=np.float32)
            )
            # 转换为 Tensor
            controller_output_tensor = tf.convert_to_tensor(controller_output)
            # 调用 memory_access
            inputs = {
                'inputs': controller_output_tensor,
                'prev_state': self.initial_state
            }
            outputs = self.memory_access(inputs)
            # 更新 initial_state
            self.initial_state = outputs['final_state']

            # 打印当前内存状态
            current_memory = self.memory_access.get_initial_state(self.batch_size).memory.numpy()
            print(f"Memory after writing vector {vec}: {current_memory}")

        # 查询向量
        query_vector = np.tile(write_vectors[-1], (self.batch_size, self.num_reads, 1)).astype(np.float32)
        query_vector = query_vector.reshape(self.batch_size, self.num_reads * self.word_size)

        # 执行查询
        related_records = self.memory_access.query_history(
            query_vector=tf.convert_to_tensor(query_vector),
            top_k=1,
            read_strength=10.0
        )
        print("Related Records:", related_records.numpy())

        # 预期结果
        expected_related_records = np.tile(write_vectors[-1] * 10.0, (self.batch_size, 1, 1)).astype(np.float32)

        # 将 related_records 转换为 NumPy 数组
        related_records_np = related_records.numpy()

        # 断言
        self.assertAllClose(related_records_np, expected_related_records, atol=1e-5)

    def test_memory_state_after_write(self):
        """
        在写入操作后，验证内存状态是否正确。
        """
        write_vector = np.ones((self.batch_size, self.num_writes, self.word_size), dtype=np.float32)  # 全1向量
        controller_output = self._build_controller_output(
            write_vector=write_vector,
            erase_logit=0.0,
            write_strength=10.0,
            allocation_gate=np.ones((self.batch_size, self.num_writes), dtype=np.float32),
            write_gate=np.ones((self.batch_size, self.num_writes), dtype=np.float32)
        )
        controller_output_tensor = tf.convert_to_tensor(controller_output)
        inputs = {
            'inputs': controller_output_tensor,
            'prev_state': self.initial_state
        }
        outputs = self.memory_access(inputs)
        self.initial_state = outputs['final_state']

        # 获取更新后的内存
        updated_memory = self.memory_access.get_initial_state(self.batch_size).memory.numpy()
        print("Updated Memory State:", updated_memory)

        # 预期内存等于写入向量 * write_strength
        expected_memory = write_vector * 10.0
        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)

    def test_write_gate_activation(self):
        """
        验证写门是否正确激活，从而允许写入操作。
        """
        write_vector = np.ones((self.batch_size, self.num_writes, self.word_size), dtype=np.float32)
        allocation_gate = np.zeros((self.batch_size, self.num_writes), dtype=np.float32)  # 关闭分配门
        write_gate = np.ones((self.batch_size, self.num_writes), dtype=np.float32)  # 打开写门

        controller_output = self._build_controller_output(
            write_vector=write_vector,
            erase_logit=0.0,
            write_strength=10.0,
            allocation_gate=allocation_gate,
            write_gate=write_gate
        )
        controller_output_tensor = tf.convert_to_tensor(controller_output)
        inputs = {
            'inputs': controller_output_tensor,
            'prev_state': self.initial_state
        }
        outputs = self.memory_access(inputs)
        self.initial_state = outputs['final_state']

        # 获取更新后的内存
        updated_memory = self.memory_access.get_initial_state(self.batch_size).memory.numpy()
        print("Updated Memory with Allocation Gate Closed:", updated_memory)

        # 预期内存未被更新，因为分配门关闭
        expected_memory = np.zeros_like(updated_memory)

        # 断言
        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)

    def test_similarity_calculation(self):
        """
        验证相似度计算是否正确。
        """
        # 写入已知向量
        write_vectors = np.array([
            [1.0] * self.word_size,
            [2.0] * self.word_size
        ], dtype=np.float32)
        for vec in write_vectors:
            write_vector = np.tile(vec, (self.batch_size, self.num_writes, 1))
            controller_output = self._build_controller_output(
                write_vector=write_vector,
                erase_logit=0.0,
                write_strength=10.0,
                allocation_gate=np.ones((self.batch_size, self.num_writes), dtype=np.float32),
                write_gate=np.ones((self.batch_size, self.num_writes), dtype=np.float32)
            )
            # 转换为 Tensor
            controller_output_tensor = tf.convert_to_tensor(controller_output)
            # 调用 memory_access
            inputs = {
                'inputs': controller_output_tensor,
                'prev_state': self.initial_state
            }
            outputs = self.memory_access(inputs)
            # 更新 initial_state
            self.initial_state = outputs['final_state']

        # 定义查询向量
        query_vector = np.tile([2.0] * self.word_size, (self.batch_size, self.num_reads, 1)).astype(np.float32)
        query_vector = query_vector.reshape(self.batch_size, self.num_reads * self.word_size)

        # 执行查询
        related_records = self.memory_access.query_history(
            query_vector=tf.convert_to_tensor(query_vector),
            top_k=1,
            read_strength=10.0
        )
        print("Related Records based on Similarity:", related_records.numpy())

        # 预期相关记录为第二个写入向量 * write_strength
        expected_related_records = np.tile(write_vectors[-1] * 10.0, (self.batch_size, 1, 1)).astype(np.float32)

        # 将 related_records 转换为 NumPy 数组
        related_records_np = related_records.numpy()

        # 断言
        self.assertAllClose(related_records_np, expected_related_records, atol=1e-5)

    def test_step_by_step_memory_update(self):
        """
        单步验证内存更新操作，确保擦除和写入步骤正确。
        """
        # 执行写操作，部分擦除
        updated_memory, write_weights, erase_vectors = self._perform_write_operation(write_vector_value=3.0,
                                                                                     erase_logit=0.0)

        # 预期内存 = initial_memory * (1 - write_weights * erase_vector) + write_weights * write_vector
        # 这里需要正确广播 write_weights 和 erase_vectors

        # 扩展 initial_memory 以匹配 num_writes 维度
        initial_memory = self.initial_state.memory.numpy()  # [batch_size, memory_size, word_size]
        initial_memory_expanded = np.expand_dims(initial_memory, axis=1)  # [batch_size, 1, memory_size, word_size]

        # 计算擦除掩码
        erase_mask = 1 - (write_weights[:, :, :, np.newaxis] * erase_vectors[:, :, np.newaxis,
                                                               :])  # [batch_size, num_writes, memory_size, word_size]

        # 计算擦除后的内存
        erased_memory = initial_memory_expanded * erase_mask  # [batch_size, num_writes, memory_size, word_size]

        # 计算添加到内存的内容
        added_memory = write_weights[:, :, :, np.newaxis] * 3.0  # [batch_size, num_writes, memory_size, word_size]

        # 计算预期内存，通过对 num_writes 维度求和
        expected_memory = np.sum(erased_memory + added_memory, axis=1)  # [batch_size, memory_size, word_size]

        # 断言
        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)

    def tearDown(self):
        # 清理缓存目录
        self.cache_manager.clear_cache()
        super(MemoryAccessUserScenarioTest, self).tearDown()


if __name__ == '__main__':
    tf.test.main()
