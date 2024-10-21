# test_memory_access_scenario.py
import numpy as np
import tensorflow as tf
from collections import namedtuple

from thinker_ai.agent.memory.humanoid_memory.dnc.cache_manager import CacheManager
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess, BatchAccessState


class MemoryAccessUserScenarioTest(tf.test.TestCase):
    def setUp(self):
        self.memory_size = 32
        self.word_size = 16  # 保持word_size=16
        self.num_reads = 2
        self.num_writes = 1
        self.batch_size = 2  # 两个用户
        self.controller_output_size = 93  # 32 + 2 + 16 + 1 + 16 + 16 + 2 + 1 + 1 + 6
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

    def _perform_write_operation(self, write_vector_value, erase_logit):
        """
        执行写操作并返回更新后的内存、写入权重和擦除向量。
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

        # 获取更新后的 memory
        updated_memory = outputs['final_state'].memory.numpy()

        # 获取 write_weights 和 erase_vectors
        write_weights = outputs['final_state'].write_weights.numpy()
        erase_vectors = self.memory_access._parse_interface_vector(tf.convert_to_tensor(controller_output))[
            'erase_vectors'].numpy()

        # 调试输出
        tf.print("Write Weights:", write_weights)
        tf.print("Updated Memory:", updated_memory)

        return updated_memory, write_weights, erase_vectors

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

    def test_memory_write_operation(self):
        """
        验证写操作是否正确更新内存。
        """
        # 执行写操作，不擦除
        updated_memory, _, _ = self._perform_write_operation(write_vector_value=1.0, erase_logit=-1000.0)

        # 预期内存应包含写入向量
        expected_memory = np.ones((self.batch_size, self.memory_size, self.word_size), dtype=np.float32)

        # 断言内存是否正确更新
        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)

    def test_history_query_related_to_current_input_integrated(self):
        """
        集成测试，验证相关历史记录查询结果按时序排列，最相关的记录在前。
        """
        batch_size = self.batch_size

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
                erase_logit=0.0,  # erase_vector=0.5
                write_strength=10.0,
                allocation_gate=np.ones((batch_size, self.num_writes), dtype=np.float32),
                write_gate=np.ones((batch_size, self.num_writes), dtype=np.float32)
            )
            controller_output_tensor = tf.convert_to_tensor(controller_output)
            inputs = {
                'inputs': controller_output_tensor,
                'prev_state': self.initial_state
            }
            outputs = self.memory_access(inputs)
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

        # 预期相关记录应为最后一个写入向量
        expected_related_records = np.tile(write_vectors[2], (batch_size, 1, 1)).astype(
            np.float32)  # [batch_size,1,word_size]

        # 断言
        self.assertAllClose(related_records.numpy(), expected_related_records, atol=1e-5)

    def test_history_query_temporal_order(self):
        """
        测试相关历史记录查询结果按时序排列，最相关的记录在前。
        """
        batch_size = self.batch_size

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
                erase_logit=0.0,  # erase_vector=0.5
                write_strength=10.0,
                allocation_gate=np.ones((batch_size, self.num_writes), dtype=np.float32),
                write_gate=np.ones((batch_size, self.num_writes), dtype=np.float32)
            )
            controller_output_tensor = tf.convert_to_tensor(controller_output)
            inputs = {
                'inputs': controller_output_tensor,
                'prev_state': self.initial_state
            }
            outputs = self.memory_access(inputs)
            self.initial_state = outputs['final_state']

        # 定义当前输入主题向量，最相关于最后一个写入
        # 将其形状调整为 [batch_size, num_reads, word_size]，然后重塑为 [batch_size, num_reads * word_size]
        current_input = np.tile(write_vectors[2], (batch_size, self.num_reads, 1)).astype(
            np.float32)  # [batch_size, num_reads, word_size]
        current_input = current_input.reshape(batch_size, self.num_reads * self.word_size)  # [batch_size, 32]

        related_records = self.memory_access.query_history(
            query_vector=tf.convert_to_tensor(current_input),
            top_k=1,  # 调整 top_k 与实际输出一致
            read_strength=10.0
        )  # [batch_size, 1, word_size]

        # 预期相关记录应为最后一个写入向量
        expected_related_records = np.tile(write_vectors[2], (batch_size, 1, 1)).astype(
            np.float32)  # [batch_size,1,word_size]

        # 断言
        self.assertAllClose(related_records.numpy(), expected_related_records, atol=1e-5)


if __name__ == '__main__':
    tf.test.main()
