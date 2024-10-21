# test_memory_access_scenario.py
from thinker_ai.agent.memory.humanoid_memory.dnc.cache_manager import CacheManager
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess
import numpy as np
import tensorflow as tf


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
        identity_matrix = np.eye(self.controller_output_size, self.interface_size, dtype=np.float32)
        # 如果 controller_output_size 和 interface_size 不相等，需要调整 identity_matrix 的形状
        if self.controller_output_size >= self.interface_size:
            identity_matrix = identity_matrix[:self.controller_output_size, :self.interface_size]
        else:
            identity_matrix = np.pad(identity_matrix, ((0, 0), (0, self.interface_size - self.controller_output_size)),
                                     'constant')

        # 设置 weights 和 biases
        weights = [identity_matrix, np.zeros(self.interface_size, dtype=np.float32)]

        # 手动构建 interface_layer
        # 这里的 input_shape 不包含批次维度，因此使用 (self.controller_output_size,)
        self.memory_access.interface_layer.build((None, self.controller_output_size))

        # 设置权重
        self.memory_access.interface_layer.set_weights(weights)

    def _build_controller_output(self, write_vector=None, erase_logit=None, write_strength=10.0,
                                 read_content_keys=None, read_strengths=None, read_modes=None,
                                 write_content_keys=None, write_content_strengths=None,
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

        # 设置 write_vectors 部分
        if write_vector is not None:
            write_vectors_start = indices['write_vectors_start']
            write_vectors_end = indices['write_vectors_end']
            write_vector_flat = write_vector.reshape(batch_size, -1)
            controller_output[:, write_vectors_start:write_vectors_end] = write_vector_flat

        # 设置 erase_vectors 部分
        if erase_logit is not None:
            erase_vectors_start = indices['erase_vectors_start']
            erase_vectors_end = indices['erase_vectors_end']
            erase_vector_flat = np.full((batch_size, (erase_vectors_end - erase_vectors_start)), erase_logit,
                                        dtype=np.float32)
            controller_output[:, erase_vectors_start:erase_vectors_end] = erase_vector_flat

        # 设置 write_strengths 部分
        if write_strength is not None:
            write_strengths_start = indices['write_strengths_start']
            write_strengths_end = indices['write_strengths_end']
            controller_output[:, write_strengths_start:write_strengths_end] = write_strength

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

        # 设置 read_content_keys 部分
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

    def test_user_memory_isolation(self):
        """
        测试不同用户的内存是否相互隔离。
        """

    def test_full_erase(self):
        """
        全部擦除 (erase_vector = 1.0)：
        验证内存被全部擦除，旧内容被新内容覆盖。
        """
        # 构建初始的 write_vector，用于写入初始内容
        initial_write_vector = np.random.rand(self.batch_size, self.num_writes, self.word_size).astype(np.float32)

        # 设置 erase_logit 为非常大的正值，使 erase_vector = sigmoid(large_value) ≈ 1.0
        erase_logit = 1000.0  # 非常大的值

        # 构建 controller_output，使得 interface_vector 包含所需的 write_vector 和 erase_vector
        controller_output = self._build_controller_output(
            write_vector=initial_write_vector,
            erase_logit=erase_logit,
            write_strength=10.0,
            allocation_gate=np.ones((self.batch_size, self.num_writes), dtype=np.float32),
            write_gate=np.ones((self.batch_size, self.num_writes), dtype=np.float32)
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

        # 由于 erase_vector ≈ 1.0，内存应该被完全擦除，然后写入新的 write_vector
        # 计算预期的内存
        expected_memory = np.zeros((self.batch_size, self.memory_size, self.word_size), dtype=np.float32)
        write_weights = outputs['write_weights'].numpy()  # [batch_size, num_writes, memory_size]
        for b in range(self.batch_size):
            for w in range(self.num_writes):
                for i in range(self.memory_size):
                    ww = write_weights[b, w, i]  # 写入权重
                    expected_memory[b, i, :] += ww * initial_write_vector[b, w, :]  # 写入新的内容

        # 验证 updated_memory 与 expected_memory 近似相等
        self.assertAllClose(updated_memory, expected_memory, atol=1e-5)

    def test_partial_erase(self):
        """
        部分擦除 (erase_vector = 0.5)：
        验证内存被部分擦除，旧内容与新内容的混合。
        """

    def test_no_erase(self):
        """
        不擦除 (erase_vector = 0)：
        验证内存不被擦除，旧内容与新内容相加。
        """


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
        self.assertEqual(parsed['read_modes'].shape, [batch_size, self.num_reads, 3])

    def test_memory_shape(self):
        """
        验证 memory 的形状是否正确。
        """
        expected_shape = [self.batch_size, self.memory_size, self.word_size]
        actual_shape = self.initial_state.memory.shape
        self.assertEqual(expected_shape, actual_shape)

    # def test_history_query_related_to_current_input(self):
    #     """
    #     测试与当前输入主题相关性的历史记录查询。
    #     """
    #     # 定义多个写入向量
    #     write_vectors = tf.constant([
    #         [1.0] * self.word_size,
    #         [0.0] * self.word_size,
    #         [0.5] * self.word_size
    #     ], dtype=tf.float32)  # [3, word_size]
    #
    #     # 定义写入操作的目标内存槽位（每个写入操作写入不同的槽位）
    #     write_slots = [0, 1, 2]
    #
    #     # Mock write_weight_calculator.compute 的 side_effect
    #     def mock_write_weights():
    #         for slot in write_slots:
    #             # 创建一个全零的权重矩阵
    #             write_weights = tf.zeros([self.batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
    #             # 将目标槽位的权重设置为1.0
    #             for b in range(self.batch_size):
    #                 for w in range(self.num_writes):
    #                     write_weights = tf.tensor_scatter_nd_update(
    #                         write_weights,
    #                         indices=[[b, w, slot]],
    #                         updates=[1.0]
    #                     )
    #             yield write_weights
    #
    #     write_weight_generator = mock_write_weights()
    #
    #     self.memory_access.write_weight_calculator.compute = mock.Mock(
    #         side_effect=lambda: next(write_weight_generator)
    #     )
    #
    #     # 执行多个写入操作
    #     for i in range(3):
    #         # 构建 controller_output
    #         controller_output = self._build_controller_output(write_vectors[i],
    #                                                           erase_logit=0.0)  # erase_logit=0.0 corresponds to erase_vector=0.5
    #
    #         inputs = {
    #             'inputs': controller_output,
    #             'prev_state': self.initial_state
    #         }
    #         output = self.memory_access(inputs, training=False)
    #         final_state = output['final_state']
    #
    #         # 更新初始状态
    #         self.initial_state = final_state
    #
    #     # 定义当前输入主题向量，与第一个写入向量相关
    #     current_input = tf.tile(tf.expand_dims(write_vectors[0], axis=0), [self.batch_size, 1])  # [2,16]
    #
    #     # 执行历史记录查询，检索与当前输入相关的 top_k 记录
    #     related_records = self.memory_access.query_history(
    #         query_vector=current_input,
    #         top_k=1,
    #         read_strength = 10.0
    #     )  # [batch_size, 1, word_size]
    #
    #     # 定义预期的相关记录，应为第一个写入向量 [1.0] * word_size
    #     expected_related_records = tf.expand_dims(write_vectors[0], axis=0)  # [1,16]
    #     expected_related_records = tf.tile(expected_related_records, [self.batch_size, 1])  # [2,16]
    #     expected_related_records = tf.expand_dims(expected_related_records, axis=1)  # [2,1,16]
    #
    #     # 打印相关记录以调试
    #     tf.print("Related Records (test_history_query_related_to_current_input):", related_records)
    #     tf.print("Expected Related Records (test_history_query_related_to_current_input):", expected_related_records)
    #
    #     # 断言
    #     self.assertAllClose(related_records, expected_related_records, atol=1e-6)
    #
    # def test_history_query_temporal_order(self):
    #     """
    #     测试相关历史记录查询结果按时序排列，最相关的记录在前。
    #     """
    #     # 定义多个写入向量，后写入的向量更相关
    #     write_vectors = tf.constant([
    #         [0.1] * self.word_size,
    #         [0.4] * self.word_size,
    #         [0.5] * self.word_size
    #     ], dtype=tf.float32)  # [3, word_size]
    #
    #     # 定义写入操作的目标内存槽位（每个写入操作写入不同的槽位）
    #     write_slots = [0, 1, 2]
    #
    #     # Mock write_weight_calculator.compute 的 side_effect
    #     def mock_write_weights():
    #         for slot in write_slots:
    #             # 创建一个全零的权重矩阵
    #             write_weights = tf.zeros([self.batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
    #             # 将目标槽位的权重设置为1.0
    #             for b in range(self.batch_size):
    #                 for w in range(self.num_writes):
    #                     write_weights = tf.tensor_scatter_nd_update(
    #                         write_weights,
    #                         indices=[[b, w, slot]],
    #                         updates=[1.0]
    #                     )
    #             yield write_weights
    #
    #     write_weight_generator = mock_write_weights()
    #
    #     self.memory_access.write_weight_calculator.compute = mock.Mock(
    #         side_effect=lambda: next(write_weight_generator)
    #     )
    #
    #     # 执行多个写入操作
    #     for i in range(3):
    #         # 构建 controller_output
    #         controller_output = self._build_controller_output(write_vectors[i],
    #                                                           erase_logit=0.0)  # erase_logit=0.0 corresponds to erase_vector=0.5
    #
    #         inputs = {
    #             'inputs': controller_output,
    #             'prev_state': self.initial_state
    #         }
    #         output = self.memory_access(inputs, training=False)
    #         final_state = output['final_state']
    #
    #         # 更新初始状态
    #         self.initial_state = final_state
    #
    #     # 定义当前输入主题向量，最相关于最后一个写入
    #     current_input = tf.tile(tf.expand_dims(write_vectors[2], axis=0), [self.batch_size, 1])  # [2,16]
    #
    #     # 执行历史记录查询，检索与当前输入相关的 top_k 记录
    #     related_records = self.memory_access.query_history(
    #         query_vector=current_input,
    #         top_k=3,
    #         read_strength = 10.0
    #     )  # [batch_size, 3, word_size]
    #
    #     # 定义预期的相关记录，按时序顺序 [0.5, 0.4, 0.1]
    #     expected_related_records = tf.constant([
    #         [
    #             [0.5] * self.word_size,  # 最后一个写入向量
    #             [0.4] * self.word_size,  # 第二个写入向量
    #             [0.1] * self.word_size  # 第一个写入向量
    #         ],
    #         [
    #             [0.5] * self.word_size,
    #             [0.4] * self.word_size,
    #             [0.1] * self.word_size
    #         ]
    #     ], dtype=tf.float32)  # [batch_size, 3, word_size]
    #
    #     # 打印相关记录以调试
    #     tf.print("Related Records (test_history_query_temporal_order):", related_records)
    #     tf.print("Expected Related Records (test_history_query_temporal_order):", expected_related_records)
    #
    #     # 断言
    #     self.assertAllClose(related_records, expected_related_records, atol=1e-6)


if __name__ == '__main__':
    tf.test.main()
