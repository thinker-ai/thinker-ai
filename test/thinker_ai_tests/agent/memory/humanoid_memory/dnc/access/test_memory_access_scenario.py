# test_memory_access_scenario.py
import numpy as np
import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc.cache_manager import CacheManager
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess


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

    def _set_interface_layer_weights(self, erase_vector=1.0):
        """
        设置 interface_layer 的权重和偏置，使其输出可预测。
        具体来说，将控制器输出的特定部分映射到 interface_vector 中的对应部分。
        """
        interface_size = self.interface_size  # 应为93
        controller_output_size = self.controller_output_size  # 93

        # 创建一个全零的权重张量
        fixed_weights = np.zeros((controller_output_size, interface_size), dtype=np.float32)

        # 设置 controller_output[:,0:32] 映射到 interface_vector[:,0:32] (read_keys)
        fixed_weights[0:32, 0:32] = np.eye(32)

        # 设置 controller_output[:,32:34] 映射到 interface_vector[:,32:34] (read_strengths)
        fixed_weights[32:34, 32:34] = np.eye(2)

        # 设置 controller_output[:,34:50] 映射到 interface_vector[:,34:50] (write_keys)
        fixed_weights[34:50, 34:50] = np.eye(self.word_size)

        # 设置 controller_output[:,50] 映射到 interface_vector[:,50] (write_strengths)
        fixed_weights[50, 50] = 1.0

        # 设置 controller_output[:,51:67] 映射到 interface_vector[:,51:67] (erase_vectors)
        fixed_weights[51:67, 51:67] = erase_vector * np.eye(self.word_size)

        # 设置 controller_output[:,67:83] 映射到 interface_vector[:,67:83] (write_vectors)
        fixed_weights[67:83, 67:83] = np.eye(self.word_size)

        # 设置 controller_output[:,83:85] 映射到 interface_vector[:,83:85] (free_gates)
        fixed_weights[83:85, 83:85] = np.eye(self.num_reads)

        # 设置 allocation_gates 和 write_gates 为1.0，仅对应特定的 controller_output 单元
        fixed_weights[85, 85] = 1.0  # allocation_gates
        fixed_weights[86, 86] = 1.0  # write_gates

        # 设置 controller_output[:,87:93] 映射到 interface_vector[:,87:93] (read_modes)
        fixed_weights[87:93, 87:93] = np.eye(self.num_reads * self.num_read_modes)

        # 创建一个全零的偏置张量
        fixed_bias = np.zeros((interface_size,), dtype=np.float32)

        # 确保 interface_layer 被构建（即初始化权重参数）
        dummy_controller_output = tf.zeros([1, self.controller_output_size], dtype=tf.float32)
        self.memory_access.interface_layer(dummy_controller_output)

        # 设置 interface_layer 的权重和偏置
        self.memory_access.interface_layer.set_weights([fixed_weights, fixed_bias])

        # 验证 interface_layer 的输出是否正确
        test_write_vector = tf.ones([self.word_size], dtype=tf.float32)
        write_vector_batch = tf.tile(
            tf.expand_dims(test_write_vector, axis=0),
            [self.batch_size, 1]
        )  # [batch_size,16]
        test_controller_output = self._build_controller_output(
            write_vector=write_vector_batch,
            erase_logit=0.0,  # 因为 log(0.5/(1-0.5))=0.0
            write_strength=10
        )  # [batch_size,93]

        interface_vector = self.memory_access.interface_layer(test_controller_output)  # [batch_size, interface_size=93]

        tf.print("Test interface_vector[:,34:50]:", interface_vector[:, 34:50])  # 应为1.0
        tf.print("Test interface_vector[:,50]:", interface_vector[:, 50])  # 应为10.0
        tf.print("Test interface_vector[:,51:67]:", interface_vector[:, 51:67])  # 应为0.5（因为 erase_logit=0.0 对应 erase_vector=0.5）
        tf.print("Test interface_vector[:,67:83]:", interface_vector[:, 67:83])  # 应为1.0
        tf.print("Test interface_vector[:,85]:", interface_vector[:, 85])  # 应为10.0 (allocation_gates)
        tf.print("Test interface_vector[:,86]:", interface_vector[:, 86])  # 应为10.0 (write_gates)

    def _build_controller_output(self, write_vector, erase_logit, write_strength=10.0,
                                 read_content_keys=None, read_strengths=None, read_modes=None,
                                 write_content_keys=None, write_content_strengths=None,
                                 allocation_gate=None, write_gate=None,
                                 free_gates=None):
        """
        构建 controller_output，使 interface_layer 输出特定的 interface_vector。
        """
        batch_size = tf.shape(write_vector)[0]
        num_reads = self.num_reads
        num_writes = self.num_writes
        word_size = self.word_size
        num_read_modes = self.num_read_modes

        # 计算 erase_vector 的值
        erase_vector_value = tf.sigmoid(erase_logit).numpy()  # scalar

        # 初始化 controller_output 为全零
        controller_output = np.zeros([batch_size, self.controller_output_size],
                                     dtype=np.float32)  # [batch_size,93]

        # 设置 read_content_keys at indices 0:32
        if read_content_keys is not None:
            controller_output[:, 0:32] = read_content_keys.numpy().reshape(batch_size, -1)  # [batch_size, 32]
        else:
            controller_output[:, 0:32] = 0.0  # 默认值

        # 设置 read_strengths at indices 32:34
        if read_strengths is not None:
            controller_output[:, 32:34] = read_strengths.numpy()  # [batch_size, 2]
        else:
            controller_output[:, 32:34] = 1.0  # 默认值

        # 设置 write_content_keys at indices 34:50
        if write_content_keys is not None:
            controller_output[:, 34:50] = write_content_keys.numpy().reshape(batch_size, -1)  # [batch_size, 16]
        else:
            controller_output[:, 34:50] = 0.0  # 默认值

        # 设置 write_strength at index 50
        controller_output[:, 50] = write_strength  # [batch_size,]

        # 设置 erase_vectors at indices 51:67
        controller_output[:, 51:67] = erase_vector_value  # [batch_size, 16]

        # 设置 write_vectors at indices 67:83
        controller_output[:, 67:83] = write_vector.numpy()  # [batch_size, 16]

        # 设置 free_gates at indices 83:85
        if free_gates is not None:
            controller_output[:, 83:85] = free_gates.numpy()  # [batch_size, 2]
        else:
            controller_output[:, 83:85] = 1.0  # 默认值

        # 设置 allocation_gate at index 85
        if allocation_gate is not None:
            controller_output[:, 85] = tf.squeeze(allocation_gate, axis=-1).numpy()  # [batch_size,]
        else:
            controller_output[:, 85] = 10.0  # 默认值

        # 设置 write_gate at index 86
        if write_gate is not None:
            controller_output[:, 86] = tf.squeeze(write_gate, axis=-1).numpy()  # [batch_size,]
        else:
            controller_output[:, 86] = 10.0  # 默认值

        # 设置 read_modes at indices 87:93
        if read_modes is not None:
            controller_output[:, 87:93] = read_modes.numpy().reshape(batch_size, -1)  # [batch_size,6]
        else:
            # 默认设置为内容读取模式（第一个模式）
            read_modes_default = tf.one_hot(
                indices=tf.zeros([batch_size, num_reads], dtype=tf.int32),
                depth=num_read_modes,
                dtype=tf.float32
            ).numpy().reshape(batch_size, -1)  # [batch_size, 6]
            controller_output[:, 87:93] = read_modes_default

        # 转换回 Tensor
        controller_output = tf.convert_to_tensor(controller_output, dtype=tf.float32)  # [batch_size,93]

        return controller_output

    def test_full_erase(self):
        # 设置 erase_vector 为 1.0
        erase_vector = 1.0
        self._set_interface_layer_weights(erase_vector=erase_vector)

        # 定义写入向量
        write_vectors = tf.constant([
            [1.0] * self.word_size,
            [0.0] * self.word_size,
            [0.5] * self.word_size
        ], dtype=tf.float32)

        # 初始化预期内存状态
        expected_memory = tf.zeros([self.batch_size, self.memory_size, self.word_size], dtype=tf.float32)

        # 初始化变量，用于存储最后一次的 batched_write_vector
        last_batched_write_vector = None

        # 执行多个写入操作
        for i in range(3):
            # 获取单个写入向量并复制以匹配 batch_size
            single_write_vector = write_vectors[i]
            batched_write_vector = tf.tile(
                tf.expand_dims(single_write_vector, axis=0), [self.batch_size, 1]
            )

            # 在循环中保存最后一次的 batched_write_vector
            if i == 2:
                last_batched_write_vector = batched_write_vector  # 保存最后一次的写入向量

            # 构建 controller_output
            controller_output = self._build_controller_output(
                write_vector=batched_write_vector,
                erase_logit=10.0,  # 高正值对应 erase_vector ≈ 1.0
                write_strength=10.0,
                write_content_keys=batched_write_vector,
                write_content_strengths=tf.fill([self.batch_size, self.num_writes], 10.0),
                allocation_gate=tf.zeros([self.batch_size, self.num_writes]),
                write_gate=tf.ones([self.batch_size, self.num_writes])
            )

            # 重置 initial_state 以避免累积 usage
            self.initial_state = self.memory_access.get_initial_state(batch_size=self.batch_size)

            inputs = {
                'inputs': controller_output,
                'prev_state': self.initial_state
            }

            # 执行 MemoryAccess
            outputs = self.memory_access(inputs, training=False)
            final_state = outputs['final_state']

            # 从 final_state 中提取 write_weights
            actual_write_weight = final_state.write_weights  # [batch_size, num_writes, memory_size]
            if self.num_writes == 1:
                actual_write_weight = tf.squeeze(actual_write_weight, axis=1)  # [batch_size, memory_size]

            # 计算 erase 和 add 项，使用与 DefaultMemoryUpdater 相同的逻辑
            w = tf.expand_dims(actual_write_weight, -1)  # [batch_size, memory_size, 1]
            e = tf.reshape(erase_vector, [1, 1, 1])  # [1, 1, 1]
            e = tf.tile(e, [self.batch_size, self.memory_size, self.word_size])  # [batch_size, memory_size, word_size]
            erase_term = 1 - w * e  # [batch_size, memory_size, word_size]
            memory_erased = expected_memory * erase_term  # [batch_size, memory_size, word_size]

            # 计算 add 项
            a = tf.expand_dims(batched_write_vector, 1)  # [batch_size, 1, word_size]
            add_term = w * a  # [batch_size, memory_size, word_size]

            # 更新预期内存
            expected_memory = memory_erased + add_term  # [batch_size, memory_size, word_size]

        # 确保 last_batched_write_vector 已定义
        assert last_batched_write_vector is not None, "last_batched_write_vector 未定义"

        # 执行读取操作
        query_vector = write_vectors[-1]  # [word_size]

        # 获取 num_reads
        num_reads = self.memory_access.num_reads

        # 调整 batched_query_vector 的形状
        batched_query_vector = tf.broadcast_to(
            tf.reshape(query_vector, [1, 1, self.word_size]),
            [self.batch_size, num_reads, self.word_size]
        )  # [batch_size, num_reads, word_size]

        # 构建新的 controller_output，用于读取操作
        controller_output = self._build_controller_output(
            write_vector=tf.zeros([self.batch_size, self.word_size]),
            erase_logit=10.0,  # 保持与之前一致
            write_strength=0.0,  # 不进行写入
            read_content_keys=batched_query_vector,
            read_strengths=tf.fill([self.batch_size, self.num_reads], 10.0),
            read_modes=tf.one_hot(
                indices=tf.zeros([self.batch_size, self.num_reads], dtype=tf.int32),
                depth=self.num_read_modes,
                dtype=tf.float32
            ),
            allocation_gate=tf.zeros([self.batch_size, self.num_writes]),
            write_gate=tf.zeros([self.batch_size, self.num_writes])
        )

        # 更新 initial_state
        inputs = {
            'inputs': controller_output,
            'prev_state': final_state  # 使用最后的状态
        }

        # 执行 MemoryAccess
        outputs = self.memory_access(inputs, training=False)
        final_state = outputs['final_state']

        # 获取读取的内容
        read_words = outputs['read_words']  # [batch_size, num_reads, word_size]

        # 比较读取的内容与预期一致
        expected_read_words = tf.matmul(final_state.read_weights, expected_memory)

        # 检查读取的内容是否与预期一致
        tf.debugging.assert_near(read_words, expected_read_words, atol=1e-5)

    def test_no_erase(self):
        # 设置 erase_vector 为 0.0
        erase_vector = 0.0
        self._set_interface_layer_weights(erase_vector=erase_vector)

        # 定义写入向量
        write_vectors = tf.constant([
            [1.0] * self.word_size,
            [0.0] * self.word_size,
            [0.5] * self.word_size
        ], dtype=tf.float32)

        # 初始化预期内存状态
        expected_memory = tf.zeros([self.batch_size, self.memory_size, self.word_size], dtype=tf.float32)

        # 执行多个写入操作
        for i in range(3):
            # 获取单个写入向量并复制以匹配 batch_size
            single_write_vector = write_vectors[i]
            batched_write_vector = tf.tile(
                tf.expand_dims(single_write_vector, axis=0), [self.batch_size, 1]
            )

            # 构建 controller_output
            controller_output = self._build_controller_output(
                write_vector=batched_write_vector,
                erase_logit=-10.0,  # 低负值对应 erase_vector ≈ 0.0
                write_strength=10.0,
                write_content_keys=batched_write_vector,
                write_content_strengths=tf.fill([self.batch_size, self.num_writes], 10.0),
                allocation_gate=tf.zeros([self.batch_size, self.num_writes]),
                write_gate=tf.ones([self.batch_size, self.num_writes]),
                read_content_keys=tf.zeros([self.batch_size, self.num_reads, self.word_size]),
                read_strengths=tf.fill([self.batch_size, self.num_reads], 1.0),
                read_modes=tf.one_hot(
                    indices=tf.zeros([self.batch_size, self.num_reads], dtype=tf.int32),
                    depth=self.num_read_modes,
                    dtype=tf.float32
                )
            )

            inputs = {
                'inputs': controller_output,
                'prev_state': self.initial_state
            }

            # 执行 MemoryAccess
            outputs = self.memory_access(inputs, training=False)
            final_state = outputs['final_state']

            # 更新初始状态
            self.initial_state = final_state

            # 从 final_state 中提取 write_weights
            actual_write_weight = final_state.write_weights  # [batch_size, num_writes, memory_size]
            if self.num_writes == 1:
                actual_write_weight = tf.squeeze(actual_write_weight, axis=1)  # [batch_size, memory_size]

            # 计算 erase 和 add 项，使用与 DefaultMemoryUpdater 相同的逻辑
            w = tf.expand_dims(actual_write_weight, -1)  # [batch_size, memory_size, 1]
            e = tf.reshape(erase_vector, [1, 1, 1])  # [1, 1, 1]
            e = tf.tile(e, [self.batch_size, self.memory_size, self.word_size])  # [batch_size, memory_size, word_size]
            erase_term = 1 - w * e  # [batch_size, memory_size, word_size]
            memory_erased = expected_memory * erase_term  # [batch_size, memory_size, word_size]

            # 计算 add 项
            a = tf.expand_dims(batched_write_vector, 1)  # [batch_size, 1, word_size]
            add_term = w * a  # [batch_size, memory_size, word_size]

            # 更新预期内存
            expected_memory = memory_erased + add_term  # [batch_size, memory_size, word_size]

        # 执行读取操作
        query_vector = write_vectors[-1]  # [word_size]

        # 获取 num_reads
        num_reads = self.memory_access.num_reads

        # 调整 batched_query_vector 的形状
        batched_query_vector = tf.broadcast_to(
            tf.reshape(query_vector, [1, 1, self.word_size]),
            [self.batch_size, num_reads, self.word_size]
        )  # [batch_size, num_reads, word_size]

        # 构建新的 controller_output，用于读取操作
        controller_output = self._build_controller_output(
            write_vector=tf.zeros([self.batch_size, self.word_size]),
            erase_logit=-10.0,  # 保持与之前一致
            write_strength=0.0,  # 不进行写入
            read_content_keys=batched_query_vector,
            read_strengths=tf.fill([self.batch_size, self.num_reads], 10.0),
            read_modes=tf.one_hot(
                indices=tf.zeros([self.batch_size, self.num_reads], dtype=tf.int32),
                depth=self.num_read_modes,
                dtype=tf.float32
            ),
            allocation_gate=tf.zeros([self.batch_size, self.num_writes]),
            write_gate=tf.zeros([self.batch_size, self.num_writes])
        )

        # 更新 initial_state
        inputs = {
            'inputs': controller_output,
            'prev_state': final_state  # 使用最后的状态
        }

        # 执行 MemoryAccess
        outputs = self.memory_access(inputs, training=False)
        final_state = outputs['final_state']

        # 获取读取的内容
        read_words = outputs['read_words']  # [batch_size, num_reads, word_size]

        # 比较读取的内容与预期一致
        expected_read_words = tf.matmul(final_state.read_weights, expected_memory)

        # 检查读取的内容是否与预期一致
        tf.debugging.assert_near(read_words, expected_read_words, atol=1e-5)

    def test_partial_erase(self):
        """
        部分擦除 (erase_vector = 0.5)：
        验证内存被部分擦除，旧内容与新内容的混合。
        """
        # 设置 erase_vector 为 0.5
        erase_vector = 0.5
        self._set_interface_layer_weights(erase_vector=erase_vector)

        # 定义写入向量
        write_vectors = tf.constant([
            [1.0] * self.word_size,
            [0.0] * self.word_size,
            [0.5] * self.word_size
        ], dtype=tf.float32)  # [3, word_size]

        # 初始化预期内存状态
        expected_memory = tf.zeros([self.batch_size, self.memory_size, self.word_size],
                                   dtype=tf.float32)  # [batch_size, memory_size, word_size]

        # 执行多个写入操作
        for i in range(3):
            # 获取单个写入向量并复制以匹配 batch_size
            single_write_vector = write_vectors[i]  # [word_size]
            batched_write_vector = tf.tile(
                tf.expand_dims(single_write_vector, axis=0), [self.batch_size, 1]
            )  # [batch_size, word_size]

            # 构建 controller_output
            controller_output = self._build_controller_output(
                write_vector=batched_write_vector,
                erase_logit=0.0,  # erase_logit=0.0 对应 erase_vector=0.5
                write_strength=10.0,
                write_content_keys=batched_write_vector,
                write_content_strengths=tf.fill([self.batch_size, self.num_writes], 10.0),
                allocation_gate=tf.zeros([self.batch_size, self.num_writes]),
                write_gate=tf.ones([self.batch_size, self.num_writes]),
                read_content_keys=tf.zeros([self.batch_size, self.num_reads, self.word_size]),
                read_strengths=tf.fill([self.batch_size, self.num_reads], 1.0),
                read_modes=tf.one_hot(
                    indices=tf.zeros([self.batch_size, self.num_reads], dtype=tf.int32),
                    depth=self.num_read_modes,
                    dtype=tf.float32
                )
            )  # [batch_size, interface_size]

            inputs = {
                'inputs': controller_output,
                'prev_state': self.initial_state
            }

            # 执行 MemoryAccess
            outputs = self.memory_access(inputs, training=False)
            final_state = outputs['final_state']

            # 更新初始状态
            self.initial_state = final_state

            # 从 final_state 中提取 write_weights
            actual_write_weight = final_state.write_weights  # [batch_size, num_writes, memory_size]

            # 如果 num_writes=1，移除维度
            if self.num_writes == 1:
                actual_write_weight = tf.squeeze(actual_write_weight, axis=1)  # [batch_size, memory_size]

            # 计算预期内存更新
            w = tf.expand_dims(actual_write_weight, -1)  # [batch_size, memory_size, 1]
            e = tf.reshape(erase_vector, [1, 1, 1])  # [1, 1, 1]
            e = tf.tile(e, [self.batch_size, self.memory_size, self.word_size])  # [batch_size, memory_size, word_size]
            erase_term = 1 - w * e  # [batch_size, memory_size, word_size]
            memory_erased = expected_memory * erase_term  # [batch_size, memory_size, word_size]

            # 计算 add 项
            a = tf.expand_dims(batched_write_vector, 1)  # [batch_size, 1, word_size]
            add_term = w * a  # [batch_size, memory_size, word_size]

            # 更新预期内存
            expected_memory = memory_erased + add_term  # [batch_size, memory_size, word_size]

        # 执行读取操作
        query_vector = write_vectors[-1]  # [word_size]

        # 获取 num_reads
        num_reads = self.memory_access.num_reads

        # 调整 batched_query_vector 的形状
        batched_query_vector = tf.broadcast_to(
            tf.reshape(query_vector, [1, 1, self.word_size]),
            [self.batch_size, num_reads, self.word_size]
        )  # [batch_size, num_reads, word_size]

        # 构建新的 controller_output，用于读取操作
        controller_output = self._build_controller_output(
            write_vector=tf.zeros([self.batch_size, self.word_size]),
            erase_logit=0.0,  # 保持与之前一致
            write_strength=0.0,  # 不进行写入
            read_content_keys=batched_query_vector,
            read_strengths=tf.fill([self.batch_size, self.num_reads], 10.0),
            read_modes=tf.one_hot(
                indices=tf.zeros([self.batch_size, self.num_reads], dtype=tf.int32),
                depth=self.num_read_modes,
                dtype=tf.float32
            ),
            allocation_gate=tf.zeros([self.batch_size, self.num_writes]),
            write_gate=tf.zeros([self.batch_size, self.num_writes])
        )

        # 更新 initial_state
        inputs = {
            'inputs': controller_output,
            'prev_state': final_state  # 使用最后的状态
        }

        # 执行 MemoryAccess
        outputs = self.memory_access(inputs, training=False)
        final_state = outputs['final_state']

        # 获取读取的内容
        read_words = outputs['read_words']  # [batch_size, num_reads, word_size]

        # 比较读取的内容与预期一致
        expected_read_words = tf.matmul(final_state.read_weights, expected_memory)

        # 检查读取的内容是否与预期一致
        tf.debugging.assert_near(read_words, expected_read_words, atol=1e-5)

    def test_content_read_write(self):
        """
        内容读取与写入：
        验证写入的内容能够被准确读取。
        """
        # 设置擦除向量，根据测试需求选择擦除比例
        erase_vector = 0.5  # 示例：部分擦除
        self._set_interface_layer_weights(erase_vector=erase_vector)

        # 定义写入向量
        write_vectors = tf.constant([
            [1.0] * self.word_size,
            [0.0] * self.word_size,
            [0.5] * self.word_size
        ], dtype=tf.float32)  # [3, word_size]

        # 初始化预期内存状态
        expected_memory = tf.zeros([self.batch_size, self.memory_size, self.word_size], dtype=tf.float32)

        # 初始化 final_state
        final_state = self.initial_state

        # 执行多个写入操作
        for i in range(3):
            # 构建 controller_output
            write_vector_batch = tf.tile(
                tf.expand_dims(write_vectors[i], axis=0),
                [self.batch_size, 1]
            )  # [batch_size, word_size]
            controller_output = self._build_controller_output(
                write_vector=write_vector_batch,
                erase_logit=np.float32(np.log(erase_vector / (1 - erase_vector)))  # 应为0.0
            )  # 根据擦除比例计算 erase_logit

            # 重置 initial_state 以避免累积 usage
            self.initial_state = self.memory_access.get_initial_state(batch_size=self.batch_size)

            inputs = {
                'inputs': controller_output,
                'prev_state': self.initial_state
            }
            outputs = self.memory_access(inputs, training=False)
            final_state = outputs['final_state']

            # 从 final_state 中提取 write_weights
            actual_write_weight = final_state.write_weights  # [batch_size, num_writes, memory_size]
            if self.num_writes == 1:
                actual_write_weight = tf.squeeze(actual_write_weight, axis=1)  # [batch_size, memory_size]

            # 计算 erase 和 add 项，使用与 DefaultMemoryUpdater 相同的逻辑
            w = tf.expand_dims(actual_write_weight, -1)  # [batch_size, memory_size, 1]
            e = tf.reshape(erase_vector, [1, 1, 1])  # [1, 1, 1]
            e = tf.tile(e, [self.batch_size, self.memory_size, self.word_size])  # [batch_size, memory_size, word_size]
            erase_term = 1 - w * e  # [batch_size, memory_size, word_size]
            memory_erased = expected_memory * erase_term  # [batch_size, memory_size, word_size]

            # 计算 add 项
            a = tf.reshape(write_vectors[i], [1, 1, self.word_size])  # [1, 1, word_size]
            a = tf.tile(a, [self.batch_size, self.memory_size, 1])  # [batch_size, memory_size, word_size]
            add_term = w * a  # [batch_size, memory_size, word_size]

            # 更新预期内存
            expected_memory = memory_erased + add_term  # [batch_size, memory_size, word_size]

        # 执行读取操作
        # 定义查询向量，假设与最后一个写入向量相同
        query_vector = write_vectors[-1]  # [word_size]
        batched_query_vector = tf.tile(
            tf.expand_dims(query_vector, axis=0),
            [self.batch_size, 1]
        )  # [batch_size, word_size]

        # 重置 initial_state
        self.initial_state = self.memory_access.get_initial_state(batch_size=self.batch_size)

        # 执行历史记录查询
        related_records = self.memory_access.query_history(
            query_vector=batched_query_vector,
            top_k=1,
            read_strength=10.0
        )  # [batch_size, top_k, word_size]

        # 计算预期相关记录
        expected_memory_single_value = expected_memory[:, 0, :]  # [batch_size, word_size]
        expected_related_records = tf.expand_dims(expected_memory_single_value, axis=1)  # [batch_size,1,word_size]

        # 断言
        self.assertAllClose(related_records, expected_related_records, atol=1e-3)

    def test_user_memory_isolation(self):
        """
        测试不同用户的内存是否相互隔离。
        """
        # 用户1
        write_vector_user1 = tf.ones([1, self.word_size], dtype=tf.float32) * 1.0
        controller_output_user1 = self._build_controller_output(
            write_vector=write_vector_user1,
            erase_logit=0.0,
            write_strength=10.0,
            write_content_keys=write_vector_user1,
            write_content_strengths=tf.fill([1, self.num_writes], 10.0),
            allocation_gate=tf.zeros([1, self.num_writes]),
            write_gate=tf.ones([1, self.num_writes])
        )

        # 用户2
        write_vector_user2 = tf.ones([1, self.word_size], dtype=tf.float32) * 0.5
        controller_output_user2 = self._build_controller_output(
            write_vector=write_vector_user2,
            erase_logit=0.0,
            write_strength=10.0,
            write_content_keys=write_vector_user2,
            write_content_strengths=tf.fill([1, self.num_writes], 10.0),
            allocation_gate=tf.zeros([1, self.num_writes]),
            write_gate=tf.ones([1, self.num_writes])
        )

        # 合并两个用户的控制器输出
        controller_output = tf.concat([controller_output_user1, controller_output_user2], axis=0)

        # 执行 MemoryAccess
        inputs = {
            'inputs': controller_output,
            'prev_state': self.initial_state
        }

        output = self.memory_access(inputs, training=False)
        final_state = output['final_state']

        # 验证每个用户的记忆是隔离的
        memory_user1 = final_state.memory[0]  # [memory_size, word_size]
        memory_user2 = final_state.memory[1]  # [memory_size, word_size]

        # 使用 assertNotAllClose 并设置容差，确保内存内容不同
        self.assertNotAllClose(memory_user1, memory_user2, atol=1e-6)

        # 验证张量形状一致性
        self.assertEqual(memory_user1.shape, memory_user2.shape)

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
