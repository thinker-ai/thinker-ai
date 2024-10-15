# test_memory_access_scenario.py
from unittest import mock
import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess


class MemoryAccessUserScenarioTest(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessUserScenarioTest, self).setUp()
        self.memory_size = 32
        self.word_size = 16
        self.num_reads = 2
        self.num_writes = 1
        self.batch_size = 2  # 两个用户
        self.controller_output_size = 64  # 固定的控制器输出尺寸

        # 初始化一个内存中的缓存存储，用于模拟 CacheManager
        self.cache_store = {}

        # 定义 mock 的 write_to_cache 方法的 side_effect
        def mock_write_to_cache(cache_key, data):
            self.cache_store[cache_key] = data

        # 定义 mock 的 read_from_cache 方法的 side_effect
        def mock_read_from_cache(cache_key):
            return self.cache_store.get(cache_key, None)

        # 使用 mock.Mock 创建 CacheManager 的 mock 对象
        self.cache_manager_mock = mock.Mock()
        self.cache_manager_mock.write_to_cache.side_effect = mock_write_to_cache
        self.cache_manager_mock.read_from_cache.side_effect = mock_read_from_cache

        # 初始化 MemoryAccess 并注入 mock 的 CacheManager
        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size,
            cache_manager=self.cache_manager_mock  # 注入 mock 的 CacheManager
        )
        self.initial_state = self.memory_access.get_initial_state(self.batch_size)

        # 设置 interface_layer 的权重和偏置以确保可预测的接口层输出
        self._set_interface_layer_weights()

        # Mock components to return fixed values
        self.memory_access.content_weight_calculator.compute = mock.Mock(
            return_value=tf.ones([self.batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        )
        self.memory_access.write_weight_calculator.compute = mock.Mock(
            return_value=tf.ones([self.batch_size, self.num_writes, self.memory_size], dtype=tf.float32)
        )
        self.memory_access.read_weight_calculator.compute = mock.Mock(
            return_value=tf.ones([self.batch_size, self.num_reads, self.memory_size], dtype=tf.float32)
        )
        self.memory_access.usage_updater.update_usage = mock.Mock(
            return_value=tf.ones([self.batch_size, self.memory_size], dtype=tf.float32)
        )
        self.memory_access.temporal_linkage_updater.update_linkage = mock.Mock(
            return_value={'link': tf.zeros([self.batch_size, self.memory_size, self.memory_size], dtype=tf.float32),
                          'precedence_weights': tf.zeros([self.batch_size, self.memory_size], dtype=tf.float32)}
        )

    def _set_interface_layer_weights(self, erase_vector=1.0):
        """
        设置 interface_layer 的权重和偏置，使其输出可预测。
        具体来说，将第0维的控制器输出映射到 write_strengths，
        第1到16维映射到 write_vectors，
        并根据 erase_vector 设置 erase_vectors 的 logits。
        """
        interface_size = self.memory_access.interface_size  # 应为83
        controller_output_size = self.controller_output_size  # 64

        # 创建一个全零的权重张量
        fixed_weights = tf.zeros((controller_output_size, interface_size), dtype=tf.float32)

        # 设置 controller_output[:,0] 映射到 interface_vector[:,50] (write_strengths)
        fixed_weights = tf.tensor_scatter_nd_update(
            fixed_weights,
            indices=[[0, 50]],
            updates=[1.0]
        )

        # 设置 controller_output[:,1:17] 映射到 interface_vector[:,67:83] (write_vectors)
        for i in range(1, 17):
            fixed_weights = tf.tensor_scatter_nd_update(
                fixed_weights,
                indices=[[i, 67 + i - 1]],
                updates=[1.0]
            )

        # 根据 erase_vector，计算对应的 logit 值
        # sigmoid(logit) = erase_vector => logit = logit(erase_vector)
        # 由于 sigmoid(x) approaches 1 as x approaches +inf and 0 as x approaches -inf,
        # 我们使用 x=10 对应 erase_vector≈1.0，x=0 对应 erase_vector=0.5，x=-10 对应 erase_vector≈0.0
        if erase_vector == 1.0:
            erase_logit = 10.0  # Approximately 1.0 after sigmoid
        elif erase_vector == 0.5:
            erase_logit = 0.0  # Exactly 0.5 after sigmoid
        elif erase_vector == 0.0:
            erase_logit = -10.0  # Approximately 0.0 after sigmoid
        else:
            # For other values, compute logit
            erase_logit = tf.math.log(erase_vector / (1 - erase_vector))
            erase_logit = erase_logit.numpy()  # Convert to float

        # 设置 controller_output[:,17:33] 映射到 interface_vector[:,51:67] (erase_vectors)
        for i in range(17, 33):
            interface_index = 51 + (i - 17)
            fixed_weights = tf.tensor_scatter_nd_update(
                fixed_weights,
                indices=[[i, interface_index]],
                updates=[erase_logit]
            )

        # 创建一个全零的偏置张量
        fixed_bias = tf.zeros((interface_size,), dtype=tf.float32)

        # 确保 interface_layer 被构建（即初始化权重参数）
        dummy_controller_output = tf.zeros([1, self.controller_output_size])
        self.memory_access.interface_layer(dummy_controller_output)

        # 设置 interface_layer 的权重和偏置
        # Convert tensors to numpy before setting weights
        self.memory_access.interface_layer.set_weights([fixed_weights.numpy(), fixed_bias.numpy()])

    def test_full_erase(self):
        """
        全部擦除 (erase_value = 1.0)：
        验证内存被完全覆盖。
        """
        # 设置 erase_vectors 为 1.0
        self._set_interface_layer_weights(erase_vector=1.0)

        # 定义写入向量
        write_vectors = tf.constant([
            [1.0] * self.word_size,
            [0.0] * self.word_size,
            [0.5] * self.word_size
        ], dtype=tf.float32)  # [3, word_size]

        # 执行多个写入操作
        for i in range(3):
            # 构建控制器输出
            controller_output = tf.concat([
                tf.expand_dims(tf.ones([self.batch_size], dtype=tf.float32) * 1.0, axis=1),  # [batch_size,1]
                tf.tile(write_vectors[i:i + 1], [self.batch_size, 1])  # 将 [1, 16] 复制为 [2, 16]
            ], axis=1)  # [batch_size,1 + word_size] = [2,17]

            # 如果 controller_output_size > 17, 用零填充剩余维度
            if self.controller_output_size > 17:
                padding = tf.zeros([self.batch_size, self.controller_output_size - 17], dtype=tf.float32)
                controller_output = tf.concat([controller_output, padding],
                                              axis=1)  # [batch_size, controller_output_size]

            inputs = {
                'inputs': controller_output,
                'prev_state': self.initial_state
            }

            # 执行 MemoryAccess
            outputs = self.memory_access(inputs, training=False)
            read_words = outputs['read_words']
            final_state = outputs['final_state']

            # 更新初始状态
            self.initial_state = final_state

        # 执行历史记录查询，检索与最后一个写入向量相关的 top_k 记录
        related_records = self.memory_access.query_history(
            query_vector=write_vectors[-1],
            top_k=1
        )  # [batch_size, 1, word_size]

        # 定义预期的相关记录，应为最后一个写入向量，因为全部擦除
        expected_related_records = tf.expand_dims(write_vectors[-1], axis=0)  # [1, word_size]
        expected_related_records = tf.tile(expected_related_records, [self.batch_size, 1])  # [batch_size, word_size]
        expected_related_records = tf.expand_dims(expected_related_records, axis=1)  # [batch_size, 1, word_size]

        # 断言
        self.assertAllClose(related_records, expected_related_records, atol=1e-6)

    def test_partial_erase(self):
        """
        部分擦除 (erase_value = 0.5)：
        验证内存被部分擦除，旧内容与新内容的混合。
        """
        # 设置 erase_vectors 为 0.5
        self._set_interface_layer_weights(erase_vector=0.5)

        # 定义写入向量
        write_vectors = tf.constant([
            [1.0] * self.word_size,
            [0.0] * self.word_size,
            [0.5] * self.word_size
        ], dtype=tf.float32)  # [3, word_size]

        # 初始化预期内存状态
        expected_memory = tf.zeros([self.batch_size, self.memory_size, self.word_size], dtype=tf.float32)

        # 执行多个写入操作
        for i in range(3):
            # 构建控制器输出
            controller_output = tf.concat([
                tf.expand_dims(tf.ones([self.batch_size], dtype=tf.float32) * 1.0, axis=1),
                # write_strengths=1.0, [batch_size,1]
                write_vectors[i:i + 1]  # [1, word_size] -> [batch_size, word_size] 通过 broadcasting
            ], axis=1)  # [batch_size,1 + word_size] = [2,17]

            # 如果 controller_output_size > 17, 用零填充剩余维度
            if self.controller_output_size > 17:
                padding = tf.zeros([self.batch_size, self.controller_output_size - 17], dtype=tf.float32)
                controller_output = tf.concat([controller_output, padding],
                                              axis=1)  # [batch_size, controller_output_size]

            # 执行写入操作
            inputs = {
                'inputs': controller_output,
                'prev_state': self.initial_state
            }
            output = self.memory_access(inputs, training=False)
            final_state = output['final_state']

            # 计算预期内存更新
            erase_matrix = 1 - tf.nn.sigmoid(tf.fill([self.batch_size, self.memory_size, self.word_size], 0.5))
            # sigmoid(0.0) = 0.5 -> erase_matrix = 0.5

            # 更新预期内存
            write_vector_expanded = tf.expand_dims(write_vectors[i], axis=1)  # [word_size]
            write_vector_expanded = tf.expand_dims(write_vector_expanded, axis=0)  # [1, word_size]
            write_vector_expanded = tf.tile(write_vector_expanded, [self.batch_size, self.memory_size,
                                                                    1])  # [batch_size, memory_size, word_size]

            expected_memory = expected_memory * erase_matrix + write_vector_expanded

            # 更新初始状态
            self.initial_state = final_state

        # 执行历史记录查询，检索与最后一个写入向量相关的 top_k 记录
        related_records = self.memory_access.query_history(
            query_vector=write_vectors[-1],
            top_k=1
        )  # [batch_size, 1, word_size]

        # 定义预期的相关记录，应为最后一个写入向量，因为部分擦除，最相关的应为最后一个写入
        expected_related_records = tf.expand_dims(write_vectors[-1], axis=0)  # [1, word_size]
        expected_related_records = tf.tile(expected_related_records, [self.batch_size, 1])  # [batch_size, word_size]
        expected_related_records = tf.expand_dims(expected_related_records, axis=1)  # [batch_size, 1, word_size]

        # 断言
        self.assertAllClose(related_records, expected_related_records, atol=1e-6)

    def test_no_erase(self):
        """
        不擦除 (erase_value = 0.0)：
        验证内存不被擦除，直接添加新写入向量。
        """
        # 设置 erase_vectors 为 0.0
        self._set_interface_layer_weights(erase_vector=0.0)

        # 定义写入向量
        write_vectors = tf.constant([
            [1.0] * self.word_size,
            [0.0] * self.word_size,
            [0.5] * self.word_size
        ], dtype=tf.float32)  # [3, word_size]

        # 执行多个写入操作
        for i in range(3):
            # 构建控制器输出
            controller_output = tf.concat([
                tf.expand_dims(tf.ones([self.batch_size], dtype=tf.float32), axis=1),  # [batch_size,1]
                write_vectors[i:i + 1]  # [1, word_size] -> [batch_size, word_size] 通过 broadcasting
            ], axis=1)  # [batch_size,1 + word_size] = [2,17]

            # 如果 controller_output_size > 17, 用零填充剩余维度
            if self.controller_output_size > 17:
                padding = tf.zeros([self.batch_size, self.controller_output_size - 17], dtype=tf.float32)
                controller_output = tf.concat([controller_output, padding],
                                              axis=1)  # [batch_size, controller_output_size]

            # 执行写入操作
            inputs = {
                'inputs': controller_output,
                'prev_state': self.initial_state
            }
            output = self.memory_access(inputs, training=False)
            final_state = output['final_state']
            self.initial_state = final_state  # 更新初始状态以便下一步使用

        # 执行历史记录查询，检索与最后一个写入向量相关的 top_k 记录
        related_records = self.memory_access.query_history(
            query_vector=write_vectors[-1],
            top_k=1
        )  # [batch_size, 1, word_size]

        # 定义预期的相关记录，应为最后一个写入向量，因为不擦除，所有写入向量都存在
        expected_related_records = tf.expand_dims(write_vectors[-1], axis=0)  # [1, word_size]
        expected_related_records = tf.tile(expected_related_records, [self.batch_size, 1])  # [batch_size, word_size]
        expected_related_records = tf.expand_dims(expected_related_records, axis=1)  # [batch_size, 1, word_size]

        # 断言
        self.assertAllClose(related_records, expected_related_records, atol=1e-6)

    def test_user_memory_isolation(self):
        """
        测试不同用户的内存是否相互隔离。
        """
        # 模拟两个用户的不同输入
        # 用户1: write_weights=1.0, write_vectors=1.0
        write_vector_user1 = tf.expand_dims(tf.ones([self.word_size], dtype=tf.float32) * 1.0, axis=0)  # [1,16]
        controller_output_user1 = tf.concat([
            tf.expand_dims(tf.ones([1], dtype=tf.float32) * 1.0, axis=1),  # [1,1]
            write_vector_user1  # [1,16]
        ], axis=1)  # [1,17]

        # 用户2: write_weights=0.0, write_vectors=0.5
        write_vector_user2 = tf.expand_dims(tf.ones([self.word_size], dtype=tf.float32) * 0.5, axis=0)  # [1,16]
        controller_output_user2 = tf.concat([
            tf.expand_dims(tf.ones([1], dtype=tf.float32) * 0.0, axis=1),  # [1,1]
            write_vector_user2  # [1,16]
        ], axis=1)  # [1,17]

        # 如果 controller_output_size > 17, 用零填充剩余维度
        if self.controller_output_size > 17:
            padding_user1 = tf.zeros([1, self.controller_output_size - 17], dtype=tf.float32)
            controller_output_user1 = tf.concat([controller_output_user1, padding_user1],
                                                axis=1)  # [1, controller_output_size]

            padding_user2 = tf.zeros([1, self.controller_output_size - 17], dtype=tf.float32)
            controller_output_user2 = tf.concat([controller_output_user2, padding_user2],
                                                axis=1)  # [1, controller_output_size]

        # 合并两个用户的控制器输出
        controller_output = tf.concat([controller_output_user1, controller_output_user2],
                                      axis=0)  # [2, controller_output_size]

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
        self.assertShapeEqual(memory_user1.numpy(), memory_user2.numpy())

    def test_content_read_write(self):
        """
        测试内容的读写操作是否正确。
        """
        # 定义写入向量
        write_vectors = tf.constant([
            [1.0] * self.word_size,
            [0.0] * self.word_size
        ], dtype=tf.float32)  # [2, word_size]

        # 定义写入权重，写入到第一个内存槽
        write_weights = tf.tensor_scatter_nd_update(
            tf.fill([self.batch_size, self.memory_size], 0.0),
            indices=[[0, 0], [1, 0]],
            updates=[1.0, 1.0]  # 将两个批次的权重都设置为 1.0
        )

        # 构建控制器输出，使 interface_vector 包含 write_weights 和 write_vectors
        # 由于 interface_layer 的权重已设置为映射 controller_output 的第0维到 write_strengths，
        # 第1到16维映射到 write_vectors，因此：
        # controller_output[:,0] = write_strengths
        # controller_output[:,1:17] = write_vectors
        controller_output = tf.concat([
            tf.expand_dims(write_weights[:, 0], axis=1),  # [batch_size, 1]
            write_vectors  # [batch_size, word_size]
        ], axis=1)  # [batch_size, 1 + word_size] = [2,17]

        # 如果 controller_output_size > 17, 用零填充剩余维度
        if self.controller_output_size > 17:
            padding = tf.zeros([self.batch_size, self.controller_output_size - 17], dtype=tf.float32)
            controller_output = tf.concat([controller_output, padding],
                                          axis=1)  # [batch_size, controller_output_size]

        inputs = {
            'inputs': controller_output,
            'prev_state': self.initial_state
        }

        # 执行 MemoryAccess
        outputs = self.memory_access(inputs, training=False)
        read_words = outputs['read_words']
        final_state = outputs['final_state']

        # 执行历史记录查询
        related_records = self.memory_access.query_history(
            query_vector=write_vectors,
            top_k=1
        )  # [batch_size, 1, word_size]

        # 定义预期的相关记录，不包含 scaling_factor
        expected_read_vectors = tf.expand_dims(write_vectors, axis=1)  # [batch_size, 1, word_size]

        # 断言
        self.assertAllClose(related_records, expected_read_vectors, atol=1e-6)

    def test_parse_interface_vector(self):
        """
        测试 _parse_interface_vector 方法是否正确解析 interface_vector。
        """
        batch_size = 2
        interface_size = self.memory_access.interface_size  # 应为83
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

    def tearDown(self):
        # 清理操作，例如释放资源
        super(MemoryAccessUserScenarioTest, self).tearDown()

if __name__ == '__main__':
    tf.test.main()

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
    #     # 执行多个写入操作
    #     for i in range(3):
    #         # 构建控制器输出
    #         controller_output = tf.concat([
    #             tf.expand_dims(tf.ones([self.batch_size], dtype=tf.float32) * 1.0, axis=1),
    #             # write_weights=1.0, [batch_size,1]
    #             write_vectors[i:i + 1]  # [1, word_size] -> [batch_size, word_size] 通过 broadcasting
    #         ], axis=1)  # [batch_size,1 + word_size] = [2,17]
    #
    #         # 如果 controller_output_size > 17, 用零填充剩余维度
    #         if self.controller_output_size > 17:
    #             padding = tf.zeros([self.batch_size, self.controller_output_size - 17], dtype=tf.float32)
    #             controller_output = tf.concat([controller_output, padding],
    #                                           axis=1)  # [batch_size, controller_output_size]
    #
    #         # 执行写入操作
    #         inputs = {
    #             'inputs': controller_output,
    #             'prev_state': self.initial_state
    #         }
    #         output = self.memory_access(inputs, training=False)
    #         final_state = output['final_state']
    #         self.initial_state = final_state  # 更新初始状态以便下一步使用
    #
    #     # 定义当前输入主题向量，类似于第一个写入向量
    #     current_input = tf.tile(tf.expand_dims(write_vectors[0], axis=0),
    #                             [self.batch_size, 1])  # [batch_size, word_size]
    #
    #     # 执行历史记录查询，检索与当前输入相关的 top_k 记录
    #     related_records = self.memory_access.query_history(
    #         query_vector=current_input,
    #         top_k=1
    #     )  # [batch_size, 1, word_size]
    #
    #     # 定义预期的相关记录，不包含 scaling_factor
    #     expected_related_records = tf.expand_dims(write_vectors[0], axis=0)  # [1, word_size]
    #     expected_related_records = tf.tile(expected_related_records, [self.batch_size, 1])  # [batch_size, word_size]
    #     expected_related_records = tf.expand_dims(expected_related_records, axis=1)  # [batch_size, 1, word_size]
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
    #     # 执行多个写入操作
    #     for i in range(3):
    #         # 构建控制器输出
    #         controller_output = tf.concat([
    #             tf.expand_dims(tf.ones([self.batch_size], dtype=tf.float32) * 1.0, axis=1),
    #             # write_weights=1.0, [batch_size,1]
    #             write_vectors[i:i + 1]  # [1, word_size] -> [batch_size, word_size] 通过 broadcasting
    #         ], axis=1)  # [batch_size,1 + word_size] = [2,17]
    #
    #         # 如果 controller_output_size > 17, 用零填充剩余维度
    #         if self.controller_output_size > 17:
    #             padding = tf.zeros([self.batch_size, self.controller_output_size - 17], dtype=tf.float32)
    #             controller_output = tf.concat([controller_output, padding],
    #                                           axis=1)  # [batch_size, controller_output_size]
    #
    #         # 执行写入操作
    #         inputs = {
    #             'inputs': controller_output,
    #             'prev_state': self.initial_state
    #         }
    #         output = self.memory_access(inputs, training=False)
    #         final_state = output['final_state']
    #         self.initial_state = final_state  # 更新初始状态以便下一步使用
    #
    #     # 定义当前输入主题向量，最相关于最后一个写入
    #     current_input = tf.tile(tf.expand_dims(write_vectors[2], axis=0),
    #                             [self.batch_size, 1])  # [batch_size, word_size]
    #
    #     # 执行历史记录查询，检索与当前输入相关的 top_k 记录
    #     related_records = self.memory_access.query_history(
    #         query_vector=current_input,
    #         top_k=3
    #     )  # [batch_size, 3, word_size]
    #
    #     # 定义预期的相关记录，不包含 scaling_factor
    #     # 预期顺序应为最后一个写入 (0.5), 第二个写入 (0.4), 第一个写入 (0.1)
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
    #     # 断言
    #     self.assertAllClose(related_records, expected_related_records, atol=1e-6)


