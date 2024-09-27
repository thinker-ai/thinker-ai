import tensorflow as tf

from thinker_ai.agent.memory.humanoid_memory.dnc_new.addressing import WriteAllocation, weighted_softmax


class WriteAllocationTest(tf.test.TestCase):
    def setUp(self):
        super(WriteAllocationTest, self).setUp()
        self.memory_size = 3
        self.num_writes = 2
        self.epsilon = 1e-3  # 增大 epsilon

        # 定义生成 write_content_weights 的函数
        def custom_write_content_weights_fn(inputs: dict) -> tf.Tensor:
            write_content_keys = inputs.get('write_content_keys')
            write_content_strengths = inputs.get('write_content_strengths')
            if write_content_keys is not None and write_content_strengths is not None:
                # 使用 write_content_keys 和 write_content_strengths 以及 memory 来生成 write_content_weights
                memory = inputs.get('memory')
                if memory is None:
                    raise KeyError(
                        "'memory' must be provided when using 'write_content_keys' and 'write_content_strengths'")
                # 计算 scores
                scores = tf.matmul(write_content_keys, memory,
                                   transpose_b=True)  # [batch_size, num_writes, memory_size]
                # 计算 weighted_softmax
                write_content_weights = weighted_softmax(scores, write_content_strengths,
                                                         tf.nn.softmax)  # [batch_size, num_writes, memory_size]
                return write_content_weights
            else:
                # 使用 softmax 基于 usage 生成 write_content_weights
                usage = inputs.get('usage')
                write_gates_sum = inputs.get('write_gates_sum')
                if usage is not None and write_gates_sum is not None:
                    write_content_weights = tf.nn.softmax(tf.expand_dims(usage, axis=1),
                                                          axis=-1)  # [batch_size, num_writes, memory_size]
                    # 不要乘以 write_gates_sum，保持 write_content_weights 为均匀分布
                    return write_content_weights
                else:
                    raise KeyError(
                        "Inputs must contain either ('write_content_keys' and 'write_content_strengths') or ('usage' and 'write_gates_sum').")

        # 实例化 WriteAllocation 层时仅传入 write_content_weights_fn
        self.write_allocation_layer = WriteAllocation(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            epsilon=self.epsilon,
            write_content_weights_fn=custom_write_content_weights_fn
            # allocation_gate_fn 和 write_gate_fn 使用默认值
        )

    def test_gradient_computability(self):
        """验证梯度是否可以计算"""
        batch_size = 2  # 使用更大的批量大小以更好地测试
        memory = tf.Variable(tf.random.normal([batch_size, self.memory_size, 4]),
                             trainable=True)  # [batch_size, memory_size, word_size]
        usage = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32,
                            trainable=True)  # [batch_size, memory_size]
        write_content_keys = tf.random.uniform([batch_size, self.num_writes, 4])  # [batch_size, num_writes, word_size]
        write_content_strengths = tf.ones([batch_size, self.num_writes], dtype=tf.float32)  # [batch_size, num_writes]

        # 实例化 WriteAllocation 层，传入自定义的 gate 函数
        write_allocation_layer = WriteAllocation(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            epsilon=self.epsilon,
            write_content_weights_fn=self.write_allocation_layer.write_content_weights_fn,
            allocation_gate_fn=None,  # 使用默认的 allocation_gate_fn
            write_gate_fn=None  # 使用默认的 write_gate_fn
        )

        with tf.GradientTape() as tape:
            # 创建 inputs 字典，包含 memory
            inputs = {
                'usage': usage,
                'write_gates_sum': tf.ones([batch_size, self.num_writes], dtype=tf.float32),
                'write_content_keys': write_content_keys,
                'write_content_strengths': write_content_strengths,
                'memory': memory
            }

            # 调用 WriteAllocation 的 call 方法，并可选择覆盖 num_writes
            write_weights = write_allocation_layer(inputs, training=True)  # [batch_size, num_writes, memory_size]

            # 定义一个依赖于 usage 的损失函数
            loss = tf.reduce_sum(write_weights * tf.expand_dims(usage, axis=1))  # scalar

        # 计算梯度
        gradients = tape.gradient(loss, [memory, usage])

        # 检查梯度
        for grad, var_name in zip(gradients, ['memory', 'usage']):
            self.assertIsNotNone(grad, f"No gradient for {var_name}")
            grad_norm = tf.norm(grad).numpy()
            self.assertGreater(grad_norm, 1e-12, f"Gradient too small for {var_name}")
            tf.print(f"Gradient norm for {var_name}: {grad_norm}")

    def test_gradient_path_validity(self):
        """验证梯度传播路径的正确性"""
        batch_size = 2
        memory = tf.Variable(tf.random.normal([batch_size, self.memory_size, 4]),
                             trainable=True)  # [batch_size, memory_size, word_size]
        usage = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32,
                            trainable=True)  # [batch_size, memory_size]
        write_content_keys = tf.random.uniform([batch_size, self.num_writes, 4])  # [batch_size, num_writes, word_size]
        write_content_strengths = tf.ones([batch_size, self.num_writes], dtype=tf.float32)  # [batch_size, num_writes]

        # 实例化 WriteAllocation 层，传入自定义的 gate 函数
        write_allocation_layer = WriteAllocation(
            memory_size=self.memory_size,
            num_writes=self.num_writes,
            epsilon=self.epsilon,
            write_content_weights_fn=self.write_allocation_layer.write_content_weights_fn,
            allocation_gate_fn=None,  # 使用默认的 allocation_gate_fn
            write_gate_fn=None  # 使用默认的 write_gate_fn
        )

        with tf.GradientTape() as tape:
            # 创建 inputs 字典，包含 memory
            inputs = {
                'usage': usage,
                'write_gates_sum': tf.ones([batch_size, self.num_writes], dtype=tf.float32),
                'write_content_keys': write_content_keys,
                'write_content_strengths': write_content_strengths,
                'memory': memory
            }

            # 调用 WriteAllocation 的 call 方法，并可选择覆盖 num_writes
            write_weights = write_allocation_layer(inputs, training=True)  # [batch_size, num_writes, memory_size]

            # 定义一个依赖于 usage 的损失函数
            loss = tf.reduce_sum(write_weights * tf.expand_dims(usage, axis=1))  # scalar

        # 计算梯度
        gradients = tape.gradient(loss, [memory, usage])

        # 验证每个变量的梯度正确传播
        for grad, var_name in zip(gradients, ['memory', 'usage']):
            self.assertIsNotNone(grad, f"No gradient for {var_name}")
            grad_norm = tf.reduce_sum(tf.abs(grad)).numpy()
            self.assertGreater(grad_norm, 0, f"Some gradients are zero for {var_name}")
            tf.print(f"Gradient norm for {var_name}: {grad_norm}")

    def test_gradient_value_reasonableness(self):
        """验证梯度值的合理性"""
        batch_size = 2
        memory = tf.Variable(tf.random.normal([batch_size, self.memory_size, 4]),
                             trainable=True)  # [batch_size, memory_size, word_size]
        usage = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32,
                            trainable=True)  # [batch_size, memory_size]
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)  # [batch_size, num_writes]

        with tf.GradientTape() as tape:
            # 创建 inputs 字典
            inputs = {
                'usage': usage,
                'write_gates_sum': write_gates_sum
            }

            # 调用 WriteAllocation 的 call 方法，并可选择覆盖 num_writes
            write_weights = self.write_allocation_layer(inputs, training=True)  # [batch_size, num_writes, memory_size]

            # 定义一个依赖于 usage 的损失函数
            loss = tf.reduce_sum(write_weights * tf.expand_dims(usage, axis=1))  # scalar

        # 计算梯度
        gradients = tape.gradient(loss, [memory, usage])
        gradient_value = tf.reduce_sum(tf.abs(gradients[1])).numpy()  # usage 的梯度

        # 验证梯度值在合理范围内
        self.assertGreater(gradient_value, 1e-6, "Gradient is too small")
        self.assertLess(gradient_value, 1e6, "Gradient is too large")
        tf.print(f"Gradient value for usage: {gradient_value}")

    def test_write_allocation(self):
        """
        测试 WriteAllocation 类的 call 方法，确保其功能正确。
        """
        batch_size = 2

        # 创建初始使用率
        initial_usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [2,3]

        # 定义 write_gates_sum 全为1
        write_gates_sum = tf.ones([batch_size, self.num_writes], dtype=tf.float32)  # [2,2]

        # 创建 inputs 字典，不包含 'write_content_keys' 和 'write_content_strengths'
        inputs = {
            'usage': initial_usage,
            'write_gates_sum': write_gates_sum
        }

        # 调用 WriteAllocation 层
        write_allocation_weights = self.write_allocation_layer(inputs, training=False)  # [2,2,3]

        # 预期 write_allocation_weights 应为 soft allocations, 例如 [1/3,1/3,1/3]
        expected_write_allocation_weights = tf.constant([
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]],
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]]
        ], dtype=tf.float32)  # [2,2,3]

        self.assertAllClose(write_allocation_weights.numpy(), expected_write_allocation_weights.numpy(), atol=1e-5)

    def test_write_allocation_with_dynamic_num_writes(self):
        """测试 WriteAllocation 层在调用时动态覆盖 num_writes"""
        batch_size = 2
        dynamic_num_writes = 3  # 动态覆盖的 num_writes

        # 创建 usage 和 write_gates_sum
        usage = tf.zeros([batch_size, self.memory_size], dtype=tf.float32)  # [2,3]
        write_gates_sum = tf.ones([batch_size, dynamic_num_writes], dtype=tf.float32)  # [2,3]

        # 创建 inputs 字典，不包含 'write_content_keys' 和 'write_content_strengths'
        inputs = {
            'usage': usage,
            'write_gates_sum': write_gates_sum
        }

        # 调用 WriteAllocation 的 call 方法，并覆盖 num_writes
        write_weights = self.write_allocation_layer(inputs, training=False, num_writes=dynamic_num_writes)  # [2,3,3]

        # 预期 write_allocation_weights 应为 soft allocations, 例如 [1/3,1/3,1/3]
        expected_write_allocation_weights = tf.constant([
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]],
            [[1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3],
             [1 / 3, 1 / 3, 1 / 3]]
        ], dtype=tf.float32)  # [2,3,3]

        self.assertAllClose(write_weights.numpy(), expected_write_allocation_weights.numpy(), atol=1e-5)