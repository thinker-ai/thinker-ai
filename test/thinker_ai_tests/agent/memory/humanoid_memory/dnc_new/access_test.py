import collections
import tensorflow as tf
import numpy as np
import os

from thinker_ai.agent.memory.humanoid_memory.dnc_new import access


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

# 定义测试常量
BATCH_SIZE = 2
MEMORY_SIZE = 20
WORD_SIZE = 6
NUM_READS = 2
NUM_WRITES = 3
TIME_STEPS = 4
SEQUENCE_LENGTH = 1  # 可变的 sequence_length，您可以修改此值进行测试
INPUT_SIZE = 12      # 输入大小

# 定义 AccessState，用于保存访问模块的状态
AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))


class MemoryAccessTest(tf.test.TestCase):

    def setUp(self):
        super(MemoryAccessTest, self).setUp()
        # 初始化 MemoryAccess 模块
        self.module = access.MemoryAccess(
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_reads=NUM_READS,
            num_writes=NUM_WRITES
        )
        # 初始化状态
        self.initial_state = self.module.get_initial_state(batch_size=BATCH_SIZE)

    def testEdgeCaseInputs(self):
        """
        测试边界条件输入：所有输入为零，确保写操作和读操作的使用率更新正确。
        """
        # 将输入调整为 [TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]
        inputs = tf.zeros([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32)
        targets = tf.zeros([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], dtype=tf.float32)

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]  # [BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]
                inputs_t = {'inputs': input_t, 'prev_state': prev_state}
                output, prev_state = self.module(inputs_t, training=True)
                # 输出形状为 [BATCH_SIZE, SEQUENCE_LENGTH, NUM_READS, WORD_SIZE]
                # 为了匹配 targets，需要去除 SEQUENCE_LENGTH 维度
                output = tf.squeeze(output, axis=1)  # [BATCH_SIZE, NUM_READS, WORD_SIZE]
                outputs.append(output)

            output = tf.stack(outputs, axis=0)  # [TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE]
            loss = tf.reduce_mean(tf.square(output - targets))
            tf.print("Edge Case Test: Loss value:", loss)
            tf.print("Edge Case Test: Output values (sample):", output[:2, :2, :])

        gradients = tape.gradient(loss, self.module.trainable_variables)

        # 打印所有可训练变量的名称和形状
        tf.print("\nAll trainable variables (Edge Case):")
        for var in self.module.trainable_variables:
            tf.print("Variable:", var.name, ", Shape:", var.shape)

        # 添加调试信息，打印变量名称和梯度范数
        for var, grad in zip(self.module.trainable_variables, gradients):
            if grad is None:
                tf.print(f"Gradient for variable '{var.name}' is None")
            else:
                grad_norm = tf.norm(grad)
                tf.print(f"Gradient norm for variable '{var.name}':", grad_norm)
                # 仅断言特定变量具有非零梯度
                if 'write_vectors' in var.name:
                    self.assertGreater(grad_norm, 1e-12,
                                       f"Gradient for variable '{var.name}' is too small for edge case inputs")

    def testNonEdgeCaseInputs(self):
        """
        测试非边界条件输入：使用随机非零输入和目标，确保写操作和读操作的使用率更新正确。
        """
        # 使用随机非零输入和目标，调整为四维输入
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32)
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], dtype=tf.float32)

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]  # [BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]
                inputs_t = {'inputs': input_t, 'prev_state': prev_state}
                output, prev_state = self.module(inputs_t, training=True)
                # 输出形状为 [BATCH_SIZE, SEQUENCE_LENGTH, NUM_READS, WORD_SIZE]
                output = tf.squeeze(output, axis=1)  # [BATCH_SIZE, NUM_READS, WORD_SIZE]
                outputs.append(output)

            output = tf.stack(outputs, axis=0)  # [TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE]
            loss = tf.reduce_mean(tf.square(output - targets))
            tf.print("Non-Edge Case Test: Loss value:", loss)
            tf.print("Non-Edge Case Test: Output values (sample):", output[:2, :2, :])

        gradients = tape.gradient(loss, self.module.trainable_variables)

        # 打印所有可训练变量的名称和形状
        tf.print("\nAll trainable variables (Non-Edge Case):")
        for var in self.module.trainable_variables:
            tf.print("Variable:", var.name, ", Shape:", var.shape)

        # 添加调试信息，打印变量名称和梯度范数
        for var, grad in zip(self.module.trainable_variables, gradients):
            if grad is None:
                tf.print(f"Gradient for variable '{var.name}' is None (Non-Edge Case)")
            else:
                grad_norm = tf.norm(grad)
                tf.print(f"Gradient norm for variable '{var.name}':", grad_norm)
                self.assertGreater(grad_norm, 1e-12,
                                   f"Gradient for variable '{var.name}' is too small for non-edge case inputs")
                self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var.name}' is too large for non-edge case inputs")

    def testGradients_SingleConfiguration(self):
        """
        测试单一配置下模型的梯度流动情况。
        包含以下步骤：
        1. 初始化输入为 tf.Variable 以确保梯度追踪。
        2. 通过模型前向传播计算输出。
        3. 计算损失（均方误差）。
        4. 使用 tf.GradientTape 计算损失相对于输入和所有可训练变量的梯度。
        5. 打印并断言梯度的有效性。
        """
        # 使用随机非零输入和目标
        inputs = tf.Variable(np.random.randn(TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE), dtype=tf.float32)
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        with tf.GradientTape() as tape:
            # 初始化前一状态
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]  # [BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]
                # 前向传播
                inputs_t = {'inputs': input_t, 'prev_state': prev_state}
                output, prev_state = self.module(inputs_t, training=True)
                # 输出形状为 [BATCH_SIZE, SEQUENCE_LENGTH, NUM_READS, WORD_SIZE]
                output = tf.squeeze(output, axis=1)  # [BATCH_SIZE, NUM_READS, WORD_SIZE]
                outputs.append(output)

            # 堆叠所有时间步的输出
            output = tf.stack(outputs, axis=0)  # [TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE]
            loss = tf.reduce_mean(tf.square(output - targets))
            tf.print("Single Configuration Test: Loss value:", loss)
            tf.print("Single Configuration Test: Output values (sample):", output[:2, :2, :])

        # 计算梯度，相对于输入和所有可训练变量
        tensors_to_check = [inputs] + list(self.module.trainable_variables)
        gradients = tape.gradient(loss, tensors_to_check)

        # 打印所有可训练变量的名称和形状
        tf.print("\nAll trainable variables (Single Configuration):")
        for var in self.module.trainable_variables:
            tf.print("Variable:", var.name, ", Shape:", var.shape)

        # 打印并检查每个变量的梯度
        for grad, var in zip(gradients, tensors_to_check):
            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
            if grad is None:
                tf.print(f"Gradient for {var_name} is None.")
            else:
                grad_norm = tf.norm(grad)
                tf.print(f"Variable: {var_name}, Gradient Norm: {grad_norm.numpy()}")

                # 断言梯度不为 None 且在合理范围内
                self.assertIsNotNone(grad, f"Gradient is None for variable {var_name}")
                self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var_name}' is too large")
                self.assertGreater(grad_norm, 1e-12, f"Gradient for variable '{var_name}' is too small")

    def testBuildAndTrain(self):
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE])

        optimizer = tf.optimizers.SGD(learning_rate=1.0)

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]  # [BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]
                output, prev_state = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                }, training=True)
                output = tf.squeeze(output, axis=1)  # [BATCH_SIZE, NUM_READS, WORD_SIZE]
                outputs.append(output)

            output = tf.stack(outputs, axis=0)  # [TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE]

            targets = tf.random.uniform([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
            loss = tf.reduce_mean(tf.square(output - targets))

        gradients = tape.gradient(loss, self.module.trainable_variables)
        # 检查哪些梯度为 None
        for grad, var in zip(gradients, self.module.trainable_variables):
            if grad is None:
                tf.print(f"No gradient provided for variable {var.name}")
        optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))

    def testValidReadMode(self):
        # 构建输入字典，而不是直接传入 Tensor
        inputs = {
            'inputs': tf.random.normal([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]),
            'prev_state': self.initial_state
        }

        # 手动调用 build 方法
        self.module.build({'inputs': tf.TensorShape([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]),
                           'prev_state': self.initial_state})

        # Tile memory to match sequence_length
        memory_tiled = tf.tile(inputs['prev_state'].memory, [SEQUENCE_LENGTH, 1, 1])

        # 提取 'inputs' 字段传递给 _read_inputs
        processed_inputs = self.module._read_inputs(tf.reshape(inputs['inputs'], [-1, INPUT_SIZE]), memory_tiled)

        # 检查 processed_inputs 的内容
        self.assertIn('read_content_weights', processed_inputs)
        self.assertIn('write_content_weights', processed_inputs)
        self.assertIn('write_vectors', processed_inputs)
        self.assertIn('erase_vectors', processed_inputs)
        # 其他相关断言

    def testWriteWeights(self):
        # 注意，这个测试需要根据最新的 MemoryAccess 类进行调整
        # 由于 _write_weights 方法已被集成到 _step 方法中，需要调整测试方式

        # 初始化必要的输入
        batch_size = BATCH_SIZE
        num_writes = NUM_WRITES
        memory_size = MEMORY_SIZE
        word_size = WORD_SIZE

        prev_state = self.initial_state

        # 创建输入参数
        controller_output = tf.random.normal([batch_size, SEQUENCE_LENGTH, INPUT_SIZE])
        reshaped_controller_output = tf.reshape(controller_output, [-1, INPUT_SIZE])

        # Tile memory to match sequence_length
        memory_tiled = tf.tile(prev_state.memory, [SEQUENCE_LENGTH, 1, 1])

        # 处理输入，生成需要的参数
        processed_inputs = self.module._read_inputs(reshaped_controller_output, memory_tiled)

        # 调用 _step 方法
        next_state = self.module._step(processed_inputs, prev_state)

        # 检查 write_weights 是否符合预期
        write_weights = tf.squeeze(next_state['write_weights'], axis=1)  # [batch_size, num_writes, memory_size]
        self.assertEqual(write_weights.shape, [batch_size, num_writes, memory_size])

    def testReadWeights(self):
        # 与 testWriteWeights 类似，需要调整测试方式

        # 初始化必要的输入
        batch_size = BATCH_SIZE
        num_reads = NUM_READS
        memory_size = MEMORY_SIZE
        word_size = WORD_SIZE

        prev_state = self.initial_state

        # 创建输入参数
        controller_output = tf.random.normal([batch_size, SEQUENCE_LENGTH, INPUT_SIZE])
        reshaped_controller_output = tf.reshape(controller_output, [-1, INPUT_SIZE])

        # Tile memory to match sequence_length
        memory_tiled = tf.tile(prev_state.memory, [SEQUENCE_LENGTH, 1, 1])

        # 处理输入，生成需要的参数
        processed_inputs = self.module._read_inputs(reshaped_controller_output, memory_tiled)

        # 调用 _step 方法
        next_state = self.module._step(processed_inputs, prev_state)

        # 检查 read_weights 是否符合预期
        read_weights = tf.squeeze(next_state['read_weights'], axis=1)  # [batch_size, num_reads, memory_size]
        self.assertEqual(read_weights.shape, [batch_size, num_reads, memory_size])

    def testRandomInitialState(self):
        # 使用随机初始化状态而非全零状态
        random_state = AccessState(
            memory=tf.Variable(tf.random.uniform([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE]), name='memory'),
            read_weights=tf.Variable(tf.random.uniform([BATCH_SIZE, 1, NUM_READS, MEMORY_SIZE]), name='read_weights'),
            write_weights=tf.Variable(tf.random.uniform([BATCH_SIZE, 1, NUM_WRITES, MEMORY_SIZE]), name='write_weights'),
            linkage=self.module.temporal_linkage.get_initial_state(batch_size=BATCH_SIZE),
            usage=tf.Variable(tf.random.uniform([BATCH_SIZE, MEMORY_SIZE]), name='usage')
        )

        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE])

        optimizer = tf.optimizers.SGD(learning_rate=1.0)

        with tf.GradientTape() as tape:
            prev_state = random_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, prev_state = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                }, training=True)
                output = tf.squeeze(output, axis=1)
                outputs.append(output)

            output = tf.stack(outputs, axis=0)

            targets = tf.random.uniform([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
            loss = tf.reduce_mean(tf.square(output - targets))

        gradients = tape.gradient(loss, self.module.trainable_variables)
        # 检查哪些梯度为 None
        for grad, var in zip(gradients, self.module.trainable_variables):
            self.assertIsNotNone(grad, f"No gradient provided for variable {var.name}")

    def testGradientRange(self):
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE])

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, prev_state = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                }, training=True)
                output = tf.squeeze(output, axis=1)
                outputs.append(output)

            output = tf.stack(outputs, axis=0)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, self.module.trainable_variables)

        for grad in gradients:
            if grad is not None:
                grad_norm = tf.norm(grad).numpy()
                self.assertGreater(grad_norm, 1e-12, "Gradient is too small")
                self.assertLess(grad_norm, 1e3, "Gradient is too large")

    def testSmallNonZeroInputs(self):
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], mean=0.0, stddev=1e-3)
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], mean=0.0, stddev=1e-3)

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, prev_state = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                }, training=True)
                output = tf.squeeze(output, axis=1)
                outputs.append(output)

            output = tf.stack(outputs, axis=0)
            loss = tf.reduce_mean(tf.square(output - targets))
            tf.print("Test Small Non-Zero: Loss value:", loss)
            tf.print("Test Small Non-Zero: Output values (sample):", output.numpy()[0, :2, :])

        gradients = tape.gradient(loss, self.module.trainable_variables)

        # 添加调试信息，打印变量名称和梯度范数
        for var, grad in zip(self.module.trainable_variables, gradients):
            if grad is None:
                tf.print(f"Gradient for variable '{var.name}' is None")
            else:
                grad_norm = tf.norm(grad).numpy()
                tf.print(f"Gradient norm for variable '{var.name}': {grad_norm}")

                self.assertGreater(grad_norm, 1e-12,
                                   f"Gradient for variable '{var.name}' is too small for small non-zero inputs")
                self.assertLess(grad_norm, 1e3,
                                f"Gradient for variable '{var.name}' is too large for small non-zero inputs")

    def testNonZeroInputs(self):
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE])
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        with tf.GradientTape() as tape:
            prev_state = self.initial_state
            outputs = []
            for t in range(TIME_STEPS):
                input_t = inputs[t]
                output, prev_state = self.module({
                    'inputs': input_t,
                    'prev_state': prev_state
                }, training=True)
                output = tf.squeeze(output, axis=1)
                outputs.append(output)

            output = tf.stack(outputs, axis=0)
            loss = tf.reduce_mean(tf.square(output - targets))
            tf.print("Test Non-Zero: Loss value:", loss)

        gradients = tape.gradient(loss, self.module.trainable_variables)

        # 添加调试信息
        for i, grad in enumerate(gradients):
            if grad is None:
                tf.print(f"Gradient for variable {i} is None")
            else:
                grad_norm = tf.norm(grad).numpy()
                tf.print(f"Gradient norm for variable {i}: {grad_norm}")

                self.assertGreater(grad_norm, 1e-12, "Gradient is too small for non-zero inputs")
                self.assertLess(grad_norm, 1e3, "Gradient is too large for non-zero inputs")

    def testGradients_VaryingConfigurations(self):
        MEMORY_SIZE_OPTIONS = [20, 50]  # 不同的内存大小
        WORD_SIZE_OPTIONS = [6, 12]  # 不同的字大小
        NUM_READS_OPTIONS = [2, 4]  # 不同的读取数量
        NUM_WRITES_OPTIONS = [3, 5]  # 不同的写入数量

        for MEMORY_SIZE in MEMORY_SIZE_OPTIONS:
            for WORD_SIZE in WORD_SIZE_OPTIONS:
                for NUM_READS in NUM_READS_OPTIONS:
                    for NUM_WRITES in NUM_WRITES_OPTIONS:
                        print(f"Testing with MEMORY_SIZE={MEMORY_SIZE}, WORD_SIZE={WORD_SIZE}, "
                              f"NUM_READS={NUM_READS}, NUM_WRITES={NUM_WRITES}")

                        self.module = access.MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)

                        # Initialize prev_state with random values
                        initial_memory = tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])
                        initial_read_weights = tf.random.uniform([BATCH_SIZE, 1, NUM_READS, MEMORY_SIZE])
                        initial_write_weights = tf.random.uniform([BATCH_SIZE, 1, NUM_WRITES, MEMORY_SIZE])
                        initial_linkage = self.module.temporal_linkage.get_initial_state(batch_size=BATCH_SIZE)
                        initial_usage = self.module.freeness.get_initial_state([BATCH_SIZE])

                        self.initial_state = AccessState(
                            memory=tf.Variable(initial_memory, name='memory'),
                            read_weights=tf.Variable(initial_read_weights, name='read_weights'),
                            write_weights=tf.Variable(initial_write_weights, name='write_weights'),
                            linkage=initial_linkage,
                            usage=tf.Variable(initial_usage, name='usage')
                        )

                        # 创建一个随机序列输入
                        input_sequence = tf.Variable(np.random.randn(TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE),
                                                     dtype=tf.float32, trainable=True)
                        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

                        with tf.GradientTape() as tape:
                            prev_state = self.initial_state
                            outputs = []
                            for t in range(TIME_STEPS):
                                input_t = input_sequence[t]
                                input_dict = {'inputs': input_t, 'prev_state': prev_state}
                                output, prev_state = self.module(input_dict, training=True)
                                output = tf.squeeze(output, axis=1)
                                outputs.append(output)
                            output = tf.stack(outputs, axis=0)
                            loss = tf.reduce_mean(tf.square(output - targets))
                            tf.debugging.check_numerics(loss, "Loss contains NaNs or Infs")
                            print(f"Loss: {loss.numpy()}")

                        tensors_to_check = [input_sequence] + list(self.module.trainable_variables)

                        gradients = tape.gradient(loss, tensors_to_check)

                        # Debug output for gradient information
                        for grad, var in zip(gradients, tensors_to_check):
                            var_name = var.name if isinstance(var, tf.Variable) else 'inputs'
                            if grad is None:
                                print(f"Gradient for {var_name} is None.")
                                self.fail(f"Gradient for {var_name} is None, test failed.")
                            else:
                                grad_norm = tf.norm(grad).numpy()
                                if grad_norm == 0.0:
                                    print(f"Warning: Gradient for {var_name} is zero.")
                                    print(f"Details for {var_name}: {var.numpy()}")
                                    self.fail(f"Gradient for {var_name} is zero, test failed.")
                                else:
                                    print(f"Variable: {var_name}, Gradient Norm: {grad_norm}")
                                    self.assertIsNotNone(grad, f"Gradient is None for variable {var_name}")
                                    self.assertLess(grad_norm, 1e3, f"Gradient norm for {var_name} is too large.")
                                    self.assertGreater(grad_norm, 1e-12, f"Gradient norm for {var_name} is too small.")

    def testSublayersRegistration(self):
        """
        Test whether all sublayers are properly registered in the MemoryAccess module.
        """
        expected_sublayers = [
            'write_content_weights', 'read_content_weights', 'temporal_linkage', 'freeness',
            'write_vectors', 'erase_vectors', 'free_gate', 'allocation_gate',
            'write_gate', 'read_mode', 'write_strengths', 'read_strengths', 'write_keys', 'read_keys'
        ]

        # Instantiate the MemoryAccess module
        module = access.MemoryAccess(
            memory_size=MEMORY_SIZE,
            word_size=WORD_SIZE,
            num_reads=NUM_READS,
            num_writes=NUM_WRITES
        )

        # Initialize prev_state
        initial_memory = tf.zeros([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE], dtype=tf.float32)
        initial_read_weights = tf.zeros([BATCH_SIZE, 1, NUM_READS, MEMORY_SIZE], dtype=tf.float32)
        initial_write_weights = tf.zeros([BATCH_SIZE, 1, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
        initial_linkage = module.temporal_linkage.get_initial_state(batch_size=BATCH_SIZE)
        initial_usage = module.freeness.get_initial_state([BATCH_SIZE])

        initial_state = AccessState(
            memory=initial_memory,
            read_weights=initial_read_weights,
            write_weights=initial_write_weights,
            linkage=initial_linkage,
            usage=initial_usage
        )

        # Create a single time-step random input
        dummy_input = tf.random.normal([BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32)

        input_dict = {
            'inputs': dummy_input,
            'prev_state': initial_state
        }

        # Call the module once to build it
        try:
            output, _ = module(input_dict, training=False)
        except Exception as e:
            self.fail(f"MemoryAccess module failed to build with input_dict: {e}")

        # Manually collect sublayers from the module's attributes using vars()
        actual_sublayers = []
        for attr_name, attr_value in vars(module).items():
            if isinstance(attr_value, tf.keras.layers.Layer):
                actual_sublayers.append(attr_value.name)

        # Print the actual sublayers for debugging
        print("Expected sublayers:", expected_sublayers)
        print("Actual sublayers:", actual_sublayers)

        # Check that all expected sublayers are in actual_sublayers
        for sublayer in expected_sublayers:
            self.assertIn(sublayer, actual_sublayers, f"Sublayer '{sublayer}' is not registered.")

    def testSublayersGradients(self):
        """
        测试所有子层内部的梯度流动情况，确保没有梯度阻断。
        """
        inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE])
        targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])

        prev_state = self.initial_state

        for t in range(TIME_STEPS):
            input_t = inputs[t]  # 形状为 (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
            target_t = targets[t]

            with tf.GradientTape() as tape:
                tape.watch(self.module.trainable_variables)
                output, next_state = self.module({'inputs': input_t, 'prev_state': prev_state}, training=True)
                output = tf.squeeze(output, axis=1)
                loss = tf.reduce_mean((output - target_t) ** 2)

            gradients = tape.gradient(loss, self.module.trainable_variables)
            # 检查梯度是否为 None
            for grad, var in zip(gradients, self.module.trainable_variables):
                self.assertIsNotNone(grad, f"Gradient for variable {var.name} is None.")

            # 更新 prev_state
            prev_state = next_state
    # @parameterized.expand([
    #     ("all_zero_inputs", tf.zeros, tf.zeros),
    #     ("random_inputs", tf.random.normal, tf.random.normal),
    # ])
    # def test_allocation_cases(self, name, input_fn, target_fn):
    #     """
    #     参数化测试：测试不同类型的输入（全零、随机）。
    #     """
    #     # 构建输入和目标
    #     inputs = input_fn([TIME_STEPS, BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE], dtype=tf.float32)
    #     targets = target_fn([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], dtype=tf.float32)
    #
    #     with tf.GradientTape() as tape:
    #         prev_state = self.initial_state
    #         outputs = []
    #         for t in range(TIME_STEPS):
    #             input_t = inputs[t]  # [BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]
    #             # 构建输入字典，包含 'inputs' 和 'prev_state'
    #             inputs_t = {'inputs': input_t, 'prev_state': prev_state}
    #             output, prev_state = self.module(inputs_t, training=True)
    #             outputs.append(output)
    #
    #         output = tf.stack(outputs, axis=0)  # [TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE]
    #         loss = tf.reduce_mean(tf.square(output - targets))
    #         tf.print(f"{name}: Loss value:", loss)
    #         tf.print(f"{name}: Output values (sample):", output[:2, :2, :])
    #
    #     gradients = tape.gradient(loss, self.module.trainable_variables)
    #
    #     # 打印所有可训练变量的名称和形状
    #     tf.print(f"\nAll trainable variables for {name}:")
    #     for var in self.module.trainable_variables:
    #         tf.print("Variable:", var.name, ", Shape:", var.shape)
    #
    #     # 添加调试信息，打印变量名称和梯度范数
    #     for var, grad in zip(self.module.trainable_variables, gradients):
    #         if grad is None:
    #             tf.print(f"Gradient for variable '{var.name}' is None ({name})")
    #         else:
    #             grad_norm = tf.norm(grad)
    #             tf.print(f"Gradient norm for variable '{var.name}':", grad_norm)
    #             self.assertGreater(grad_norm, 1e-12,
    #                                f"Gradient for variable '{var.name}' is too small for {name}")
    #             self.assertLess(grad_norm, 1e3, f"Gradient for variable '{var.name}' is too large for {name}")

    def testWriteGradient(self):
        memory = tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])
        usage = tf.random.uniform([BATCH_SIZE, MEMORY_SIZE])
        allocation_gate = tf.random.uniform([BATCH_SIZE, NUM_WRITES])
        write_gate = tf.random.uniform([BATCH_SIZE, NUM_WRITES])
        write_content_keys = tf.random.uniform([BATCH_SIZE, NUM_WRITES, WORD_SIZE])
        write_content_strengths = tf.random.uniform([BATCH_SIZE, NUM_WRITES])

        inputs = {
            'allocation_gate': allocation_gate,
            'write_gate': write_gate,
            'write_content_keys': write_content_keys,
            'write_content_strengths': write_content_strengths
        }

        with tf.GradientTape() as tape:
            tape.watch([memory, usage])
            write_weights = self.module._write_weights(inputs, memory, usage)
            loss = tf.reduce_sum(write_weights)

        gradients = tape.gradient(loss, [memory, usage])

        for grad, var_name in zip(gradients, ['memory', 'usage']):
            self.assertIsNotNone(grad, f"No gradient for {var_name}")
            self.assertGreater(tf.norm(grad).numpy(), 1e-12, f"Gradient too small for {var_name}")

    def testReadWeightsGradient(self):
        memory = tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])
        prev_read_weights = tf.random.uniform([BATCH_SIZE, NUM_READS, MEMORY_SIZE])
        link = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE])
        read_content_keys = tf.random.uniform([BATCH_SIZE, NUM_READS, WORD_SIZE])
        read_content_strengths = tf.random.uniform([BATCH_SIZE, NUM_READS])
        read_mode = tf.random.uniform([BATCH_SIZE, NUM_READS, 1 + 2 * NUM_WRITES])

        inputs = {
            'read_content_weights': read_content_keys,
            'read_content_strengths': read_content_strengths,
            'read_mode': read_mode,
        }

        with tf.GradientTape() as tape:
            tape.watch([memory, prev_read_weights, link])
            read_weights = self.module._read_weights(inputs, memory, prev_read_weights, link)
            loss = tf.reduce_sum(read_weights)

        gradients = tape.gradient(loss, [memory, prev_read_weights, link])

        for grad, var_name in zip(gradients, ['memory', 'prev_read_weights', 'link']):
            self.assertIsNotNone(grad, f"No gradient for {var_name}")
            self.assertGreater(tf.norm(grad).numpy(), 1e-12, f"Gradient too small for {var_name}")

    # def testModelSavingAndLoading(self):
    #     """
    #     测试 MemoryAccess 模型的保存和加载功能，确保权重和配置被正确保留。
    #     """
    #     import tempfile
    #     from tensorflow.keras.models import save_model, load_model
    #
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         save_path = os.path.join(tmpdirname, 'memory_access_model')
    #         save_model(self.module, save_path, save_format='tf')
    #
    #         # 加载模型
    #         loaded_module = load_model(save_path, custom_objects={
    #             'MemoryAccess': access.MemoryAccess,
    #             'AccessState': access.AccessState,
    #             'TemporalLinkageState': addressing.TemporalLinkageState
    #         })
    #
    #         # 比较权重
    #         for original_var, loaded_var in zip(self.module.trainable_variables, loaded_module.trainable_variables):
    #             self.assertAllClose(original_var.numpy(), loaded_var.numpy(), atol=1e-6,
    #                                 msg=f"Mismatch in variable '{original_var.name}' after loading.")

    # def testDifferentDataTypes(self):
    #     """
    #     测试 MemoryAccess 模型在不同数据类型（如 float16）下的兼容性。
    #     """
    #     # 创建新的模块实例，使用 float16
    #     module_fp16 = access.MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES, dtype=tf.float16)
    #
    #     # 初始化状态
    #     initial_state_fp16 = access.AccessState(
    #         memory=tf.Variable(tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE], dtype=tf.float16),
    #                            name='memory_fp16'),
    #         read_weights=tf.Variable(tf.random.normal([BATCH_SIZE, NUM_READS, MEMORY_SIZE], dtype=tf.float16),
    #                                  name='read_weights_fp16'),
    #         write_weights=tf.Variable(tf.random.normal([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float16),
    #                                   name='write_weights_fp16'),
    #         linkage=addressing.TemporalLinkageState(
    #             link=tf.Variable(tf.random.normal([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float16),
    #                              name='link_fp16'),
    #             precedence_weights=tf.Variable(
    #                 tf.random.normal([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float16),
    #                 name='precedence_weights_fp16')
    #         ),
    #         usage=tf.Variable(tf.random.normal([BATCH_SIZE, MEMORY_SIZE], dtype=tf.float16), name='usage_fp16')
    #     )
    #
    #     # 创建输入和目标
    #     inputs = tf.Variable(np.random.randn(TIME_STEPS, BATCH_SIZE, INPUT_SIZE).astype(np.float16), dtype=tf.float16,
    #                          trainable=True)
    #     targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], dtype=tf.float16)
    #
    #     with tf.GradientTape() as tape:
    #         output, _ = module_fp16({'inputs': inputs, 'prev_state': initial_state_fp16})
    #         loss = tf.reduce_sum(output)
    #
    #     gradients = tape.gradient(loss, [inputs] + list(module_fp16.trainable_variables))
    #
    #     # 检查梯度
    #     for grad, var in zip(gradients, [inputs] + list(module_fp16.trainable_variables)):
    #         var_name = var.name
    #         if grad is None:
    #             self.fail(f"Gradient for variable '{var_name}' is None.")
    #         else:
    #             grad_norm = tf.norm(grad).numpy()
    #             self.assertGreater(grad_norm, 1e-3, f"Gradient norm for '{var_name}' is too small for float16.")
    #             self.assertLess(grad_norm, 1e3, f"Gradient norm for '{var_name}' is too large for float16.")    #
    #
    # def testDifferentBatchSizesAndSequenceLengths(self):
    #     """
    #     测试模型在不同批次大小和序列长度下的表现。
    #     """
    #     batch_sizes = [1, 4, 8]
    #     time_steps_options = [1, 10, 20]
    #
    #     for batch_size in batch_sizes:
    #         for time_steps in time_steps_options:
    #             print(f"Testing with BATCH_SIZE={batch_size}, TIME_STEPS={time_steps}")
    #
    #             # 创建新的模块实例
    #             self.module = access.MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
    #
    #             # 初始化随机状态
    #             self.initial_state = AccessState(
    #                 memory=tf.zeros([batch_size, MEMORY_SIZE, WORD_SIZE], dtype=tf.float32),
    #                 read_weights=tf.zeros([batch_size, NUM_READS, MEMORY_SIZE], dtype=tf.float32),
    #                 write_weights=tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32),
    #                 linkage=addressing.TemporalLinkageState(
    #                     link=tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
    #                     precedence_weights=tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
    #                 ),
    #                 usage=tf.zeros([batch_size, MEMORY_SIZE], dtype=tf.float32)
    #             )
    #
    #             # 创建输入和目标
    #             inputs = tf.random.normal([time_steps, batch_size, INPUT_SIZE])
    #             targets = tf.random.normal([time_steps, batch_size, NUM_READS, WORD_SIZE])
    #
    #             loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
    #
    #             # 检查梯度
    #             self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))
    #
    #
    # def testMalformedInputs(self):
    #     """
    #     测试模型对异常输入的处理能力，确保抛出适当的错误。
    #     """
    #     # 错误的输入形状
    #     malformed_inputs_shape = tf.random.normal([BATCH_SIZE, INPUT_SIZE], dtype=tf.float32)  # 缺少 TIME_STEPS 维度
    #
    #     with self.assertRaises(ValueError):
    #         self.module({'inputs': malformed_inputs_shape, 'prev_state': self.initial_state})
    #
    #     # 错误的数据类型
    #     malformed_inputs_dtype = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE], dtype=tf.int32)  # 错误的数据类型
    #
    #     with self.assertRaises(TypeError):
    #         self.module({'inputs': malformed_inputs_dtype, 'prev_state': self.initial_state})
    #
    #     # 缺少必要的输入字段
    #     incomplete_input_dict = {'inputs': tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE], dtype=tf.float32)}
    #
    #     with self.assertRaises(KeyError):
    #         self.module(incomplete_input_dict)
    #
    # def testTrainingAndInferenceModes(self):
    #     """
    #     测试模型在训练模式和推理模式下的行为是否一致。
    #     """
    #     inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
    #     targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
    #
    #     # 训练模式
    #     with tf.GradientTape() as tape_train:
    #         output_train, _ = self.module({'inputs': inputs, 'prev_state': self.initial_state})
    #         loss_train = tf.reduce_sum(output_train)
    #
    #     gradients_train = tape_train.gradient(loss_train, [inputs] + list(self.module.trainable_variables))
    #
    #     # 推理模式
    #     with tf.GradientTape() as tape_infer:
    #         output_infer, _ = self.module({'inputs': inputs, 'prev_state': self.initial_state})
    #         loss_infer = tf.reduce_sum(output_infer)
    #
    #     gradients_infer = tape_infer.gradient(loss_infer, [inputs] + list(self.module.trainable_variables))
    #
    #     # 比较损失
    #     self.assertAllClose(loss_train, loss_infer, atol=1e-6, msg="Loss mismatch between training and inference modes")
    #
    #     # 比较梯度
    #     for grad_train, grad_infer in zip(gradients_train, gradients_infer):
    #         if grad_train is not None and grad_infer is not None:
    #             self.assertAllClose(grad_train, grad_infer, atol=1e-6,
    #                                 msg="Gradient mismatch between training and inference modes")
    #         else:
    #             self.assertEqual(grad_train is None, grad_infer is None,
    #                              msg="Gradient presence mismatch between training and inference modes")
    #
    #
    # def testCosineWeightsFunctionality(self):
    #     """
    #     测试 CosineWeights 子层的功能，确保其计算内容权重正确。
    #     """
    #     cosine_weights = addressing.CosineWeights(num_heads=NUM_WRITES, word_size=WORD_SIZE, name='test_cosine_weights')
    #
    #     memory = tf.random.normal([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE])
    #     keys = tf.random.normal([BATCH_SIZE, NUM_WRITES, WORD_SIZE])
    #     strengths = tf.random.uniform([BATCH_SIZE, NUM_WRITES], minval=0.1, maxval=10.0)
    #
    #     inputs = {
    #         'memory': memory,
    #         'keys': keys,
    #         'strengths': strengths
    #     }
    #
    #     content_weights = cosine_weights(inputs)
    #
    #     # 检查输出形状
    #     self.assertEqual(content_weights.shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE])
    #
    #     # 检查权重和为1（softmax性质）
    #     sum_weights = tf.reduce_sum(content_weights, axis=2)
    #     self.assertAllClose(sum_weights, tf.ones_like(sum_weights), atol=1e-3, msg="Content weights do not sum to 1")
    #
    #
    # def testTemporalLinkageFunctionality(self):
    #     """
    #     测试 TemporalLinkage 子层的功能，确保时序链路正确更新。
    #     """
    #     temporal_linkage = addressing.TemporalLinkage(memory_size=MEMORY_SIZE, num_writes=NUM_WRITES,
    #                                                   name='test_temporal_linkage')
    #
    #     write_weights = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], minval=0.0, maxval=1.0)
    #     write_weights /= tf.reduce_sum(write_weights, axis=2, keepdims=True) + 1e-6
    #
    #     prev_linkage = addressing.TemporalLinkageState(
    #         link=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
    #         precedence_weights=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
    #     )
    #
    #     inputs = {
    #         'write_weights': write_weights,
    #         'prev_linkage': prev_linkage
    #     }
    #
    #     new_linkage = temporal_linkage(inputs)
    #
    #     # 检查输出形状
    #     self.assertEqual(new_linkage.link.shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE])
    #     self.assertEqual(new_linkage.precedence_weights.shape, [BATCH_SIZE, NUM_WRITES, MEMORY_SIZE])
    #
    #
    # def testFreenessFunctionality(self):
    #     """
    #     测试 Freeness 子层的功能，确保使用率和自由权重正确计算。
    #     """
    #     freeness = addressing.Freeness(memory_size=MEMORY_SIZE, name='test_freeness')
    #
    #     write_weights = tf.random.uniform([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], minval=0.0, maxval=1.0)
    #     free_gate = tf.random.uniform([BATCH_SIZE, NUM_READS], minval=0.0, maxval=1.0)
    #     read_weights = tf.random.uniform([BATCH_SIZE, NUM_READS, MEMORY_SIZE], minval=0.0, maxval=1.0)
    #     prev_usage = tf.random.uniform([BATCH_SIZE, MEMORY_SIZE], minval=0.0, maxval=1.0)
    #
    #     inputs = {
    #         'write_weights': write_weights,
    #         'free_gate': free_gate,
    #         'read_weights': read_weights,
    #         'prev_usage': prev_usage
    #     }
    #
    #     usage = freeness(inputs)
    #
    #     # 检查输出形状
    #     self.assertEqual(usage.shape, [BATCH_SIZE, MEMORY_SIZE])
    #
    #     # 检查使用率的合理范围
    #     self.assertTrue(tf.reduce_all(usage >= 0.0))
    #     self.assertTrue(tf.reduce_all(usage <= 1.0))
    #
    #
    # def testNumericalStability(self):
    #     """
    #     测试模型在处理极端输入值时的数值稳定性。
    #     """
    #     # 使用非常大的输入值
    #     large_inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE], mean=0.0, stddev=1e6)
    #     large_targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], mean=0.0, stddev=1e6)
    #
    #     with self.assertRaises(tf.errors.InvalidArgumentError):
    #         # 期望在前向传播过程中捕获数值问题
    #         self.module({'inputs': large_inputs, 'prev_state': self.initial_state})
    #
    #     # 使用非常小的输入值
    #     small_inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE], mean=0.0, stddev=1e-6)
    #     small_targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], mean=0.0, stddev=1e-6)
    #
    #     with tf.GradientTape() as tape:
    #         output, _ = self.module({'inputs': small_inputs, 'prev_state': self.initial_state})
    #         loss = tf.reduce_sum(output)
    #
    #     gradients = tape.gradient(loss, [small_inputs] + list(self.module.trainable_variables))
    #
    #     # 检查梯度
    #     for grad, var in zip(gradients, [small_inputs] + list(self.module.trainable_variables)):
    #         var_name = var.name
    #         if grad is not None:
    #             grad_norm = tf.norm(grad).numpy()
    #             self.assertGreater(grad_norm, 1e-12, f"Gradient for '{var_name}' is too small.")
    #             self.assertLess(grad_norm, 1e3, f"Gradient for '{var_name}' is too large.")
    #         else:
    #             self.fail(f"Gradient for '{var_name}' is None.")
    #
    #
    #     def testDifferentLearningRates(self):
    #         """
    #         测试模型在不同学习率下的训练表现。
    #         """
    #         learning_rates = [1e-3, 1e-2, 1e-1, 1.0]
    #
    #         for lr in learning_rates:
    #             print(f"Testing with learning rate: {lr}")
    #             optimizer = tf.optimizers.SGD(learning_rate=lr)
    #
    #             inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
    #             targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
    #
    #             with tf.GradientTape() as tape:
    #                 loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
    #
    #             # 检查梯度
    #             self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))
    #
    #             # 应用梯度
    #             optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))
    #
    #
    # def testGradientClipping(self):
    #     """
    #     测试模型的梯度裁剪功能，确保梯度不会超过指定的范数。
    #     """
    #     inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
    #     targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
    #
    #     optimizer = tf.optimizers.SGD(learning_rate=1.0)
    #     clip_norm = 1.0
    #
    #     with tf.GradientTape() as tape:
    #         loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
    #
    #     # 应用梯度裁剪
    #     clipped_gradients = [tf.clip_by_norm(g, clip_norm) if g is not None else None for g in gradients]
    #
    #     # 检查梯度裁剪
    #     for grad in clipped_gradients:
    #         if grad is not None:
    #             grad_norm = tf.norm(grad).numpy()
    #             self.assertLessEqual(grad_norm, clip_norm + 1e-6, "Gradient norm exceeds clipping threshold")
    #
    #     # 应用裁剪后的梯度
    #     optimizer.apply_gradients(zip(clipped_gradients, self.module.trainable_variables))
    #
    #
    # def testBatchProcessing(self):
    #     """
    #     测试模型在不同批次大小下的处理能力，确保批处理是正确并且高效的。
    #     """
    #     batch_sizes = [2, 16, 32]
    #
    #     for batch_size in batch_sizes:
    #         print(f"Testing with batch size: {batch_size}")
    #
    #         # 创建新的模块实例
    #         self.module = access.MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
    #
    #         # 初始化随机状态
    #         self.initial_state = AccessState(
    #             memory=tf.zeros([batch_size, MEMORY_SIZE, WORD_SIZE], dtype=tf.float32),
    #             read_weights=tf.zeros([batch_size, NUM_READS, MEMORY_SIZE], dtype=tf.float32),
    #             write_weights=tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32),
    #             linkage=addressing.TemporalLinkageState(
    #                 link=tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
    #                 precedence_weights=tf.zeros([batch_size, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
    #             ),
    #             usage=tf.zeros([batch_size, MEMORY_SIZE], dtype=tf.float32)
    #         )
    #
    #         # 创建输入和目标
    #         inputs = tf.random.normal([TIME_STEPS, batch_size, INPUT_SIZE])
    #         targets = tf.random.normal([TIME_STEPS, batch_size, NUM_READS, WORD_SIZE])
    #
    #         loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
    #
    #         # 检查梯度
    #         self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))
    #
    #
    # def testIntegrationWithLargerModel(self):
    #     """
    #     测试 MemoryAccess 模块与更大的模型或系统的集成情况。
    #     """
    #
    #     # 假设有一个简单的 RNN 模型，集成 MemoryAccess
    #     class SimpleRNNWithMemory(tf.keras.Model):
    #         def __init__(self, memory_access):
    #             super(SimpleRNNWithMemory, self).__init__()
    #             self.memory_access = memory_access
    #             self.dense = tf.keras.layers.Dense(NUM_READS * WORD_SIZE, activation=None)
    #
    #         def call(self, inputs, states):
    #             read_output, new_state = self.memory_access({
    #                 'inputs': inputs,
    #                 'prev_state': states
    #             })
    #             output = self.dense(read_output)
    #             return output, new_state
    #
    #     # 创建 MemoryAccess 实例
    #     memory_access = access.MemoryAccess(MEMORY_SIZE, WORD_SIZE, NUM_READS, NUM_WRITES)
    #     model = SimpleRNNWithMemory(memory_access)
    #
    #     # 初始化状态
    #     initial_state = AccessState(
    #         memory=tf.zeros([BATCH_SIZE, MEMORY_SIZE, WORD_SIZE], dtype=tf.float32),
    #         read_weights=tf.zeros([BATCH_SIZE, NUM_READS, MEMORY_SIZE], dtype=tf.float32),
    #         write_weights=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32),
    #         linkage=addressing.TemporalLinkageState(
    #             link=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE, MEMORY_SIZE], dtype=tf.float32),
    #             precedence_weights=tf.zeros([BATCH_SIZE, NUM_WRITES, MEMORY_SIZE], dtype=tf.float32)
    #         ),
    #         usage=tf.zeros([BATCH_SIZE, MEMORY_SIZE], dtype=tf.float32)
    #     )
    #
    #     # 创建输入和目标
    #     inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
    #     targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS * WORD_SIZE])
    #
    #     optimizer = tf.optimizers.Adam()
    #
    #     with tf.GradientTape() as tape:
    #         states = initial_state
    #         outputs = []
    #         for t in range(TIME_STEPS):
    #             input_t = inputs[t]
    #             output, states = model(input_t, states)
    #             outputs.append(output)
    #
    #         output = tf.stack(outputs, axis=0)
    #         loss = tf.reduce_mean(tf.square(output - targets))
    #
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #
    #     # 检查梯度
    #     for grad, var in zip(gradients, model.trainable_variables):
    #         self.assertIsNotNone(grad, f"No gradient provided for variable {var.name}")
    #         grad_norm = tf.norm(grad).numpy()
    #         self.assertGreater(grad_norm, 1e-12, f"Gradient too small for {var.name}")
    #         self.assertLess(grad_norm, 1e3, f"Gradient too large for {var.name}")
    #
    #     # 应用梯度
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #
    # def testDifferentOptimizers(self):
    #     """
    #     测试模型在使用不同优化器时的训练表现。
    #     """
    #     optimizers = {
    #         'SGD': tf.optimizers.SGD(learning_rate=0.1),
    #         'Adam': tf.optimizers.Adam(learning_rate=0.001),
    #         'RMSProp': tf.optimizers.RMSprop(learning_rate=0.001)
    #     }
    #
    #     inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
    #     targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
    #
    #     for opt_name, optimizer in optimizers.items():
    #         print(f"Testing with optimizer: {opt_name}")
    #
    #         with tf.GradientTape() as tape:
    #             loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
    #
    #         # 检查梯度
    #         self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))
    #
    #         # 应用梯度
    #         optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))
    #
    #
    # def testWeightUpdates(self):
    #     """
    #     测试模型权重在训练过程中的更新，确保优化器能够正确地更新权重。
    #     """
    #     optimizer = tf.optimizers.Adam(learning_rate=0.01)
    #
    #     inputs = tf.Variable(np.random.randn(TIME_STEPS, BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)
    #     targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
    #
    #     # 记录初始权重
    #     initial_weights = [var.numpy().copy() for var in self.module.trainable_variables]
    #
    #     with tf.GradientTape() as tape:
    #         output, _ = self.module({'inputs': inputs, 'prev_state': self.initial_state})
    #         loss = tf.reduce_sum(output)
    #
    #     gradients = tape.gradient(loss, self.module.trainable_variables)
    #
    #     # 应用梯度
    #     optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))
    #
    #     # 检查权重是否更新
    #     for initial, var in zip(initial_weights, self.module.trainable_variables):
    #         updated = var.numpy()
    #         self.assertFalse(np.array_equal(initial, updated), f"Variable '{var.name}' did not update.")
    #
    #
    # def testInferencePerformance(self):
    #     """
    #     测试模型在推理阶段的计算速度和内存占用。
    #     """
    #     import time
    #
    #     inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
    #     prev_state = self.initial_state
    #
    #     # 测试多次前向传播以评估平均时间
    #     num_runs = 100
    #     start_time = time.time()
    #
    #     for _ in range(num_runs):
    #         output, _ = self.module({
    #             'inputs': inputs,
    #             'prev_state': prev_state
    #         })
    #
    #     end_time = time.time()
    #     avg_time = (end_time - start_time) / num_runs
    #     print(f"Average inference time per run: {avg_time * 1000:.2f} ms")
    #
    #     # 简单断言，确保推理时间在合理范围内（具体数值根据硬件而定）
    #     self.assertLess(avg_time, 0.1, "Inference time is too high.")
    #
    #
    # def testParallelExecution(self):
    #     """
    #     测试模型在多线程环境下的并行执行能力。
    #     """
    #     import threading
    #
    #     def run_test():
    #         inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
    #         targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
    #         loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
    #         self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))
    #
    #     threads = []
    #     num_threads = 4
    #     for _ in range(num_threads):
    #         thread = threading.Thread(target=run_test)
    #         threads.append(thread)
    #         thread.start()
    #
    #     for thread in threads:
    #         thread.join()
    #
    #
    # def testExceptionHandling(self):
    #     """
    #     测试模型在异常情况下的处理能力，确保其能够优雅地处理错误。
    #     """
    #     # 模拟内存不足（尝试分配过大的张量）
    #     try:
    #         large_memory = tf.random.normal([BATCH_SIZE, 1000000, WORD_SIZE])  # 非常大的内存
    #         inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
    #         with tf.GradientTape() as tape:
    #             loss, gradients = self._run_forward_pass(inputs, targets=tf.random.normal(
    #                 [TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE]), track_gradients=True)
    #     except tf.errors.ResourceExhaustedError:
    #         print("Caught ResourceExhaustedError as expected.")
    #     else:
    #         self.fail("ResourceExhaustedError was not raised when expected.")
    #
    #     # 模拟计算溢出（使用极端大值）
    #     try:
    #         extreme_inputs = tf.constant(1e20, shape=[TIME_STEPS, BATCH_SIZE, INPUT_SIZE], dtype=tf.float32)
    #         targets = tf.constant(1e20, shape=[TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE], dtype=tf.float32)
    #         loss, gradients = self._run_forward_pass(extreme_inputs, targets=targets, track_gradients=True)
    #         tf.debugging.check_numerics(loss, "Loss contains NaNs or Infs")
    #     except tf.errors.InvalidArgumentError:
    #         print("Caught InvalidArgumentError due to numerical instability as expected.")
    #     else:
    #         # 如果没有捕获错误，确保损失中没有 NaN 或 Inf
    #         self.assertFalse(tf.reduce_any(tf.math.is_nan(loss)))
    #         self.assertFalse(tf.reduce_any(tf.math.is_inf(loss)))
    #
    #
    # def testModelSerialization(self):
    #     """
    #     测试模型的序列化和反序列化，确保所有权重和配置被正确保存和恢复。
    #     """
    #     import tempfile
    #     from tensorflow.keras.models import save_model, load_model
    #
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         # 保存模型
    #         save_path = os.path.join(tmpdirname, 'memory_access_model')
    #         save_model(self.module, save_path, save_format='tf')
    #
    #         # 加载模型
    #         loaded_module = load_model(save_path, custom_objects={
    #             'MemoryAccess': access.MemoryAccess,
    #             'AccessState': AccessState,
    #             'TemporalLinkageState': addressing.TemporalLinkageState
    #         })
    #
    #         # 比较原始模型和加载后的模型的权重
    #         for original_var, loaded_var in zip(self.module.trainable_variables, loaded_module.trainable_variables):
    #             original_values = original_var.numpy()
    #             loaded_values = loaded_var.numpy()
    #             self.assertAllClose(original_values, loaded_values, atol=1e-6,
    #                                 msg=f"Mismatch in variable '{original_var.name}' after serialization.")
    #
    #
    # def testDynamicConfigurationChange(self):
    #     """
    #     测试模型在运行时动态改变配置（如内存大小、读取/写入数量）后的行为。
    #     """
    #     # 初始配置
    #     initial_memory_size = MEMORY_SIZE
    #     initial_num_reads = NUM_READS
    #     initial_num_writes = NUM_WRITES
    #
    #     # 创建模块
    #     self.module = access.MemoryAccess(initial_memory_size, WORD_SIZE, initial_num_reads, initial_num_writes)
    #
    #     # 初始化状态
    #     self.initial_state = AccessState(
    #         memory=tf.zeros([BATCH_SIZE, initial_memory_size, WORD_SIZE], dtype=tf.float32),
    #         read_weights=tf.zeros([BATCH_SIZE, initial_num_reads, initial_memory_size], dtype=tf.float32),
    #         write_weights=tf.zeros([BATCH_SIZE, initial_num_writes, initial_memory_size], dtype=tf.float32),
    #         linkage=addressing.TemporalLinkageState(
    #             link=tf.zeros([BATCH_SIZE, initial_num_writes, initial_memory_size, initial_memory_size], dtype=tf.float32),
    #             precedence_weights=tf.zeros([BATCH_SIZE, initial_num_writes, initial_memory_size], dtype=tf.float32)
    #         ),
    #         usage=tf.zeros([BATCH_SIZE, initial_memory_size], dtype=tf.float32)
    #     )
    #
    #     # 运行前向传播
    #     inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
    #     targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, initial_num_reads, WORD_SIZE])
    #     loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
    #     self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))
    #
    #     # 动态改变配置
    #     new_memory_size = 30
    #     new_num_reads = 3
    #     new_num_writes = 4
    #     self.module.memory_size = new_memory_size
    #     self.module.num_reads = new_num_reads
    #     self.module.num_writes = new_num_writes
    #
    #     # 重新初始化模块以应用新配置
    #     self.module = access.MemoryAccess(new_memory_size, WORD_SIZE, new_num_reads, new_num_writes)
    #
    #     # 更新初始状态
    #     self.initial_state = AccessState(
    #         memory=tf.zeros([BATCH_SIZE, new_memory_size, WORD_SIZE], dtype=tf.float32),
    #         read_weights=tf.zeros([BATCH_SIZE, new_num_reads, new_memory_size], dtype=tf.float32),
    #         write_weights=tf.zeros([BATCH_SIZE, new_num_writes, new_memory_size], dtype=tf.float32),
    #         linkage=addressing.TemporalLinkageState(
    #             link=tf.zeros([BATCH_SIZE, new_num_writes, new_memory_size, new_memory_size], dtype=tf.float32),
    #             precedence_weights=tf.zeros([BATCH_SIZE, new_num_writes, new_memory_size], dtype=tf.float32)
    #         ),
    #         usage=tf.zeros([BATCH_SIZE, new_memory_size], dtype=tf.float32)
    #     )
    #
    #     # 运行前向传播
    #     new_targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, new_num_reads, WORD_SIZE])
    #     loss, gradients = self._run_forward_pass(inputs, targets=new_targets, track_gradients=True)
    #     self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))
    #
    #
    # def testLongTermDependencies(self):
    #     """
    #     测试模型在处理具有长期依赖性的序列数据时的性能。
    #     """
    #     long_time_steps = 100  # 增加时间步长以模拟长期依赖
    #     inputs = tf.random.normal([long_time_steps, BATCH_SIZE, INPUT_SIZE])
    #     targets = tf.random.normal([long_time_steps, BATCH_SIZE, NUM_READS, WORD_SIZE])
    #
    #     loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
    #
    #     # 检查梯度
    #     self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))
    #
    #
    # def testDeterministicOutputs(self):
    #     """
    #     测试模型在相同输入和初始状态下是否产生确定性的输出和梯度。
    #     """
    #     tf.random.set_seed(42)
    #     np.random.seed(42)
    #
    #     inputs = tf.Variable(np.random.randn(TIME_STEPS, BATCH_SIZE, INPUT_SIZE), dtype=tf.float32)
    #     targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
    #
    #     # 第一次运行
    #     with tf.GradientTape() as tape1:
    #         output1, _ = self.module({'inputs': inputs, 'prev_state': self.initial_state})
    #         loss1 = tf.reduce_sum(output1)
    #     gradients1 = tape1.gradient(loss1, [inputs] + list(self.module.trainable_variables))
    #
    #     # 重置随机种子
    #     tf.random.set_seed(42)
    #     np.random.seed(42)
    #
    #     # 第二次运行
    #     with tf.GradientTape() as tape2:
    #         output2, _ = self.module({'inputs': inputs, 'prev_state': self.initial_state})
    #         loss2 = tf.reduce_sum(output2)
    #     gradients2 = tape2.gradient(loss2, [inputs] + list(self.module.trainable_variables))
    #
    #     # 比较损失
    #     self.assertAllClose(loss1, loss2, atol=1e-6, msg="Loss values do not match across runs.")
    #
    #     # 比较梯度
    #     for grad1, grad2 in zip(gradients1, gradients2):
    #         if grad1 is not None and grad2 is not None:
    #             self.assertAllClose(grad1, grad2, atol=1e-6, msg="Gradients do not match across runs.")
    #         else:
    #             self.assertEqual(grad1 is None, grad2 is None, msg="Gradient presence mismatch across runs.")
    #
    #
    # def testCustomTrainingLoop(self):
    #     """
    #     测试模型在自定义训练循环中的工作情况。
    #     """
    #     optimizer = tf.optimizers.Adam()
    #     inputs = tf.random.normal([TIME_STEPS, BATCH_SIZE, INPUT_SIZE])
    #     targets = tf.random.normal([TIME_STEPS, BATCH_SIZE, NUM_READS, WORD_SIZE])
    #
    #     # 简单的自定义训练步骤
    #     with tf.GradientTape() as tape:
    #         loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
    #
    #     # 应用梯度
    #     optimizer.apply_gradients(zip(gradients, self.module.trainable_variables))
    #
    #     # 检查梯度
    #     self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))
    #
    #     # 再次运行前向传播，检查是否没有错误
    #     loss, gradients = self._run_forward_pass(inputs, targets=targets, track_gradients=True)
    #     self._check_gradients(gradients, [inputs] + list(self.module.trainable_variables))


if __name__ == '__main__':
    tf.test.main()
