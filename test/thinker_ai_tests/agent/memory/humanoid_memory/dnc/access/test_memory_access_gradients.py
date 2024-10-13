import unittest
import tensorflow as tf
import numpy as np

from thinker_ai.agent.memory.humanoid_memory.dnc.memory_access import MemoryAccess


# 模拟控制器
class SimpleController(tf.keras.layers.Layer):
    def __init__(self, output_size):
        super(SimpleController, self).__init__()
        self.dense = tf.keras.layers.Dense(output_size, activation='relu')

    def call(self, inputs, training=False):
        return self.dense(inputs)

# 创建测试模型
class TestModel(tf.keras.Model):
    def __init__(self, controller_output_size, memory_access):
        super(TestModel, self).__init__()
        self.controller = SimpleController(controller_output_size)
        self.memory_access = memory_access

    def call(self, inputs, prev_state, training=False):
        controller_output = self.controller(inputs, training=training)
        memory_access_inputs = {
            'inputs': controller_output,
            'prev_state': prev_state
        }
        output = self.memory_access(memory_access_inputs, training=training)
        return output['read_words'], output['final_state']

class MemoryAccessGradientTest(unittest.TestCase):
    def setUp(self):
        # 设置随机种子
        tf.random.set_seed(42)
        np.random.seed(42)

        # 设置模型参数
        self.batch_size = 2
        self.input_size = 32  # 控制器输入尺寸，可以变化
        self.controller_output_size = 64  # 固定的控制器输出尺寸
        self.memory_size = 16
        self.word_size = 8
        self.num_reads = 1
        self.num_writes = 1

        # 实例化 MemoryAccess 模块
        self.memory_access = MemoryAccess(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_reads=self.num_reads,
            num_writes=self.num_writes,
            controller_output_size=self.controller_output_size
        )

        # 创建测试模型
        self.model = TestModel(self.controller_output_size, self.memory_access)

        # 准备数据
        self.controller_input = tf.random.uniform([self.batch_size, self.input_size])
        self.initial_state = self.memory_access.get_initial_state(self.batch_size)
        self.target = tf.random.uniform([self.batch_size, self.num_reads, self.word_size])

        # 定义损失函数和优化器
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

    def test_gradients_exist(self):
        # 梯度存在性测试
        with tf.GradientTape() as tape:
            read_words, final_state = self.model(self.controller_input, self.initial_state, training=False)
            loss = self.loss_fn(self.target, read_words)

        # 获取可训练参数
        trainable_vars = self.model.trainable_variables

        # 计算梯度
        gradients = tape.gradient(loss, trainable_vars)

        # 检查梯度是否为 None，是否存在数值问题
        for var, grad in zip(trainable_vars, gradients):
            self.assertIsNotNone(grad, f"Gradient for variable {var.name} is None")
            tf.debugging.assert_all_finite(grad, f"Gradient for variable {var.name} has NaN or Inf values")

    def test_gradients_numeric_check(self):
        # 设置随机种子
        tf.random.set_seed(42)
        np.random.seed(42)

        # 计算自动微分梯度
        with tf.GradientTape() as tape:
            read_words, final_state = self.model(self.controller_input, self.initial_state, training=False)
            loss = self.loss_fn(self.target, read_words)

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # 选择一个梯度非零的变量和索引
        for var_index, (var, grad) in enumerate(zip(trainable_vars, gradients)):
            grad_np = grad.numpy()
            grad_flat = grad_np.flatten()
            non_zero_indices = np.where(grad_flat != 0)[0]
            if len(non_zero_indices) > 0:
                idx = non_zero_indices[0]
                break
        else:
            raise ValueError("No non-zero gradients found in trainable variables.")

        # 获取变量值
        var_value = var.numpy()

        # 设置扰动值
        epsilon = 1e-4  # 调整 epsilon 值
        perturb = np.zeros_like(var_value)
        perturb_flat = perturb.flatten()
        perturb_flat[idx] = epsilon
        perturb = perturb_flat.reshape(var.shape)

        # 正向扰动
        var.assign_add(perturb)
        read_words_pos, _ = self.model(self.controller_input, self.initial_state, training=False)
        loss_pos = self.loss_fn(self.target, read_words_pos)

        # 反向扰动
        var.assign_sub(2 * perturb)
        read_words_neg, _ = self.model(self.controller_input, self.initial_state, training=False)
        loss_neg = self.loss_fn(self.target, read_words_neg)

        # 恢复变量原值
        var.assign_add(perturb)

        # 计算数值梯度
        numeric_grad = (loss_pos - loss_neg) / (2 * epsilon)

        # 提取自动微分梯度对应的元素
        grad_flat = grad.numpy().flatten()
        auto_grad = grad_flat[idx]

        # 比较数值梯度和自动微分梯度
        np.testing.assert_almost_equal(
            numeric_grad.numpy(),
            auto_grad,
            decimal=4,
            err_msg=f"Numeric gradient and automatic gradient do not match at variable index {var_index}, element index {idx}"
        )

    def test_training_step(self):
        # 简单训练测试
        num_epochs = 10
        loss_history = []

        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                read_words, final_state = self.model(self.controller_input, self.initial_state, training=False)
                loss = self.loss_fn(self.target, read_words)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            loss_history.append(loss.numpy())
            print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

        # 验证损失是否下降
        self.assertLess(loss_history[-1], loss_history[0], "Loss did not decrease during training")

    def test_gradient_clipping(self):
        # 梯度裁剪测试（可选）
        max_gradient_norm = 1.0  # 设置梯度裁剪阈值
        num_epochs = 10
        loss_history = []

        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                read_words, final_state = self.model(self.controller_input, self.initial_state, training=False)
                loss = self.loss_fn(self.target, read_words)
            gradients = tape.gradient(loss, self.model.trainable_variables)

            # 进行梯度裁剪
            gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            loss_history.append(loss.numpy())
            print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

        # 验证损失是否下降
        self.assertLess(loss_history[-1], loss_history[0], "Loss did not decrease during training with gradient clipping")

if __name__ == '__main__':
    unittest.main()
