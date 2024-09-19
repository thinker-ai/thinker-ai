import unittest
import numpy as np
import tensorflow as tf
from thinker_ai.agent.memory.humanoid_memory.dnc.repeat_copy import RepeatCopy, DatasetTensors


class TestRepeatCopy(unittest.TestCase):

    def setUp(self):
        # 初始化 RepeatCopy 对象
        self.num_bits = 6
        self.batch_size = 4
        self.min_length = 3
        self.max_length = 5
        self.min_repeats = 2
        self.max_repeats = 3
        self.repeat_copy = RepeatCopy(
            num_bits=self.num_bits,
            batch_size=self.batch_size,
            min_length=self.min_length,
            max_length=self.max_length,
            min_repeats=self.min_repeats,
            max_repeats=self.max_repeats
        )

    def test_call(self):
        """测试 RepeatCopy 类的 call 方法，检查生成的数据格式和内容"""
        inputs = tf.zeros([self.batch_size, self.num_bits])
        dataset: DatasetTensors = self.repeat_copy(inputs)

        # 验证生成的 observations、target 和 mask 的形状
        self.assertEqual(dataset.observations.shape[1], self.batch_size)
        self.assertEqual(dataset.target.shape[1], self.batch_size)

        # 验证 mask 的形状
        expected_time_steps = dataset.observations.shape[0]  # mask 的第一个维度应该是时间步长
        self.assertEqual(dataset.mask.shape[0], expected_time_steps)  # 使用时间步长进行验证
        self.assertEqual(dataset.mask.shape[1], self.batch_size)  # 批次大小应该作为第二个维度

    def test_cost(self):
        """测试 RepeatCopy 的 cost 方法是否能够正确计算损失"""
        inputs = tf.zeros([self.batch_size, self.num_bits])
        dataset: DatasetTensors = self.repeat_copy(inputs)

        # 随机初始化模型输出 logits（模拟预测值）
        logits = tf.random.uniform(shape=dataset.target.shape)

        # 计算损失
        loss = self.repeat_copy.cost(logits, dataset.target, dataset.mask)
        self.assertGreater(loss.numpy(), 0)  # 损失应为正值

    def test_to_human_readable(self):
        """测试 to_human_readable 方法生成的可视化字符串"""
        inputs = tf.zeros([self.batch_size, self.num_bits])
        dataset: DatasetTensors = self.repeat_copy(inputs)

        # 调用 to_human_readable 方法
        readable_str = self.repeat_copy.to_human_readable(dataset)

        # 验证生成的可视化字符串不为空
        self.assertIsNotNone(readable_str)
        self.assertGreater(len(readable_str), 0)  # 验证生成的字符串是否有内容

    def test_normalisation(self):
        """测试 normalisation 和 unnormalisation 的行为"""
        value = 5.0
        norm_val = self.repeat_copy._normalise(value)
        unnorm_val = self.repeat_copy._unnormalise(norm_val)

        # 确保正常化和反正常化之后恢复原始值
        self.assertAlmostEqual(value, unnorm_val, places=5)

    def test_min_length_and_repeats(self):
        """测试 min_length 和 min_repeats 的边界值"""
        inputs = tf.zeros([self.batch_size, self.num_bits])
        repeat_copy = RepeatCopy(
            num_bits=self.num_bits,
            batch_size=self.batch_size,
            min_length=1,  # 最小长度
            max_length=1,
            min_repeats=1,  # 最小重复次数
            max_repeats=1
        )
        dataset: DatasetTensors = repeat_copy(inputs)
        # 验证生成的形状
        self.assertEqual(dataset.observations.shape[0], 1 * (1 + 1) + 3)  # 最小时间步长

    def test_all_ones_input(self):
        """测试全为 1 的输入"""
        inputs = tf.ones([self.batch_size, self.num_bits])
        dataset: DatasetTensors = self.repeat_copy(inputs)
        self.assertEqual(dataset.observations.shape[1], self.batch_size)

    def test_fixed_repeats(self):
        """测试当重复次数固定时，生成的数据是否正确"""
        inputs = tf.zeros([self.batch_size, self.num_bits])
        repeat_copy = RepeatCopy(
            num_bits=self.num_bits,
            batch_size=self.batch_size,
            min_length=4,
            max_length=4,
            min_repeats=2,  # 固定重复次数
            max_repeats=2
        )
        dataset: DatasetTensors = repeat_copy(inputs)
        # 验证生成的数据符合预期
        expected_time_steps = 4 * (2 + 1) + 3  # 计算时间步长
        self.assertEqual(dataset.observations.shape[0], expected_time_steps)

    def test_observations_and_target_consistency(self):
        """测试观察序列和目标序列的一致性"""
        inputs = tf.zeros([self.batch_size, self.num_bits])
        dataset: DatasetTensors = self.repeat_copy(inputs)

        # 提取 observations 中的二进制模式，忽略开始标记和重复次数标记
        obs_pattern = dataset.observations[1:self.max_length + 1, :, :-2]  # 从 1 开始去掉开始标记

        for batch_index in range(self.batch_size):
            num_repeats = int(self.repeat_copy._unnormalise(dataset.observations[-1, batch_index, -1]))

            # 提取 target 中的重复二进制模式
            targ_pattern = dataset.target[self.max_length + 2:self.max_length + 2 + (self.max_length * num_repeats),
                           batch_index, :-1]

            # 验证 observations 的模式与 target 的模式是一致的
            repeated_obs = np.tile(obs_pattern[:, batch_index, :].numpy(), (num_repeats, 1))
            self.assertTrue(
                np.array_equal(repeated_obs, targ_pattern.numpy()),
                f"Batch {batch_index} observations and target patterns are not consistent.\n"
                f"Observations:\n{obs_pattern[:, batch_index, :].numpy()}\n"
                f"Repeated Observations:\n{repeated_obs}\n"
                f"Targets:\n{targ_pattern.numpy()}"
            )
if __name__ == '__main__':
    unittest.main()
