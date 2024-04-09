import unittest

import numpy as np
from thinker_ai.machine_learning.keras_regressor import regress, build_keras_model


class TestKerasRegressor(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X = np.arange(1, 101).reshape(-1, 1)
        # 数据不带噪声
        self.y = 0.05 * self.X.squeeze() ** 2 + np.sin(self.X.squeeze()) - np.cos(self.X.squeeze()) + np.exp(
            self.X.squeeze() * 0.05)
        # 数据带噪声
        self.y_noise = self.y + np.random.normal(0, 20, size=self.X.shape[0])

    def test_regress(self):
        score = regress(self.X, self.y, build_keras_model, epochs=200)
        print(f"Keras Model, R^2 Score: {score:.3f}")
        self.assertGreaterEqual(score, 0.8, "Keras model should fit the data well")

    def test_regress_with_noise(self):
        score_noise = regress(self.X, self.y_noise, build_keras_model, epochs=200)
        print(f"Keras Model, R^2 Score: {score_noise:.3f}")
        self.assertGreaterEqual(score_noise, 0.8, "Keras model should fit the data well")


# 运行测试
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
