import numpy as np
import unittest

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from thinker_ai.machine_learning.sklearn_regression_analyzer import regress


class TestRegressionModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.days = np.arange(1, 101).reshape(-1, 1)
        # 数据不带噪声
        self.y_linear = 2.5 * self.days.squeeze()
        self.y_poly = 0.05 * self.days.squeeze() ** 2 + 2 * self.days.squeeze() + 5
        # 数据带噪声
        self.y_linear_noise = self.y_linear + np.random.normal(0, 10, size=self.days.shape[0])
        self.y_poly_noise = self.y_poly + np.random.normal(0, 20, size=self.days.shape[0])

    def test_linear_regression(self):
        # 线性回归 - 不带噪声
        score = regress(self.days, self.y_linear, LinearRegression())
        self.assertGreaterEqual(score, 0.99, "Linear regression should fit perfectly to linear data without noise")

        # 线性回归 - 带噪声
        score_noise = regress(self.days, self.y_linear_noise, LinearRegression())
        self.assertGreaterEqual(score_noise, 0.8, "Linear regression should fit well to linear data with noise")



    def test_ridge_regression(self):
        # 岭回归 - 不带噪声
        score = regress(self.days, self.y_poly, Ridge(alpha=0.5))
        self.assertGreaterEqual(score, 0.99, "Ridge regression should fit perfectly to polynomial data without noise")

        # 岭回归 - 带噪声
        score_noise = regress(self.days, self.y_poly_noise, Ridge(alpha=0.5))
        self.assertGreaterEqual(score_noise, 0.8, "Ridge regression should fit well to polynomial data with noise")

    def test_lasso_regression(self):
        # LASSO回归 - 不带噪声
        score = regress(self.days, self.y_poly, Lasso(alpha=0.1))
        self.assertGreaterEqual(score, 0.95,
                                "LASSO regression should fit well to high-degree polynomial data without noise")

        # LASSO回归 - 带噪声
        score_noise = regress(self.days, self.y_poly_noise, Lasso(alpha=0.1))
        self.assertGreater(score_noise, 0.7,
                           "LASSO regression should reasonably fit to high-degree polynomial data with noise")

    def test_all_regression(self):
        np.random.seed(42)
        X = np.arange(1, 101).reshape(-1, 1)
        y = 0.05 * X.squeeze() ** 2 + np.sin(X.squeeze()) - np.cos(X.squeeze()) + np.exp(
            X.squeeze() * 0.05)
        for model in [LinearRegression(), Ridge(alpha=0.5), Lasso(alpha=0.1)]:
            score = regress(X, y, model)
            print(f"Model: {model.__class__.__name__}, R^2 Score: {score:.3f}")
            self.assertGreaterEqual(score, 0.8, f"{model.__class__.__name__} should fit the data well")


# 运行测试
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
