import inspect
from typing import Protocol, Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer, MinMaxScaler


class ModelProtocol(Protocol):
    def fit(self, X: Any, y: Any) -> Any:
        ...

    def predict(self, X: Any) -> Any:
        ...

    def evaluate(self, X: Any, y: Any):
        ...


class Predictor:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        # Check if the model requires 'epochs' and 'verbose'
        if 'model' in self.pipeline.named_steps and hasattr(self.pipeline.named_steps['model'], 'fit'):
            self.pipeline.fit(X, y)
        else:
            self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return r2_score(y, predictions)


def _build_sklearn_pipeline(model, degree=3) -> Pipeline:
    # 定义特征变换
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    sin_transformer = FunctionTransformer(np.sin, validate=False)
    cos_transformer = FunctionTransformer(np.cos, validate=False)
    exp_transformer = FunctionTransformer(np.exp, validate=False)

    # 创建特征处理管道
    feature_processing = ColumnTransformer(transformers=[
        ('polynomial', polynomial_features, [0]),
        ('sin', sin_transformer, [0]),
        ('cos', cos_transformer, [0]),
        ('exp', exp_transformer, [0]),
    ])

    # 创建整个处理和模型训练管道
    pipeline = Pipeline(steps=[
        ('min_max_scaler', MinMaxScaler()),
        ('feature_processing', feature_processing),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    return pipeline


# # 使用线性回归
# decision_tree_pipeline = build_model_pipeline(LinearRegression())
# # 使用岭回归
# decision_tree_pipeline = build_model_pipeline(Ridge(alpha=0.5))
# # 使用LASSO回归
# decision_tree_pipeline = build_model_pipeline(Lasso(alpha=0.1))
# # 决策树
# decision_tree_pipeline = build_model_pipeline(DecisionTreeRegressor(), degree=3)
# # 随机森林
# random_forest_pipeline = build_model_pipeline(RandomForestRegressor(n_estimators=100), degree=3)
# 支持向量机（SVM）
# svm_pipeline = build_model_pipeline(SVR(kernel='rbf'), degree=3)
def regress(X, y, model, degree=3) -> float:
    pipeline = _build_sklearn_pipeline(model=model, degree=degree)
    predictor = Predictor(pipeline)
    predictor.fit(X, y)
    score = predictor.evaluate(X, y)
    return score