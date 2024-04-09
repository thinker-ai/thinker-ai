from keras.models import Sequential
from keras.layers import Dense
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures


class KerasRegressorWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, build_fn, preprocessing_pipeline, epochs=100, batch_size=10, **kwargs):
        self.model = build_fn(**kwargs)
        self.preprocessing_pipeline = preprocessing_pipeline
        self.epochs = epochs
        self.batch_size = batch_size
        self.kwargs = kwargs

    def fit(self, X, y):
        X_transformed = self.preprocessing_pipeline.fit_transform(X)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_transformed, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        X_transformed = self.preprocessing_pipeline.transform(X)
        return self.model.predict(X_transformed)


def build_keras_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    # 确保在这里调用了compile方法，并正确配置了优化器和损失函数
    model.compile(optimizer='adam', loss='mse')
    return model


def build_keras_pipeline(model_build_fn, input_shape, degree=3, epochs=100, batch_size=10):
    preprocessing_pipeline = Pipeline(steps=[
        ('min_max_scaler', MinMaxScaler()),
        ('polynomial_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('standard_scaler', StandardScaler())
    ])

    return KerasRegressorWrapper(build_fn=model_build_fn, preprocessing_pipeline=preprocessing_pipeline,
                                 input_shape=input_shape, epochs=epochs, batch_size=batch_size)


# 定义regress方法，适用于Keras模型
def regress(X, y, build_fn, epochs=100):
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建并训练模型
    input_shape = X_train.shape[1]
    model = build_fn(input_shape)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    # 预测并计算R^2分数
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    return score