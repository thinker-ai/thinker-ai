from keras.models import Sequential
from keras.layers import Dense
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures


def build_keras_pipeline(model_build_fn, input_shape, degree=3, epochs=100, batch_size=10):
    preprocessing_pipeline = Pipeline(steps=[
        ('min_max_scaler', MinMaxScaler()),
        ('polynomial_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('standard_scaler', StandardScaler())
    ])

    return KerasRegressor(build_fn=model_build_fn, preprocessing_pipeline=preprocessing_pipeline,
                          input_shape=input_shape, epochs=epochs, batch_size=batch_size)


def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    return model


class KerasRegressor(BaseEstimator, TransformerMixin):
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

