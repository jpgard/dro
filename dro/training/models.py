import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import Sequential


def logistic_regression_model(n_features, n_outputs, activation='elu'):
    """Defines a basic logistic regression model to predict a binary output."""
    model = Sequential()
    model.add(Dense(n_outputs, input_shape=(n_features,), activation=activation))
    return model
