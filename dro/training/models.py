import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, MaxPool3D, Input
from tensorflow.keras import Sequential


def logistic_regression_model(n_features, n_outputs, activation='elu'):
    """Defines a basic logistic regression model to predict a binary output."""
    model = Sequential()
    model.add(Dense(n_outputs, input_shape=(n_features,), activation=activation))
    return model


def facenet_model():
    """A partial implementation of the FaceNet model."""
    model = Sequential()
    model.add(Conv2D(64, (7, 7), strides=2))
    model.add(MaxPool2D((3, 3), strides=2))
    # TODO(jpgard): add response normalization layers here using
    #  tf.nn.local_response_normalization,
    #  see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn
    #  /local_response_normalization .
    model.add(Conv2D(64, (5, 5), strides=2))
    model.add(MaxPool2D((3, 3), strides=2))
    model.add(Conv2D(64, (3, 3), strides=2))
    model.add(MaxPool2D((3, 3), strides=2))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(1, activation="sigmoid"))
    return model


def facenet_model_functional():
    """A partial implementation of the FaceNet model using the functional API."""
    # TODO(jpgard): implement this as subclass of model class; see
    # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model

    # TODO(jpgard): fully replicate facenet model here.
    inputs = tf.keras.Input(shape=(218, 178, 3))
    x = Conv2D(64, (7, 7), strides=2)(inputs)
    x = MaxPool2D((3, 3), strides=2)(x)
    x = tf.nn.local_response_normalization(x)
    x = Conv2D(64, (1, 1), strides=1)(x)  # conv2a
    x = Conv2D(64, (3, 3), strides=1)(x)  # conv2
    x = tf.nn.local_response_normalization(x)
    x = MaxPool2D((3, 3), strides=2)(x)
    x = Flatten()(x)
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(x)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
