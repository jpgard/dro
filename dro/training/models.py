import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras import Sequential


def logistic_regression_model(n_features, n_outputs, activation='elu'):
    """Defines a basic logistic regression model to predict a binary output."""
    model = Sequential()
    model.add(Dense(n_outputs, input_shape=(n_features,), activation=activation))
    return model

from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Activation, Dropout
from keras_vggface.vggface import VGGFace


def vggface2_model(dropout_rate, input_shape=(224, 224, 3), activation='sigmoid'):
    """Build a vggface2 model."""
    # Convolution Features
    vgg_model = VGGFace(include_top=False, input_shape=input_shape)
    # set the vgg_model layers to non-trainable
    for layer in vgg_model.layers:
        layer.trainable = False
    last_layer = vgg_model.get_layer('pool5').output
    # Classification block
    net = Flatten(name='flatten')(last_layer)
    net = Dense(4096, name='fc6')(net)
    net = Activation('relu', name='fc6/relu')(net)
    net = Dropout(rate=dropout_rate)(net)
    net = Dense(256, name='fc7')(net)
    net = Activation('relu', name='fc7/relu')(net)
    net = Dropout(rate=dropout_rate)(net)
    net = Dense(2, name='fc8')(net)
    out = Activation(activation, name='activation/{}'.format(activation))(net)
    custom_vgg_model = Model(vgg_model.input, out)
    return custom_vgg_model

