import os.path as osp

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Activation, Dropout
from keras_vggface.vggface import VGGFace

FC_SIZES = [4096, 256, 2]


def logistic_regression_model(n_features, n_outputs, activation='elu'):
    """Defines a basic logistic regression model to predict a binary output."""
    model = Sequential()
    model.add(Dense(n_outputs, input_shape=(n_features,), activation=activation))
    return model


def add_classification_block(input_model: Model, fc_sizes: list, activation: str,
                             dropout_rate: float,
                             last_layer_name: str = None) -> Model:
    """
    Construct a new Keras model from the existing model by adding fully-connected
    layers of the specified sizes, and adding an activation layer of the specified
    type.
    :param input_model: the model to use.
    :param fc_sizes: list of integers of the fully-connected layer sizes to use.
    :param activation: string, the name of the activation function to use; this is
    passed to a tf.keras.layers.Activation constructor.
    :param dropout_rate: the dropout rate to use between the layers. The same dropout
    rate is used for every dropout layer; dropout is applied after activation.
    :param last_layer_name: optional, the string name of the last layer. Otherwise,
    the final layer of the model is used by default.
    """
    if not last_layer_name:
        last_layer_name = input_model.layers[-1].name
    last_layer = input_model.get_layer(last_layer_name).output
    # Build the classification block
    net = Flatten(name='flatten')(last_layer)
    for fc_nodes in fc_sizes[:-1]:
        net = Dense(fc_nodes)(net)
        net = Activation('relu')(net)
        net = Dropout(rate=dropout_rate)(net)
    # For the final layer, we do not use dropout.
    net = Dense(fc_sizes[-1])(net)
    out = Activation(activation, name='activation/{}'.format(activation))(net)
    new_model = Model(input_model.input, out)
    return new_model


def vggface2_model(dropout_rate, input_shape=(224, 224, 3), activation='sigmoid'):
    """Build a vggface2 model."""
    # Convolution Features
    vgg_model = VGGFace(include_top=False, input_shape=input_shape)
    # set the vgg_model layers to non-trainable
    for layer in vgg_model.layers:
        layer.trainable = False
    # last_layer = vgg_model.get_layer('pool5').output
    # Classification block
    custom_vgg_model = add_classification_block(vgg_model, fc_sizes=FC_SIZES,
                                                activation=activation,
                                                dropout_rate=dropout_rate,
                                                last_layer_name='pool5')
    # net = Flatten(name='flatten')(last_layer)
    # net = Dense(4096, name='fc6')(net)
    # net = Activation('relu', name='fc6/relu')(net)
    # net = Dropout(rate=dropout_rate)(net)
    # net = Dense(256, name='fc7')(net)
    # net = Activation('relu', name='fc7/relu')(net)
    # net = Dropout(rate=dropout_rate)(net)
    # net = Dense(2, name='fc8')(net)
    # out = Activation(activation, name='activation/{}'.format(activation))(net)
    # custom_vgg_model = Model(vgg_model.input, out)
    return custom_vgg_model


def facenet_model(dropout_rate, activation='sigmoid'):
    cwd, _ = osp.split(__file__)
    facenet_base = load_model(osp.join(cwd, "facenet", "facenet_keras.h5"))
    for layer in facenet_base.layers:
        layer.trainable = False
    last_layer_name = facenet_base.layers[-1].name
    custom_facenet_model = add_classification_block(facenet_base, fc_sizes=FC_SIZES,
                                                    activation=activation,
                                                    dropout_rate=dropout_rate,
                                                    last_layer_name=last_layer_name)
    return custom_facenet_model
