"""
Train a simple classifier on simulated data.
"""

import numpy as np

import keras
from keras import backend
from keras.models import load_model
from keras.backend import manual_variable_initialization
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags


from dro.sinha.attacks import WassersteinRobustMethod
from dro.sinha.utils_mnist import data_mnist
from dro.sinha.utils_tf import model_train, model_eval
from dro.sinha.utils import cnn_model
from dro.datasets import generate_simulated_dataset

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 25, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_string('train_dir', '.', 'Training directory')
flags.DEFINE_string('filename_erm', 'erm.h5', 'Training directory')
flags.DEFINE_string('filename_wrm', 'wrm.h5', 'Training directory')

train_params = {
    'nb_epochs': FLAGS.nb_epochs,
    'batch_size': FLAGS.batch_size,
    'learning_rate': FLAGS.learning_rate,
}
eval_params = {'batch_size': FLAGS.batch_size}

seed = 12345
np.random.seed(seed)
tf.set_random_seed(seed)


def main(argv):
    keras.layers.core.K.set_learning_phase(1)
    manual_variable_initialization(True)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Generate some test data
    X,y = generate_simulated_dataset(n=1e6, p=0.5)

    # TODO(jpgard): build the model and run some experiments.



if __name__ == "__main__":
    app.run(main)
