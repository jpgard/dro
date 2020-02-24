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
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_string('train_dir', './training-logs', 'Training directory')
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

    # Generate training and test data
    X_train, Y_train = generate_simulated_dataset(n=1e6, p=0.5)
    X_test, Y_test = generate_simulated_dataset(n=1e6, p=0.5)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.placeholder(tf.float32, shape=(None, 2))

    # TODO(jpgard): build the model and run some experiments.
    from dro.training.models import logistic_regression_model
    model = logistic_regression_model(n_features=2, n_outputs=2)
    predictions = model(x)
    wrm = WassersteinRobustMethod(model, sess=sess)
    # TODO: tune these parameters and conduct a sensitivity analysis.
    wrm_params = {'eps': 1.3, 'ord': 2, 'y': y, 'steps': 15}
    predictions_adv_wrm = model(wrm.generate(x, **wrm_params))

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args=eval_params)
        print('Test accuracy on legitimate test examples: %0.4f' % accuracy)

        # Accuracy of the model on Wasserstein adversarial examples
        accuracy_adv_wass = model_eval(sess, x, y, predictions_adv_wrm, X_test,
                                       Y_test, args=eval_params)
        print('Test accuracy on Wasserstein examples: %0.4f\n' % accuracy_adv_wass)

    # Train the model
    model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate,
                args=train_params, save=False)
    model.model.save(FLAGS.train_dir + '/' + FLAGS.filename_erm)


if __name__ == "__main__":
    app.run(main)
