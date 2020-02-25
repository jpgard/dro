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
from dro.sinha.utils_tf import model_train, model_eval
from dro.datasets import generate_simulated_dataset

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 5, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_integer('num_samples', 10 ** 6, 'Number of samples to use in simulation')
flags.DEFINE_multi_float('pos_prob', [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                         'Probability of positive class in simulated data')
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


def model_eval_fn(sess, x, y, predictions, predictions_adv, X_test, Y_test, eval_params):
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args=eval_params)
    print('Test accuracy on legitimate test examples: %0.4f' % accuracy)

    # Accuracy of the model on Wasserstein adversarial examples
    accuracy_adv_wass = model_eval(sess, x, y, predictions_adv, X_test,
                                   Y_test, args=eval_params)
    print('Test accuracy on Wasserstein examples: %0.4f\n' % accuracy_adv_wass)
    # TODO(jpgard): return the accuracies and capture this when called in
    #  model_train().


def main(argv):
    keras.layers.core.K.set_learning_phase(1)
    manual_variable_initialization(True)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Dictionary to store experimental results; {(n,p) : accuracy}
    results = dict()
    n = FLAGS.num_samples
    for p in FLAGS.pos_prob:
        # Generate training and test data; note that the expected accuracy of the optimal
        # linear classifier on this data is approximately 0.6915.
        X_train, Y_train = generate_simulated_dataset(n, p)
        X_test, Y_test = generate_simulated_dataset(n, p)

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

        from functools import partial
        evaluate = partial(model_eval_fn, sess, x, y, predictions, predictions_adv_wrm,
                           X_test, Y_test, eval_params)

        # Train the model
        model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate,
                    args=train_params, save=False)
        # model.save(FLAGS.train_dir + '/' + FLAGS.filename_erm)

        print("Repeating the process, using Wasserstein adversarial training")
        model_adv = logistic_regression_model(n_features=2, n_outputs=2)
        predictions_adv = model_adv(x)
        wrm2 = WassersteinRobustMethod(model_adv, sess=sess)
        predictions_adv_adv_wrm = model_adv(wrm2.generate(x, **wrm_params))

        evaluate_adv = partial(model_eval_fn, sess, x, y, predictions_adv,
                               predictions_adv_adv_wrm,
                               X_test, Y_test, eval_params)

        model_train(sess, x, y, predictions_adv_adv_wrm, X_train, Y_train,
                    predictions_adv=predictions_adv_adv_wrm, evaluate=evaluate_adv,
                    args=train_params, save=False)
        # model_adv.save(FLAGS.train_dir + '/' + FLAGS.filename_wrm)
        results[(n, p)] = []


if __name__ == "__main__":
    app.run(main)
