"""
Train a simple classifier on simulated data.
"""

import numpy as np
import os
from functools import partial

import pandas as pd
import keras
from keras import backend
from keras.backend import manual_variable_initialization
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from dro.sinha.attacks import WassersteinRobustMethod
from dro.sinha.utils_tf import model_train, model_eval
from dro.datasets import generate_simulated_dataset
from dro import keys
from dro.training.models import logistic_regression_model

FLAGS = flags.FLAGS

# training parameters
flags.DEFINE_integer('nb_epochs', 5, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')

# the wrm parameters
flags.DEFINE_float('wrm_eps', 1.3, 'epsilon value to use for Wasserstein robust method')
flags.DEFINE_integer('wrm_ord', 2, 'order of norm to use in Wasserstein robust method')
flags.DEFINE_integer('wrm_steps', 15, 'number of steps to use in Wasserstein robus method')

# simulation parameters
flags.DEFINE_integer('num_samples', 10 ** 6, 'Number of samples to use in simulation')
flags.DEFINE_multi_float('pos_prob', [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                         'Probability of positive class in simulated data')
flags.DEFINE_string('train_dir', './training-logs', 'Training directory')
flags.DEFINE_string('metrics_dir', './metrics', 'Metrics directory')
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
    metrics_dict = {keys.ACC: accuracy, keys.ACC_ADV_W: accuracy_adv_wass}
    return metrics_dict


def run_simulation_experiment(n, p, sess, adversarial_training=False, save_model=False):
    """Run a simulation experiment with the specified parameters."""
    # Generate training and test data; note that the expected accuracy of the optimal
    # linear classifier on this data is approximately 0.6915.
    X_train, Y_train = generate_simulated_dataset(n, p)
    X_test, Y_test = generate_simulated_dataset(n, p)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.placeholder(tf.float32, shape=(None, 2))
    # TODO: tune these parameters and conduct a sensitivity analysis.
    wrm_params = {'eps': FLAGS.wrm_eps, 'ord': FLAGS.wrm_ord, 'y': y,
                  'steps': FLAGS.wrm_steps}
    # Define the TensorFlow graph.
    model = logistic_regression_model(n_features=2, n_outputs=2)
    predictions = model(x)
    wrm = WassersteinRobustMethod(model, sess=sess)
    predictions_adv_wrm = model(wrm.generate(x, **wrm_params))
    eval_fn = partial(model_eval_fn, sess, x, y, predictions, predictions_adv_wrm,
                      X_test, Y_test, eval_params)
    if adversarial_training:  # use the adversarial output for training
        model_train_fn = partial(model_train,
                                 sess, x, y, predictions_adv_wrm, X_train, Y_train,
                                 evaluate=eval_fn, args=train_params, save=False)
    else:
        model_train_fn = partial(model_train,
                                 sess, x, y, predictions, X_train, Y_train,
                                 evaluate=eval_fn, args=train_params, save=False)
    metrics = model_train_fn()
    if save_model:
        model.save(FLAGS.train_dir + '/' + FLAGS.filename_wrm)
    return metrics


def main(argv):
    keras.layers.core.K.set_learning_phase(1)
    manual_variable_initialization(True)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Dictionary to store experimental results; {(n,p, is_adversarial) : accuracy}
    results = list()
    n = FLAGS.num_samples
    for p in FLAGS.pos_prob:
        for is_adversarial in (True, False):
            metrics = run_simulation_experiment(
                n, p, sess, adversarial_training=is_adversarial)
            results.append(
                (n, p, is_adversarial, metrics[keys.ACC], metrics[keys.ACC_ADV_W])
            )
    metrics_df = pd.DataFrame(
        results,
        columns=["n", "p", "is_adversarial", keys.ACC, keys.ACC_ADV_W]
    )
    metrics_df.to_csv(os.path.join(FLAGS.metrics_dir, "metrics.csv"), index=False)


if __name__ == "__main__":
    app.run(main)
