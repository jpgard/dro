# Based on code from https://github.com/tensorflow/cleverhans

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
import math
import numpy as np
import os
import six
import tensorflow as tf
import time
import warnings

from dro.sinha.utils import _ArgsWrapper, batch_indices
from dro.utils.training_utils import get_batch

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class _FlagsWrapper(_ArgsWrapper):
    """
    Wrapper that tries to find missing parameters in TensorFlow FLAGS
    for backwards compatibility.

    Plain _ArgsWrapper should be used instead if the support for FLAGS
    is removed.
    """
    def __getattr__(self, name):
        val = self.args.get(name)
        if val is None:
            warnings.warn('Setting parameters ({}) from TensorFlow FLAGS is '
                          'deprecated.'.format(name))
            val = FLAGS.__getattr__(name)
        return val


def model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    #print(op)
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out


def model_train(sess, x, y, predictions, X_train, Y_train, save=False,
                predictions_adv=None, evaluate=None, verbose=True, args=None,
                dataset_iterator=None,
                nb_batches=None) -> dict:
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param save: boolean controling the save operation
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training
    :param args: dict or argparse `Namespace` object.
                 Should contain `nb_epochs`, `learning_rate`,
                 `batch_size`
                 If save is True, should also contain 'train_dir'
                 and 'filename'.
    :param nb_batches: optional parameter to give number of iterations per epoch;
    otherwise this is inferred from data.
    :return: metrics_dict, a dictionary of metrics.
    """
    args = _FlagsWrapper(args or {})

    # Check that necessary arguments were given (see doc above)
    assert args.nb_epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"

    if save:
        assert args.train_dir, "Directory for save was not given in args dict"
        assert args.filename, "Filename for save was not given in args dict"

    # Define loss
    loss = model_loss(y, predictions)
    if predictions_adv is not None:
        p = 1.0
        loss = ((1-p)*loss + p*model_loss(y, predictions_adv))

    train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

    with sess.as_default():
        if hasattr(tf, "global_variables_initializer"):
            tf.global_variables_initializer().run()
        else:
            sess.run(tf.initialize_all_variables())

        for epoch in six.moves.xrange(args.nb_epochs):
            if verbose:
                print("Epoch " + str(epoch))

            # Compute number of batches
            if not nb_batches:
                nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
                assert nb_batches * args.batch_size >= len(X_train)

            prev = time.time()
            for batch_num in range(nb_batches):
                print("epoch %s step %s" % (epoch, batch_num))
                if dataset_iterator is not None:
                    batch_x, batch_y = get_batch(dataset_iterator,
                                                 args.batch_size)
                else:
                    start, end = batch_indices(
                        batch_num, len(X_train), args.batch_size)
                    batch_x = X_train[start:end]
                    batch_y = Y_train[start:end]

                # TODO(jpgard): perform a check if X_train and Y_train are iterators;
                #  otherwise use the existing code as default.
                # We allow an iterator to be passed; in this case, we slice it first.

                # Perform one training step
                train_step.run(feed_dict={x: batch_x,
                                          y: batch_y})
            cur = time.time()
            if verbose:
                print("\tEpoch took " + str(cur - prev) + " seconds")
            if evaluate is not None:
                metrics_dict = evaluate()

        if save:
            save_path = os.path.join(args.train_dir, args.filename)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print("Completed model training and saved at:" + str(save_path))
        else:
            print("Completed model training.")

    return metrics_dict


def model_eval(sess, x, y, model, X_test, Y_test, args=None, dataset_iterator=None):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """
    args = _FlagsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"

    # Define symbol for accuracy
    # Keras 2.0 categorical_accuracy no longer calculates the mean internally
    # tf.reduce_mean is called in here and is backward compatible with previous
    # versions of Keras
    acc_value = tf.reduce_mean(keras.metrics.categorical_accuracy(y, model))

    # Init result var
    accuracy = 0.0

    # Variable to track size
    n_test = 0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_test)

        for batch in range(nb_batches):
            #if batch % 100 == 0 and batch > 0:
                #print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)
            cur_batch_size = end - start

            if dataset_iterator is not None:
                batch_x, batch_y = get_batch(dataset_iterator,
                                             args.batch_size)
            else:
                batch_x = X_test[start:end]
                batch_y = X_test[start:end]
            n_test += len(batch_x)

            # The last batch may be smaller than all others, so we need to
            # account for variable batch size here
            cur_acc = acc_value.eval(
                feed_dict={x: batch_x,
                           y: batch_y})

            accuracy += (cur_batch_size * cur_acc)

        assert end >= n_test

        # Divide by number of examples to get final value
        accuracy /= n_test

    return accuracy
