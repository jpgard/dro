"""
Generate adversarial examples using FGSM
and train a vggface model using adversarial training with Keras.

The original paper can be found at:
https://arxiv.org/abs/1412.6572

# set the gpu
export GPU_ID="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# run the script
export LABEL="Mouth_Open"
export DIR="/projects/grail/jpgard/vggface2/annotated_partitioned_by_label"
export SS=0.025
export EPOCHS=40
python3 scripts/train_vggface2_cleverhans.py \
    --label_name $LABEL \
    --test_dir ${DIR}/test/${LABEL} \
    --train_dir ${DIR}/train/${LABEL} \
    --adv_step_size $SS --epochs $EPOCHS \
    --anno_dir /projects/grail/jpgard/vggface2/anno

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import pandas as pd

from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent, Noise
from cleverhans.compat import flags
from cleverhans.utils_keras import KerasModelWrapper

from dro.training.models import vggface2_model
from dro.utils.training_utils import make_callbacks, make_model_uid
from dro.datasets import ImageDataset
from dro.utils.flags import define_training_flags, define_adv_training_flags
from dro import keys

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS

define_training_flags()
define_adv_training_flags()

flags.DEFINE_bool("train_mnist", False, "whether to train the cleverhans mnist model.")


def get_attack(wrap: KerasModelWrapper, sess: tf.Session):
    """Creates an instance of the attack method specified in flags."""
    return globals()[FLAGS.attack](wrap, sess=sess)


class Report:
    def __init__(self):
        self.results_list = list()
        # Set is_adversarial=True when generating the model_uid so that the adversarial
        # parameters (attack type, epsilon, etc) will be recorded in the uid.
        self.uid = make_model_uid(FLAGS, is_adversarial=True)
        self.metric = keys.ACC  # the name of the metric being recorded

    def add_result(self, val, model, data, phase):
        """Record the results of an experiment."""
        results_entry = (self.uid, self.metric, val, model, data, phase)
        # Check for duplicates; while this is not strictly a problem, it is almost
        # definitely a mistake if duplicate results are being added.
        assert results_entry not in self.results_list, "duplicate results added to report"
        self.results_list.append(results_entry)
        return

    def to_csv(self):
        df = pd.DataFrame(self.results_list, columns=["uid", "metric", "value",
                                                      "model", "data", "phase"])
        fp = "./metrics/{}.csv".format(self.uid)
        print("[INFO] writing results to {}".format(fp))
        print(df)
        df.to_csv(fp, index=False)
        return


def mnist_tutorial(label_smoothing=0.1):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param testing: if true, training error is calculated
    :param label_smoothing: float, amount of label smoothing for cross entropy
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    results = Report()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    # Force TensorFlow to use single thread to improve reproducibility
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)

    if keras.backend.image_data_format() != 'channels_last':
        raise NotImplementedError(
            "this tutorial requires keras to be configured to channels_last format")

    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Obtain Image Parameters

    # Label smoothing
    # TODO(jpgard): implement label smoothing as part of the ImageDataSet class.
    # y_train -= label_smoothing * (y_train - 1. / nb_classes)

    from dro.utils.training_utils import get_n_from_file_pattern, compute_n_train_n_val, \
        steps_per_epoch
    from dro.utils.vggface import make_vgg_file_pattern
    train_file_pattern = make_vgg_file_pattern(FLAGS.train_dir)
    test_file_pattern = make_vgg_file_pattern(FLAGS.test_dir)
    n_test = get_n_from_file_pattern(test_file_pattern)
    n_train_val = get_n_from_file_pattern(train_file_pattern)
    n_train, n_val = compute_n_train_n_val(n_train_val, FLAGS.val_frac)

    train_ds = ImageDataset()
    test_ds = ImageDataset()

    # Create the datasets.
    train_ds.from_files(train_file_pattern, shuffle=True)
    val_ds = train_ds.validation_split(n_val)
    # Preprocess the datasets to create (x,y) tuples.
    preprocess_args = {"repeat_forever": True, "batch_size": FLAGS.batch_size}
    train_ds.preprocess(**preprocess_args)
    val_ds.preprocess(**preprocess_args)

    # No matter whether precomputed train batches are used or not, the test data is
    # taken from a test directory.

    test_ds.from_files(test_file_pattern, shuffle=False)
    test_ds.preprocess(repeat_forever=False, shuffle=False, batch_size=FLAGS.batch_size)

    print("[INFO] {n_train} training observations; {n_val} validation observations"
          "{n_test} testing observations".format(n_train=n_train,
                                                 n_val=n_val,
                                                 n_test=n_test,
                                                 ))

    if not FLAGS.debug:
        steps_per_train_epoch = steps_per_epoch(n_train, FLAGS.batch_size)
        steps_per_val_epoch = steps_per_epoch(n_val, FLAGS.batch_size)
    else:
        print("[INFO] running in debug mode")
        steps_per_train_epoch = 1
        steps_per_val_epoch = 1

    attack_params = {'eps': FLAGS.adv_step_size,
                     'clip_min': 0.,
                     'clip_max': 1.}

    # Set the learning phase to False, following the issue here:
    # https://github.com/tensorflow/cleverhans/issues/1052

    K.set_learning_phase(False)

    if FLAGS.train_base:  # Base model training

        vgg_model_base = vggface2_model(dropout_rate=FLAGS.dropout_rate,
                                        activation='softmax')

        # To be able to call the model in the custom loss, we need to call it once
        # before, see https://github.com/tensorflow/tensorflow/issues/23769
        vgg_model_base(vgg_model_base.input)

        # Initialize the attack object
        wrap = KerasModelWrapper(vgg_model_base)
        attack = get_attack(wrap, sess)
        print("[INFO] using attack {} with params {}".format(FLAGS.attack, attack_params))

        adv_acc_metric = get_adversarial_acc_metric(vgg_model_base, attack, attack_params)

        model_compile_args_base = {
            "optimizer": tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate),
            "loss": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            "metrics": ['accuracy', adv_acc_metric]
        }

        vgg_model_base.compile(**model_compile_args_base)
        vgg_model_base.summary()

        # Shared training arguments for the model fitting.
        train_args = {"steps_per_epoch": steps_per_train_epoch,
                      "epochs": FLAGS.epochs,
                      "validation_steps": steps_per_val_epoch}

        print("[INFO] training base model")
        callbacks_base = make_callbacks(FLAGS, is_adversarial=False)
        vgg_model_base.fit(train_ds.dataset, callbacks=callbacks_base,
                           validation_data=val_ds.dataset, **train_args)

        # Evaluate the accuracy on legitimate and adversarial test examples
        _, acc, adv_acc = vgg_model_base.evaluate(test_ds.dataset)
        results.add_result(acc, model=keys.BASE_MODEL, data=keys.CLEAN_DATA,
                           phase=keys.TEST)
        results.add_result(adv_acc, model=keys.BASE_MODEL, data=keys.ADV_DATA,
                           phase=keys.TEST)
        print('Test accuracy on legitimate examples: %0.4f' % acc)
        print('Test accuracy on perturbed examples: %0.4f\n' % adv_acc)

        # Calculate training error
        _, train_acc, train_adv_acc = vgg_model_base.evaluate(train_ds.dataset,
                                                              steps=steps_per_train_epoch)
        results.add_result(train_acc, model=keys.BASE_MODEL, data=keys.CLEAN_DATA,
                           phase=keys.TRAIN)
        results.add_result(train_adv_acc, model=keys.BASE_MODEL, data=keys.ADV_DATA,
                           phase=keys.TRAIN)

    # Redefine Keras model
    if FLAGS.train_adversarial:
        vgg_model_adv = vggface2_model(dropout_rate=FLAGS.dropout_rate,
                                       activation='softmax')
        vgg_model_adv(vgg_model_adv.input)
        wrap_adv = KerasModelWrapper(vgg_model_adv)
        fgsm_adv = get_attack(wrap_adv, sess=sess)

        # Use a loss function based on legitimate and adversarial examples
        adv_loss_adv = get_adversarial_loss(vgg_model_adv, fgsm_adv, attack_params)
        adv_acc_metric_adv = get_adversarial_acc_metric(vgg_model_adv, fgsm_adv,
                                                        attack_params)

        model_compile_args_adv = {
            "optimizer": tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate),
            "loss": adv_loss_adv,
            "metrics": ['accuracy', adv_acc_metric_adv]
        }

        vgg_model_adv.compile(**model_compile_args_adv)
        print("[INFO] training adversarial model")
        callbacks_adv = make_callbacks(FLAGS, is_adversarial=True)
        vgg_model_adv.fit(train_ds.dataset, callbacks=callbacks_adv,
                          validation_data=val_ds.dataset, **train_args)

        # Evaluate the accuracy on legitimate and adversarial test examples
        _, acc, adv_acc = vgg_model_adv.evaluate(test_ds.dataset)
        results.add_result(acc, model=keys.ADV_MODEL, data=keys.CLEAN_DATA,
                           phase=keys.TEST)
        results.add_result(adv_acc, model=keys.ADV_MODEL, data=keys.ADV_DATA,
                           phase=keys.TEST)
        print('Test acc. w/adversarial training on clean examples: %0.4f' % acc)
        print('Test acc. w/adversarial training on perturbed examples: %0.4f\n' % adv_acc)

        # Calculate training error
        _, train_acc, train_adv_acc = vgg_model_adv.evaluate(train_ds.dataset,
                                                             steps=steps_per_train_epoch)
        results.add_result(train_acc, model=keys.ADV_MODEL, data=keys.CLEAN_DATA,
                           phase=keys.TRAIN)
        results.add_result(train_adv_acc, model=keys.ADV_MODEL, data=keys.ADV_DATA,
                           phase=keys.TRAIN)

        results.to_csv()

    return


def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        print(x_adv)
        print(model.input)
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc


def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return cross_ent + FLAGS.adv_multiplier * cross_ent_adv

    return adv_loss


def main(argv=None):
    from cleverhans_tutorials import check_installation
    check_installation(__file__)

    mnist_tutorial()


if __name__ == '__main__':
    tf.app.run()
