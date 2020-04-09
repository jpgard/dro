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
export DIR="/projects/grail/jpgard/vggface2"
export SS=0.025
export EPOCHS=40

python3 scripts/train_vggface2_cleverhans.py \
    --label_name $LABEL \
    --test_dir ${DIR}/annotated_partitioned_by_label/test/${LABEL} \
    --train_dir ${DIR}/annotated_partitioned_by_label/train/${LABEL} \
    --epochs $EPOCHS \
    --attack FastGradientMethod \
    --attack_params "{\"eps\": $SS, \"clip_min\": null, \"clip_max\": null}" \
    --adv_multiplier 0.2 \
    --anno_dir ${DIR}/anno

python3 scripts/train_vggface2_cleverhans.py \
    --label_name $LABEL \
    --test_dir ${DIR}/annotated_partitioned_by_label/test/${LABEL} \
    --train_dir ${DIR}/annotated_partitioned_by_label/train/${LABEL} \
    --epochs $EPOCHS \
    --attack BasicIterativeMethod \
    --attack_params "{\"eps\": $SS, \"nb_iter\": 8, \"eps_iter\": 0.004}" \
    --adv_multiplier 0.2 \
    --anno_dir ${DIR}/anno

python3 scripts/train_vggface2_cleverhans.py \
    --label_name $LABEL \
    --test_dir ${DIR}/annotated_partitioned_by_label/test/${LABEL} \
    --train_dir ${DIR}/annotated_partitioned_by_label/train/${LABEL} \
    --epochs $EPOCHS \
    --attack IterativeFastGradientMethod \
    --attack_params "{\"eps\": $SS, \"nb_iter\": 8, \"eps_iter\": 0.004, \"clip_min\": null, \"clip_max\": null}" \
    --adv_multiplier 0.2 \
    --anno_dir ${DIR}/anno
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from cleverhans.compat import flags

from dro.training.models import vggface2_model
from dro.utils.reports import Report
from dro.utils.training_utils import make_callbacks, get_n_from_file_pattern, \
    compute_n_train_n_val, \
    steps_per_epoch
from dro.datasets import ImageDataset
from dro.utils.flags import define_training_flags, define_adv_training_flags
from dro.utils.cleverhans import get_attack, get_adversarial_acc_metric, \
    get_adversarial_loss, attack_params_from_flags, get_model_compile_args
from dro import keys
from dro.utils.vggface import make_vgg_file_pattern

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS

define_training_flags()
define_adv_training_flags(cleverhans=True)

flags.DEFINE_bool("train_mnist", False, "whether to train the cleverhans mnist model.")


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
    results = Report(FLAGS)

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

    attack_params = attack_params_from_flags(FLAGS)

    # Set the learning phase to False, following the issue here:
    # https://github.com/tensorflow/cleverhans/issues/1052

    K.set_learning_phase(False)

    if FLAGS.train_base:  # Base model training

        vgg_model_base = vggface2_model(dropout_rate=FLAGS.dropout_rate,
                                        activation='softmax')

        # Initialize the attack object
        attack = get_attack(FLAGS, vgg_model_base, sess)
        print("[INFO] using attack {} with params {}".format(FLAGS.attack, attack_params))
        adv_acc_metric = get_adversarial_acc_metric(vgg_model_base, attack, attack_params)
        model_compile_args_base = get_model_compile_args(
            FLAGS, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            adv_acc_metric=adv_acc_metric)

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
        results.add_result({"metric": keys.ACC,
                            "value": acc,
                            "model": keys.BASE_MODEL,
                            "data": keys.CLEAN_DATA,
                            "phase": keys.TEST})
        results.add_result({"metric": keys.ACC,
                            "value": adv_acc,
                            "model": keys.BASE_MODEL,
                            "data": keys.ADV_DATA,
                            "phase": keys.TEST})


        print('Test accuracy on legitimate examples: %0.4f' % acc)
        print('Test accuracy on perturbed examples: %0.4f\n' % adv_acc)

        # Calculate training error
        _, train_acc, train_adv_acc = vgg_model_base.evaluate(train_ds.dataset,
                                                              steps=steps_per_train_epoch)
        results.add_result({"metric": keys.ACC,
                            "value": train_acc,
                            "model": keys.BASE_MODEL,
                            "data": keys.CLEAN_DATA,
                            "phase": keys.TRAIN})
        results.add_result({"metric": keys.ACC,
                            "value": train_adv_acc,
                            "model": keys.BASE_MODEL,
                            "data": keys.ADV_DATA,
                            "phase": keys.TRAIN})

    # Redefine Keras model
    if FLAGS.train_adversarial:
        vgg_model_adv = vggface2_model(dropout_rate=FLAGS.dropout_rate,
                                       activation='softmax')
        vgg_model_adv(vgg_model_adv.input)
        attack = get_attack(FLAGS, vgg_model_adv, sess=sess)

        # Use a loss function based on legitimate and adversarial examples
        adv_loss_adv = get_adversarial_loss(vgg_model_adv, attack, attack_params,
                                            FLAGS.adv_multiplier)
        adv_acc_metric_adv = get_adversarial_acc_metric(vgg_model_adv, attack,
                                                        attack_params)

        model_compile_args_adv = get_model_compile_args(
            FLAGS, loss=adv_loss_adv, adv_acc_metric=adv_acc_metric_adv)

        vgg_model_adv.compile(**model_compile_args_adv)
        print("[INFO] training adversarial model")
        callbacks_adv = make_callbacks(FLAGS, is_adversarial=True)
        vgg_model_adv.fit(train_ds.dataset, callbacks=callbacks_adv,
                          validation_data=val_ds.dataset, **train_args)

        # Evaluate the accuracy on legitimate and adversarial test examples
        _, acc, adv_acc = vgg_model_adv.evaluate(test_ds.dataset)
        results.add_result({"metric": keys.ACC,
                            "value": acc,
                            "model": keys.ADV_MODEL,
                            "data": keys.CLEAN_DATA,
                            "phase": keys.TEST})
        results.add_result({"metric": keys.ACC,
                            "value": adv_acc,
                            "model": keys.ADV_MODEL,
                            "data": keys.ADV_DATA,
                            "phase": keys.TEST})

        print('Test acc. w/adversarial training on clean examples: %0.4f' % acc)
        print('Test acc. w/adversarial training on perturbed examples: %0.4f\n' % adv_acc)

        # Calculate training error
        _, train_acc, train_adv_acc = vgg_model_adv.evaluate(train_ds.dataset,
                                                             steps=steps_per_train_epoch)
        results.add_result({"metric": keys.ACC,
                            "value": train_acc,
                            "model": keys.ADV_MODEL,
                            "data": keys.CLEAN_DATA,
                            "phase": keys.TRAIN})
        results.add_result({"metric": keys.ACC,
                            "value": train_adv_acc,
                            "model": keys.ADV_MODEL,
                            "data": keys.ADV_DATA,
                            "phase": keys.TRAIN})

        results.to_csv()

    return


def main(argv=None):
    from cleverhans_tutorials import check_installation
    check_installation(__file__)

    mnist_tutorial()


if __name__ == '__main__':
    tf.app.run()
