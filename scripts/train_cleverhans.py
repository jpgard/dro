"""
Train a base model with standard training, and an adversarial model using adversarial
training, with Keras.

# set the gpu
export GPU_ID="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# run the script
export LABEL="Mouth_Open"
export DIR="/projects/grail/jpgard/vggface2"
export SS=0.025
export EPOCHS=40
export ADV_MULTIPLIER=0.2
export MODEL_TYPE="vggface2"

# Usage with FGSM
python3 scripts/train_cleverhans.py \
    --label_name $LABEL \
    --test_dir ${DIR}/annotated_partitioned_by_label/test/${LABEL} \
    --train_dir ${DIR}/annotated_partitioned_by_label/train/${LABEL} \
    --epochs $EPOCHS \
    --attack FastGradientMethod \
    --attack_params "{\"eps\": $SS, \"clip_min\": null, \"clip_max\": null}" \
    --adv_multiplier $ADV_MULTIPLIER \
    --anno_dir ${DIR}/anno \
    --model_type $MODEL_TYPE

# Usage with Iterative FGSM
python3 scripts/train_cleverhans.py \
    --label_name $LABEL \
    --test_dir ${DIR}/annotated_partitioned_by_label/test/${LABEL} \
    --train_dir ${DIR}/annotated_partitioned_by_label/train/${LABEL} \
    --epochs $EPOCHS \
    --attack IterativeFastGradientMethod \
    --attack_params "{\"eps\": $SS, \"nb_iter\": 8, \"eps_iter\": 0.004, \"clip_min\":
    null, \"clip_max\": null}" \
    --adv_multiplier $ADV_MULTIPLIER \
    --anno_dir ${DIR}/anno \
    --model_type $MODEL_TYPE

# Usage with Randomized FGSM (truncated normal)
python3 scripts/train_cleverhans.py \
    --label_name $LABEL \
    --test_dir ${DIR}/annotated_partitioned_by_label/test/${LABEL} \
    --train_dir ${DIR}/annotated_partitioned_by_label/train/${LABEL} \
    --epochs $EPOCHS \
    --attack RandomizedFastGradientMethod \
    --attack_params "{\"eps_stddev\": 0.03125, \"clip_min\": null, \"clip_max\": null}" \
    --adv_multiplier $ADV_MULTIPLIER \
    --anno_dir ${DIR}/anno \
    --model_type $MODEL_TYPE

# Usage with Randomized FGSM (beta)
python3 scripts/train_cleverhans.py \
    --label_name $LABEL \
    --test_dir ${DIR}/annotated_partitioned_by_label/test/${LABEL} \
    --train_dir ${DIR}/annotated_partitioned_by_label/train/${LABEL} \
    --epochs $EPOCHS \
    --attack RandomizedFastGradientMethodBeta \
    --attack_params "{\"alpha\": 1, \"beta\": 100, \"clip_min\": null, \"clip_max\": null}" \
    --adv_multiplier $ADV_MULTIPLIER \
    --anno_dir ${DIR}/anno \
    --model_type $MODEL_TYPE

# Usage with Staib et al Fast Distributionally-Robust Method. We use 15 iterations and
set eps_iter to eps / nb_iter*1.5 as in staib et al persistent_epsilons_experiment.py.
python3 scripts/train_cleverhans.py \
    --label_name $LABEL \
    --test_dir ${DIR}/annotated_partitioned_by_label/test/${LABEL} \
    --train_dir ${DIR}/annotated_partitioned_by_label/train/${LABEL} \
    --epochs $EPOCHS \
    --attack FrankWolfeDistributionallyRobustMethod \
    --attack_params "{\"eps\": 0.025, \"nb_iter\": 15, \"eps_iter\": 0.0025, \"clip_min\": null, \"clip_max\": null}" \
    --adv_multiplier $ADV_MULTIPLIER \
    --anno_dir ${DIR}/anno \
    --model_type $MODEL_TYPE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from absl import flags

# from cleverhans.compat import flags

from dro.utils.reports import Report
from dro.training.training_utils import get_n_from_file_pattern, \
    compute_n_train_n_val, steps_per_epoch
from dro.training.callbacks import make_callbacks
from dro.datasets import ImageDataset
from dro.utils.flags import define_training_flags, define_adv_training_flags, \
    get_model_compile_args, get_model_from_flags, get_model_img_shape_from_flags
from dro.utils.cleverhans import get_attack, get_adversarial_acc_metric, \
    get_adversarial_loss, attack_params_from_flags
from dro import keys
from dro.utils.vggface import make_vgg_file_pattern

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS

define_training_flags()
define_adv_training_flags(cleverhans=True)

flags.DEFINE_bool("train_mnist", False, "whether to train the cleverhans mnist model.")


def get_data_type_and_metric_from_name(name, sep="_"):
    """Helper function which uses the metric name to determine whether the metric
    operates on clean or adversarial data, and trims the metric name if needed."""
    if "adv" in name:
        data = keys.ADV_DATA
        metric_name = name.split(sep)[1]
    else:
        data = keys.CLEAN_DATA
        metric_name = name
    return data, metric_name


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

    train_file_pattern = make_vgg_file_pattern(FLAGS.train_dir)
    test_file_pattern = make_vgg_file_pattern(FLAGS.test_dir)
    n_test = get_n_from_file_pattern(test_file_pattern)
    n_train_val = get_n_from_file_pattern(train_file_pattern)
    n_train, n_val = compute_n_train_n_val(n_train_val, FLAGS.val_frac)
    img_shape = get_model_img_shape_from_flags(FLAGS)

    train_ds = ImageDataset(img_shape)
    test_ds = ImageDataset(img_shape)

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

    steps_per_train_epoch = steps_per_epoch(n_train, FLAGS.batch_size, FLAGS.debug)
    steps_per_val_epoch = steps_per_epoch(n_val, FLAGS.batch_size, FLAGS.debug)

    attack_params = attack_params_from_flags(FLAGS)

    # Set the learning phase to False, following the issue here:
    # https://github.com/tensorflow/cleverhans/issues/1052

    K.set_learning_phase(False)

    # Shared training arguments for the model fitting.
    train_args = {"steps_per_epoch": steps_per_train_epoch,
                  "epochs": FLAGS.epochs,
                  "validation_steps": steps_per_val_epoch}

    if FLAGS.train_base:  # Base model training

        model_base = get_model_from_flags(FLAGS)

        # Initialize the attack object
        attack = get_attack(FLAGS.attack, model_base, sess)
        print("[INFO] using attack {} with params {}".format(FLAGS.attack, attack_params))
        adv_acc_metric = get_adversarial_acc_metric(model_base, attack, attack_params)

        model_compile_args_base = get_model_compile_args(
            FLAGS, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics_to_add=[
                adv_acc_metric,
                tf.keras.metrics.AUC(),
            ]
        )

        model_base.compile(**model_compile_args_base)
        model_base.summary()

        print("[INFO] training base model")
        callbacks_base = make_callbacks(FLAGS, is_adversarial=False)

        model_base.fit(train_ds.dataset, callbacks=callbacks_base,
                       validation_data=val_ds.dataset, **train_args)

        # Evaluate the accuracy on legitimate and adversarial test examples
        base_metrics_test = model_base.evaluate(test_ds.dataset)
        for name, value in zip(model_base.metrics_names, base_metrics_test):
            data, metric_name = get_data_type_and_metric_from_name(name)
            results.add_result({"metric": metric_name,
                                "value": value,
                                "model": keys.BASE_MODEL,
                                "data": data,
                                "phase": keys.TEST})

        # Calculate training error
        base_metrics_train = model_base.evaluate(train_ds.dataset,
                                                 steps=steps_per_train_epoch)
        for name, value in zip(model_base.metrics_names, base_metrics_train):
            data, metric_name = get_data_type_and_metric_from_name(name)
            results.add_result({"metric": metric_name,
                                "value": value,
                                "model": keys.BASE_MODEL,
                                "data": data,
                                "phase": keys.TRAIN})

    # Redefine Keras model
    if FLAGS.train_adversarial:

        model_adv = get_model_from_flags(FLAGS)
        model_adv(model_adv.input)
        attack = get_attack(FLAGS.attack, model_adv, sess=sess)

        # Use a loss function based on legitimate and adversarial examples
        adv_loss_adv = get_adversarial_loss(model_adv, attack, attack_params,
                                            FLAGS.adv_multiplier)
        adv_acc_metric_adv = get_adversarial_acc_metric(model_adv, attack,
                                                        attack_params)

        model_compile_args_adv = get_model_compile_args(
            FLAGS, loss=adv_loss_adv,
            metrics_to_add=[
                adv_acc_metric_adv,
                tf.keras.metrics.AUC(),
            ]
        )

        model_adv.compile(**model_compile_args_adv)
        print("[INFO] training adversarial model")
        callbacks_adv = make_callbacks(FLAGS, is_adversarial=True)

        # Initialize the variables; this is required for the auc computation.
        # run_variable_initializers(sess)

        model_adv.fit(train_ds.dataset, callbacks=callbacks_adv,
                      validation_data=val_ds.dataset, **train_args)

        # Evaluate the accuracy on legitimate and adversarial test examples
        adv_metrics_test = model_adv.evaluate(test_ds.dataset)
        for name, value in zip(model_adv.metrics_names, adv_metrics_test):
            data, metric_name = get_data_type_and_metric_from_name(name)
            results.add_result({"metric": metric_name,
                                "value": value,
                                "model": keys.ADV_MODEL,
                                "data": data,
                                "phase": keys.TEST})

        # Calculate training error
        adv_metrics_train = model_adv.evaluate(train_ds.dataset,
                                               steps=steps_per_train_epoch)
        for name, value in zip(model_adv.metrics_names, adv_metrics_train):
            data, metric_name = get_data_type_and_metric_from_name(name)
            results.add_result({"metric": metric_name,
                                "value": value,
                                "model": keys.ADV_MODEL,
                                "data": data,
                                "phase": keys.TRAIN})
        results.to_csv(metrics_dir=FLAGS.metrics_dir)

    return


def main(argv=None):
    from cleverhans_tutorials import check_installation
    check_installation(__file__)

    mnist_tutorial()


if __name__ == '__main__':
    tf.app.run()
