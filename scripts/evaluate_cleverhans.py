"""
A script to conduct adversarial analysis of pre-trained models.

The script applies a range of adversarial perturbations to a set of test images from
the LFW dataset, and evaluates classifier accuracy on those images. Accuracy is
reported  by image
subgroups.

usage:
export LABEL="Mouth_Open"
export LABEL="Sunglasses"
export LABEL="Male"
export SS=0.025
export EPOCHS=40

export DIR="/projects/grail/jpgard/lfw"

python3 scripts/evaluate_cleverhans.py \
    --anno_fp ${DIR}/lfw_attributes_cleaned.txt \
    --test_dir ${DIR}/lfw-deepfunneled \
    --label_name $LABEL \
    --slice_attribute_name $SLICE_ATTR \
    --adv_step_size $SS \
    --epochs $EPOCHS

for SLICE_ATTR in "Asian" "Senior" "Male" "Black"
do
    python3 scripts/evaluate_cleverhans.py \
    --anno_fp ${DIR}/lfw_attributes_cleaned.txt \
    --test_dir ${DIR}/lfw-deepfunneled \
    --label_name $LABEL \
    --slice_attribute_name $SLICE_ATTR \
    --adv_step_size $SS \
    --epochs $EPOCHS --metrics_dir ./tmp
done
"""

from collections import defaultdict
import pprint
from statistics import mean
from absl import app, flags

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds
import numpy as np

from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent, Noise
from cleverhans.compat import flags
from dro.training.models import vggface2_model
from dro.utils.training_utils import load_model_weights_from_flags
from dro.utils.flags import define_training_flags, define_eval_flags, \
    define_adv_training_flags
from dro.utils.reports import Report
from dro.utils.cleverhans import get_attack, get_attack_params, get_adversarial_loss, \
    get_adversarial_acc_metric, get_model_compile_args
from dro.utils.evaluation import make_pos_and_neg_attr_datasets, ADV_STEP_SIZE_GRID
from dro import keys

# tf.compat.v1.enable_eager_execution()

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS

# the vggface2/training parameters
define_training_flags()

# the adversarial training parameters
define_adv_training_flags()

# the evaluation flags
define_eval_flags()


def make_compiled_model(sess, attack_params, is_adversarial: bool, activation="softmax"):
    model = vggface2_model(dropout_rate=FLAGS.dropout_rate, activation=activation)
    model(model.input)
    # Initialize the attack object
    attack = get_attack(FLAGS, model, sess)
    print("[INFO] using attack {} with params {}".format(FLAGS.attack, attack_params))

    adv_acc_metric = get_adversarial_acc_metric(model, attack, attack_params)
    if is_adversarial:
        loss = get_adversarial_loss(model, attack, attack_params, FLAGS.adv_multiplier)
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model_compile_args_base = get_model_compile_args(FLAGS, loss=loss,
                                                     adv_acc_metric=adv_acc_metric)
    model.compile(**model_compile_args_base)
    load_model_weights_from_flags(model, FLAGS, is_adversarial=is_adversarial)
    return model


def evaluate_cleverhans_models_on_dataset(sess: tf.Session, eval_dset):
    # Create an iterator which generates (batch_of_x, batch_of_y) tuples of numpy
    # arrays.
    eval_dset_numpy = tfds.as_numpy(eval_dset)

    attack_params = get_attack_params(FLAGS.adv_step_size)

    vgg_model_base = vggface2_model(dropout_rate=FLAGS.dropout_rate,
                                    activation="softmax")

    # Use the same compile args for both models. Since we are not training,
    # the optimizer and loss will not be used to adjust any parameters.

    model_compile_args = {
        "optimizer": tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate),
        "loss": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        "metrics": ['accuracy', ]}
    vgg_model_base.compile(**model_compile_args)
    load_model_weights_from_flags(vgg_model_base, FLAGS, is_adversarial=False)

    vgg_model_adv = vggface2_model(dropout_rate=FLAGS.dropout_rate,
                                   activation="softmax")
    vgg_model_adv.compile(**model_compile_args)
    load_model_weights_from_flags(vgg_model_adv, FLAGS, is_adversarial=True)

    # The attack is always computed relative to the base model
    attack = get_attack(FLAGS, vgg_model_base, sess)

    accuracies = defaultdict(list)

    # Define the ops to run for evaluation

    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
    x_perturbed = attack.generate(x, **attack_params)
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))  # [batch_size, 2]
    yhat_base_perturbed = vgg_model_base(x_perturbed)
    yhat_base_clean = vgg_model_base(x)
    yhat_adv_perturbed = vgg_model_adv(x_perturbed)
    yhat_adv_clean = vgg_model_adv(x)
    # the acc_ results have shape [batch_size,] but note that the final batch of a
    # dataset may not be full-sized (when N % batch_size != 0)

    acc_bp_op = tf.keras.metrics.categorical_accuracy(y, yhat_base_perturbed)
    acc_bc_op = tf.keras.metrics.categorical_accuracy(y, yhat_base_clean)
    acc_ap_op = tf.keras.metrics.categorical_accuracy(y, yhat_adv_perturbed)
    acc_ac_op = tf.keras.metrics.categorical_accuracy(y, yhat_adv_clean)
    ops_to_run = [acc_bp_op, acc_bc_op, acc_ap_op, acc_ac_op,
                  yhat_base_perturbed, yhat_base_clean,
                  yhat_adv_perturbed, yhat_adv_clean,
                  x_perturbed]

    # TODO(jpgard): can we read these directly from the op names (probably need to
    #  assign them names above)?

    op_names = ["acc_base_perturbed", "acc_base_clean", "acc_adv_perturbed",
                "acc_adv_clean",
                "yhat_base_perturbed", "yhat_base_clean",
                "yhat_adv_perturbed", "yhat_adv_clean"
                "x_perturbed"]

    acc_keys_to_update = ["acc_base_perturbed", "acc_base_clean", "acc_adv_perturbed",
                          "acc_adv_clean"]
    batch_index = 0
    sample_batch = defaultdict()
    for batch_x, batch_y in eval_dset_numpy:
        res = sess.run(ops_to_run, feed_dict={x: batch_x, y: batch_y})
        res_dict = dict(zip(op_names, res))
        if batch_index == 0:
            print("[INFO] storing sample batch.")
            print(res_dict.keys())
            sample_batch["x_clean"] = batch_x
            sample_batch["x_perturbed"] = res_dict["x_perturbed"]
            sample_batch["y"] = batch_y
            sample_batch["yhat_base_perturbed"] = res_dict["yhat_base_perturbed"]
            sample_batch["yhat_base_clean"] = res_dict["yhat_base_clean"]
            sample_batch["yhat_adv_perturbed"] = res_dict["yhat_adv_perturbed"]
            sample_batch["yhat_adv_clean"] = res_dict["yhat_adv_clean"]

        # We store the binary "correct" vector for categorical accuracy as a list;
        # this is because we need to know the exact dataset size to compute overall
        # accuracy.
        for k in acc_keys_to_update:
            accuracies[k].extend(res_dict[k].tolist())
        # Print stats for debugging
        for k in acc_keys_to_update:
            print("batch {} {}: {}".format(batch_index, k, mean(res_dict[k])))
        batch_index += 1

    res = zip([(k, mean(accuracies[k])) for k in acc_keys_to_update])
    import ipdb;ipdb.set_trace()
    return res, sample_batch

def main(argv):
    # Object used to keep track of (and return) key accuracies
    results = Report(FLAGS)

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    # Force TensorFlow to use single thread to improve reproducibility
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)

    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Set the learning phase to False, following the issue here:
    # https://github.com/tensorflow/cleverhans/issues/1052

    K.set_learning_phase(False)

    for attr_val in ("0", "1"):
        # TODO(jpgard): don't load the model weights and call the model inside the
        #  adversarial loss function. Instead, just generate pperturbed inputs for each
        #  batch, and compute the accuracy on those batches as you iterate over them.
        #  In other words, you never call model(x_adv) inside any function; you only
        #  call attack.generate() and then use batches of those generated examples to
        #  compute the accuracy.

        #####################################################################

        print("[INFO] reached dev block.")
        #### Try to load the model without using the adversarial loss metric.
        eval_dset = make_pos_and_neg_attr_datasets(FLAGS, write_samples=False)[
            attr_val].dataset

        res = evaluate_cleverhans_models_on_dataset(sess, eval_dset)
        import ipdb;
        ipdb.set_trace()

        # print(acc)
        #####################################################################
        # load the models
        vgg_model_base = make_compiled_model(sess, attack_params, is_adversarial=False)

        print("[INFO] evaluating base model on clean data")
        _, acc, _ = vgg_model_base.evaluate(make_pos_and_neg_attr_datasets(FLAGS)[
                                                attr_val].dataset)
        results.add_result({"metric": keys.ACC,
                            "value": acc,
                            "model": keys.BASE_MODEL,
                            "data": keys.CLEAN_DATA,
                            "phase": keys.TEST,
                            "attr": FLAGS.slice_attribute_name,
                            "attr_val": attr_val,
                            "epsilon": None
                            })

        # Adversarial model
        vgg_model_adv = make_compiled_model(sess, attack_params, is_adversarial=True)

        print("[INFO] evaluating adversarial model on clean data")
        _, acc, _ = vgg_model_adv.evaluate(
            make_pos_and_neg_attr_datasets(FLAGS)[attr_val].dataset)
        results.add_result({"metric": keys.ACC,
                            "value": acc,
                            "model": keys.ADV_MODEL,
                            "data": keys.CLEAN_DATA,
                            "phase": keys.TEST,
                            "attr": FLAGS.slice_attribute_name,
                            "attr_val": attr_val,
                            "epsilon": None
                            })

        # Remove the models to eliminate their memory footprint
        del vgg_model_base
        del vgg_model_adv

        for adv_step_size_to_eval in ADV_STEP_SIZE_GRID:
            print("adv_step_size_to_eval %f" % adv_step_size_to_eval)
            # Build the models for evaluation; we need to build a new model for each
            # epsilon due to the way the cleverhans objects are constructed.
            attack_params = get_attack_params(adv_step_size_to_eval)
            # load the models
            vgg_model_base = make_compiled_model(sess, attack_params,
                                                 is_adversarial=False)
            # Adversarial model
            vgg_model_adv = make_compiled_model(sess, attack_params, is_adversarial=True)

            _, _, adv_acc = vgg_model_base.evaluate(
                make_pos_and_neg_attr_datasets(FLAGS)[attr_val].dataset)

            results.add_result({"metric": keys.ACC,
                                "value": adv_acc,
                                "model": keys.BASE_MODEL,
                                "data": keys.ADV_DATA,
                                "phase": keys.TEST,
                                "attr": FLAGS.slice_attribute_name,
                                "attr_val": attr_val,
                                "epsilon": adv_step_size_to_eval
                                })

            _, _, adv_acc = vgg_model_adv.evaluate(
                make_pos_and_neg_attr_datasets(FLAGS)[attr_val].dataset)

            results.add_result({"metric": keys.ACC,
                                "value": adv_acc,
                                "model": keys.ADV_DATA,
                                "data": keys.ADV_DATA,
                                "phase": keys.TEST,
                                "attr": FLAGS.slice_attribute_name,
                                "attr_val": attr_val,
                                "epsilon": adv_step_size_to_eval
                                })
    results.to_csv()


if __name__ == "__main__":
    print("running")
    app.run(main)
