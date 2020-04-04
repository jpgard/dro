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

from absl import app, flags

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

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

tf.compat.v1.enable_eager_execution()

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

        # Build the models for evalaution on clean data
        attack_params = get_attack_params(FLAGS.adv_step_size)
        # load the models
        # vgg_model_base = make_compiled_model(sess, attack_params, is_adversarial=False)
        print("[INFO] reached dev block.")
        #### Try to load the model without using the adversarial loss metric.
        eval_dset = make_pos_and_neg_attr_datasets(FLAGS)[attr_val].dataset
        eval_dset_x = eval_dset.map(lambda x, y: x)
        eval_dset_y = eval_dset.map(lambda x, y: y)
        vgg_model_base = vggface2_model(dropout_rate=FLAGS.dropout_rate,
                                        activation="softmax")
        model_compile_args = {
            "optimizer": tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate),
            "loss": None,  # No loss needed for evaluation.
            "metrics": ['accuracy', ]}
        vgg_model_base.compile(**model_compile_args)
        attack = get_attack(FLAGS, vgg_model_base, sess)
        x_adv = attack.generate(eval_dset_x, **attack_params)
        preds_adv = vgg_model_base(x_adv)
        acc = keras.metrics.categorical_accuracy(eval_dset_y, preds_adv)
        print(acc)
        import ipdb;
        ipdb.set_trace()
        #####################################################################

        # Adversarial model
        vgg_model_adv = make_compiled_model(sess, attack_params, is_adversarial=True)

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
