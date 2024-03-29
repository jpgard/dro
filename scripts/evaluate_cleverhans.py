"""
A script to conduct adversarial analysis of pre-trained models.

The script applies a range of adversarial perturbations to a set of test images from
the LFW dataset, and evaluates classifier accuracy on those images. Accuracy is
reported  by image
subgroups.

usage:

export SS=0.025
export EPOCHS=40
export ATTACK="FastGradientMethod"

export DIR="/projects/grail/jpgard/lfw"

# set the gpu
export GPU_ID="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MODEL_TYPE="facenet"

for LABEL in "Mouth_Open" "Sunglasses" "Male"
do
    for SLICE_ATTR in "Asian" "Senior" "Male" "Black"
    do
        echo $LABEL;
        echo $SLICE_ATTR;
        python3 scripts/evaluate_cleverhans.py \
        --anno_fp ${DIR}/lfw_attributes_cleaned.txt \
        --test_dir ${DIR}/lfw-deepfunneled \
        --label_name $LABEL \
        --slice_attribute_name $SLICE_ATTR \
        --attack $ATTACK \
        --attack_params "{\"eps\": 0.025, \"clip_min\": null, \"clip_max\": null}" \
        --adv_multiplier 0.2 \
        --epochs $EPOCHS \
        --metrics_dir ./metrics \
        --model_type $MODEL_TYPE
    done
    echo ""
done

for LABEL in "Mouth_Open" "Sunglasses" "Male" "Eyeglasses"
do
    for SLICE_ATTR in "Asian" "Senior" "Male" "Black"
    do
        echo $LABEL;
        echo $SLICE_ATTR;
        python3 scripts/evaluate_cleverhans.py \
        --anno_fp ${DIR}/lfw_attributes_cleaned.txt \
        --test_dir ${DIR}/lfw-deepfunneled \
        --label_name $LABEL \
        --slice_attribute_name $SLICE_ATTR \
        --attack FrankWolfeDistributionallyRobustMethod \
        --attack_params "{\"eps\": 0.025, \"nb_iter\": 15, \"eps_iter\": 0.0025, \"clip_min\": null, \"clip_max\": null}" \
        --adv_multiplier $ADV_MULTIPLIER \
        --epochs $EPOCHS \
        --metrics_dir ./metrics \
        --model_type $MODEL_TYPE
    done
    echo ""
done

"""

from collections import defaultdict, OrderedDict
from statistics import mean
from absl import app

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import roc_auc_score

from cleverhans.compat import flags
from dro.training.training_utils import load_model_weights_from_flags
from dro.utils.flags import define_training_flags, define_eval_flags, \
    define_adv_training_flags, extract_dataset_making_parameters_from_flags, \
    get_model_compile_args, get_model_from_flags, get_model_img_shape_from_flags
from dro.utils.reports import Report
from dro.utils.cleverhans import get_attack, attack_params_from_flags, \
    generate_attack
from dro.utils.evaluation import ADV_STEP_SIZE_GRID
from dro.utils.lfw import make_pos_and_neg_attr_datasets
from dro.training.training_utils import make_model_uid_from_flags
from dro import keys
from dro.utils.viz import show_adversarial_resuts

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS

# the vggface2/training parameters
define_training_flags()

# the adversarial training parameters
define_adv_training_flags(cleverhans=True)

# the evaluation flags
define_eval_flags()


def evaluate_cleverhans_models_on_dataset(sess: tf.Session, eval_dset_numpy, epsilon):
    """Evaluate the pretrained models (both base and adversarial) on eval_dset.

    Both models are evaluated at the same time because the attacks are the same (
    attacks are performed wrt the base model).

    :return: a tuple of dicts (res, sample_batch) where res is a dict of accuracy
        metrics, and sample_batch is a dict of elements corresponding to a sample batch
        (by default, the first batch is used).
    """

    model_compile_args = get_model_compile_args(
        FLAGS, tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics_to_add=None)
    model_base = get_model_from_flags(FLAGS)
    model_base.compile(**model_compile_args)
    load_model_weights_from_flags(model_base, FLAGS, is_adversarial=False)
    model_adv = get_model_from_flags(FLAGS)
    model_adv.compile(**model_compile_args)
    load_model_weights_from_flags(model_adv, FLAGS, is_adversarial=True)

    # Initialize the attack. The attack is always computed relative to the base model.
    # If the attack is a randomized method, get_attack(eval=True) will use the
    # deterministic version of that attack instead.

    attack_params = attack_params_from_flags(FLAGS, override_eps_value=epsilon)
    attack = get_attack(FLAGS.attack, model_base, sess, eval=True)

    # Define the ops to run for evaluation
    imshape = get_model_img_shape_from_flags(FLAGS)
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, imshape[0], imshape[1], 3))
    x_perturbed = generate_attack(attack, x, attack_params)
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))  # [batch_size, 2]
    yhat_base_perturbed = model_base(x_perturbed)
    yhat_base_clean = model_base(x)
    yhat_adv_perturbed = model_adv(x_perturbed)
    yhat_adv_clean = model_adv(x)

    # the acc_ results tensors each have shape [batch_size,] but note that the final
    # batch of a dataset may not be full-sized (when N % batch_size != 0)

    acc_bp_op = tf.keras.metrics.categorical_accuracy(y, yhat_base_perturbed)
    acc_bc_op = tf.keras.metrics.categorical_accuracy(y, yhat_base_clean)
    acc_ap_op = tf.keras.metrics.categorical_accuracy(y, yhat_adv_perturbed)
    acc_ac_op = tf.keras.metrics.categorical_accuracy(y, yhat_adv_clean)

    # Build dicts of {op_name: op_tensor} to run in the session.

    # The ops which run on clean data (these are always run even when epsilon=0)
    clean_data_ops = OrderedDict({"acc_base_clean": acc_bc_op,
                                  "acc_adv_clean": acc_ac_op,
                                  "yhat_base_clean": yhat_base_clean,
                                  "yhat_adv_clean": yhat_adv_clean,
                                  })

    # The ops which run on perturbed data (these are only run when epsilon > 0).
    perturbed_data_ops = OrderedDict({"acc_base_perturbed": acc_bp_op,
                                      "acc_adv_perturbed": acc_ap_op,
                                      "yhat_base_perturbed": yhat_base_perturbed,
                                      "yhat_adv_perturbed": yhat_adv_perturbed,
                                      "x_perturbed": x_perturbed
                                      })

    if epsilon > 0:
        ops_to_run = perturbed_data_ops
        sample_batch_keys_to_update = ["yhat_base_perturbed", "yhat_adv_perturbed",
                                       "x_perturbed"]
    else:
        ops_to_run = clean_data_ops
        sample_batch_keys_to_update = ["yhat_base_clean", "yhat_adv_clean"]

    # The names of the ops which compute accuracy
    acc_keys_to_update = [k for k in ops_to_run.keys() if k.startswith("acc")]

    batch_index = 0
    sample_batch = defaultdict()
    accuracies = defaultdict(list)
    yhats = defaultdict(list)
    y_true = list()

    for batch_x, batch_y in eval_dset_numpy:
        batch_res = sess.run(list(ops_to_run.values()),
                             feed_dict={x: batch_x, y: batch_y})
        batch_res = dict(zip(ops_to_run.keys(), batch_res))
        if batch_index == 0:
            print("[INFO] storing sample batch.")
            # Always store the x and y
            sample_batch["x"] = batch_x
            sample_batch["y"] = batch_y
            # Update the remaining sample batch keys.
            for k in sample_batch_keys_to_update:
                sample_batch[k] = batch_res[k]

        # Update the accuracy metrics. We store the binary "correct" vector for
        # categorical accuracy as a list; this is because we need to know the exact
        # dataset size to compute overall accuracy.
        for k in acc_keys_to_update:
            accuracies[k].extend(batch_res[k].tolist())

        # Store the yhats for each model. These are used to compute the AUC, which can
        # only be computed once we have predictions for the entire dataset.
        for j in batch_res.keys():
            if j.startswith("yhat"):
                yhats[j].append(batch_res[j])
        # store the ys
        y_true.append(batch_y)

        batch_index += 1
        if FLAGS.debug:
            print("[INFO] running in debug mode")
            break

    print("[INFO] completed inference for {} batches".format(batch_index))
    # Compute the accuracies; this is the mean of a binary "correct/incorrect" vector
    res = {k: mean(accuracies[k]) for k in acc_keys_to_update}
    # Compute the AUCs
    y_true = np.concatenate(y_true)
    for k in yhats.keys():
        yhat = np.concatenate(yhats[k])
        try:
            auc = roc_auc_score(y_true, yhat)
        except ValueError as ve:
            print(ve)
            print("entering null value for auc")
            auc = np.nan
        name = k.replace("yhat", "auc")
        res[name] = auc
    return res, sample_batch


def main(argv):
    # Object used to keep track of (and return) key accuracies
    results = Report(FLAGS)

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    def _clear_and_start_session():
        """Clear the current session and start a new one."""
        K.get_session().close()
        # Create TF session and set as Keras backend session
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
        # Set the learning phase to False, following the issue here:
        # https://github.com/tensorflow/cleverhans/issues/1052
        K.set_learning_phase(False)
        return sess

    # A dict of parameters for passing to make_pos_and_neg_attr_datasets
    make_datasets_parameters = extract_dataset_making_parameters_from_flags(FLAGS,
                                                                            write_samples=False)

    for attr_val in ("0", "1"):

        # Do the evalaution with no epsilon; this only evaluates on the clean data.
        sess = _clear_and_start_session()

        # Create an iterator which generates (batch_of_x, batch_of_y) tuples of numpy
        # arrays.
        eval_dsets = make_pos_and_neg_attr_datasets(**make_datasets_parameters)
        eval_dset_numpy = tfds.as_numpy(eval_dsets[attr_val].dataset)

        res, _ = evaluate_cleverhans_models_on_dataset(sess, eval_dset_numpy,
                                                       epsilon=0.)
        for k, v in res.items():
            metric, model, data = k.split("_")
            # Sanity check to ensure only metrics on clean data are returned at this step
            assert data == keys.CLEAN_DATA
            results.add_result({"metric": metric,
                                "value": v,
                                "model": model,
                                "data": data,
                                "phase": keys.TEST,
                                "attr": FLAGS.slice_attribute_name,
                                "attr_val": attr_val,
                                "epsilon": None
                                })

        for adv_step_size_to_eval in ADV_STEP_SIZE_GRID:
            print("adv_step_size_to_eval %f" % adv_step_size_to_eval)
            sess = _clear_and_start_session()

            # Create an iterator which generates (batch_of_x, batch_of_y) tuples of numpy
            # arrays.
            eval_dsets = make_pos_and_neg_attr_datasets(**make_datasets_parameters)
            eval_dset_numpy = tfds.as_numpy(eval_dsets[attr_val].dataset)

            res, sample_batch = evaluate_cleverhans_models_on_dataset(
                sess, eval_dset_numpy, epsilon=adv_step_size_to_eval)
            for k, v in res.items():
                metric, model, data = k.split("_")
                # Sanity check to ensure only metrics on perturbed data are returned at
                # this step
                assert data == keys.ADV_DATA
                results.add_result({"metric": metric,
                                    "value": v,
                                    "model": model,
                                    "data": data,
                                    "phase": keys.TEST,
                                    "attr": FLAGS.slice_attribute_name,
                                    "attr_val": attr_val,
                                    "epsilon": adv_step_size_to_eval
                                    })
            adv_image_basename = \
                "./debug/adv-examples-{uid}-{attr}-{val}-{attack}step{ss}".format(
                    uid=make_model_uid_from_flags(FLAGS, is_adversarial=True),
                    attr=FLAGS.slice_attribute_name,
                    val=attr_val,
                    attack=FLAGS.attack,
                    ss=adv_step_size_to_eval
                )
            # check the shapes, etc of the arguments to show_adversarial_results
            show_adversarial_resuts(
                n_batches=1,
                # Add a batch_dimension to the results.
                perturbed_images=[sample_batch["x_perturbed"], ],
                labels=[np.argmax(sample_batch["y"], axis=1), ],
                # Convert the predictions to a binary label
                predictions=[
                    {keys.BASE_MODEL: np.argmax(sample_batch["yhat_base_perturbed"],
                                                axis=1),
                     keys.ADV_MODEL: np.argmax(sample_batch["yhat_adv_perturbed"],
                                               axis=1)}, ],
                fp_basename=adv_image_basename,
                batch_size=FLAGS.batch_size
            )

    results.to_csv(metrics_dir=FLAGS.metrics_dir, attr_name=FLAGS.slice_attribute_name)


if __name__ == "__main__":
    print("running")
    app.run(main)
