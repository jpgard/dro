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

python3 scripts/evaluate.py \
    --anno_fp ${DIR}/lfw_attributes_cleaned.txt \
    --test_dir ${DIR}/lfw-deepfunneled \
    --label_name $LABEL \
    --slice_attribute_name $SLICE_ATTR \
    --adv_step_size $SS \
    --epochs $EPOCHS

for SLICE_ATTR in "Asian" "Senior" "Male" "Black"
do
    python3 scripts/evaluate.py \
    --anno_fp ${DIR}/lfw_attributes_cleaned.txt \
    --test_dir ${DIR}/lfw-deepfunneled \
    --label_name $LABEL \
    --slice_attribute_name $SLICE_ATTR \
    --adv_step_size $SS \
    --epochs $EPOCHS --metrics_dir ./tmp
done
"""

from absl import app, flags
from collections import OrderedDict
import os

import tensorflow as tf
import pandas as pd

from dro.utils.evaluation import make_pos_and_neg_attr_datasets, ADV_STEP_SIZE_GRID
from dro.utils.training_utils import add_keys_to_dict
from dro.training.models import vggface2_model
import neural_structured_learning as nsl
from dro.keys import LABEL_INPUT_NAME, ADV_MODEL, BASE_MODEL, \
    ADV_DATA, CLEAN_DATA
from dro.utils.training_utils import get_train_metrics, \
    add_adversarial_metric_names_to_list
from dro.utils.training_utils import make_ckpt_filepath
from dro.utils.training_utils import perturb_and_evaluate, \
    make_compiled_reference_model, load_model_weights_from_flags
from dro.utils.training_utils import make_model_uid
from dro.utils.viz import show_adversarial_resuts
from dro.utils.flags import define_training_flags, define_eval_flags, \
    define_adv_training_flags

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


def main(argv):
    # load the models
    train_metrics_dict = get_train_metrics()
    train_metrics_names = ["categorical_crossentropy", ] + list(train_metrics_dict.keys())
    train_metrics = list(train_metrics_dict.values())
    model_compile_args = {
        "optimizer": tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate),
        "loss": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        "metrics": train_metrics
    }
    vgg_model_base = vggface2_model(dropout_rate=FLAGS.dropout_rate)
    vgg_model_base.compile(**model_compile_args)
    load_model_weights_from_flags(vgg_model_base, FLAGS, is_adversarial=False)

    # Adversarial model
    adv_config = nsl.configs.make_adv_reg_config(
        multiplier=FLAGS.adv_multiplier,
        adv_step_size=FLAGS.adv_step_size,
        adv_grad_norm=FLAGS.adv_grad_norm
    )
    base_adv_model = vggface2_model(dropout_rate=FLAGS.dropout_rate)
    adv_model = nsl.keras.AdversarialRegularization(
        base_adv_model,
        label_keys=[LABEL_INPUT_NAME],
        adv_config=adv_config
    )
    adv_model.compile(**model_compile_args)
    load_model_weights_from_flags(adv_model, FLAGS, is_adversarial=True)

    # List to store the results of the experiment
    metrics_list = list()

    for attr_val in ("0", "1"):

        # Get the evaluation metrics for clean inputs.
        def get_model_metrics(model, dataset: tf.data.Dataset, metric_names: list,
                              is_adversarial: bool, data_type: str):
            metrics = model.evaluate(dataset)
            if is_adversarial:
                metric_names = add_adversarial_metric_names_to_list(metric_names)
            assert len(metric_names) == len(metrics)
            metrics_dict = OrderedDict(zip(metric_names, metrics))
            metrics_dict = add_keys_to_dict(
                metrics_dict, attr_val=attr_val,
                attr_name=FLAGS.slice_attribute_name,
                uid=make_model_uid(FLAGS, is_adversarial),
                data=data_type)
            return metrics_dict

        print("[INFO] evaluating base model on clean data")
        clean_input_metrics_base = get_model_metrics(
            vgg_model_base,
            make_pos_and_neg_attr_datasets(FLAGS)[attr_val].dataset,
            train_metrics_names,
            is_adversarial=False,
            data_type=CLEAN_DATA)
        print("[INFO] evaluating adversarial model on clean data")
        # Convert the dataset to dictionary for input to the adversarial model.
        dset = make_pos_and_neg_attr_datasets(FLAGS)[attr_val]
        dset.convert_to_dictionaries()
        clean_input_metrics_adv = get_model_metrics(
            adv_model,
            dset.dataset,
            train_metrics_names,
            is_adversarial=True,
            data_type=CLEAN_DATA)

        metrics_list.extend([clean_input_metrics_adv, clean_input_metrics_base])

        for adv_step_size_to_eval in ADV_STEP_SIZE_GRID:
            print("adv_step_size_to_eval %f" % adv_step_size_to_eval)
            reference_model = make_compiled_reference_model(
                model_base=vgg_model_base,
                adv_config=nsl.configs.make_adv_reg_config(
                    multiplier=FLAGS.adv_multiplier,
                    adv_step_size=adv_step_size_to_eval,
                    adv_grad_norm=FLAGS.adv_grad_norm
                ),
                model_compile_args=model_compile_args)
            models_to_eval = {
                BASE_MODEL: vgg_model_base,
                ADV_MODEL: adv_model.base_model
            }

            # Perturb the images and get the metrics for adversarial inputs
            dset = make_pos_and_neg_attr_datasets(FLAGS)[attr_val]
            dset.convert_to_dictionaries()
            perturbed_images, labels, predictions, adv_input_metrics_dict = \
                perturb_and_evaluate(
                    dset.dataset,
                    models_to_eval,
                    reference_model)

            # Add other identifiers to the metrics dict and save to metrics_list

            adv_input_metrics_adv = add_keys_to_dict(
                adv_input_metrics_dict[ADV_MODEL], attr_val=attr_val,
                attr_name=FLAGS.slice_attribute_name,
                uid=make_model_uid(FLAGS, is_adversarial=True),
                adv_step_size=adv_step_size_to_eval,
                data=ADV_DATA)

            adv_input_metrics_base = add_keys_to_dict(
                adv_input_metrics_dict[BASE_MODEL], attr_val=attr_val,
                attr_name=FLAGS.slice_attribute_name,
                uid=make_model_uid(FLAGS, is_adversarial=False),
                adv_step_size=adv_step_size_to_eval,
                data=ADV_DATA)

            metrics_list.extend([adv_input_metrics_adv, adv_input_metrics_base])
            # Write the results for 3 batches to a file for inspection.
            adv_image_basename = \
                "./debug/adv-examples-{uid}-{attr}-{val}-step{ss}".format(
                    uid=make_model_uid(FLAGS, is_adversarial=True),
                    attr=FLAGS.slice_attribute_name,
                    val=attr_val,
                    ss=adv_step_size_to_eval
                )

            show_adversarial_resuts(n_batches=3,
                                    perturbed_images=perturbed_images,
                                    labels=labels,
                                    predictions=predictions,
                                    fp_basename=adv_image_basename,
                                    batch_size=FLAGS.batch_size)

    metrics_csv = "{}-{}-adversarial-analysis.csv".format(
        make_model_uid(FLAGS, is_adversarial=True), FLAGS.slice_attribute_name)
    metrics_fp = os.path.join(FLAGS.metrics_dir, metrics_csv)
    print("[INFO] writing results to {}".format(metrics_fp))
    pd.DataFrame(metrics_list).to_csv(metrics_fp)


if __name__ == "__main__":
    print("running")
    app.run(main)
