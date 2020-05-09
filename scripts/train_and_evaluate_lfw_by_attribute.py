"""
A script to train and evaluate models on the LFW train/test datasets, respectively,
to compare performance of models trained on the entire dataset vs. those trained only
on minority/majority faces.

This script uses the basic keras interface for VGGFace/OpenFace, instead of cleverhans,
due to its simplicity and because cleverhans is not being used.

Usage:
python3 scripts/train_and_evaluate_lfw_by_attribute.py \
    --train_dir $DIR/train/ \
    --test_dir $DIR/test/ \
    --model_type vggface2 \
    --label_name Male \
    --slice_attribute_name Black \
    --anno_fp /Users/jpgard/Documents/research/lfw/lfw_attributes_cleaned.txt \
    --epochs 40 \
    --train_base \
    --experiment_uid dset_union

python3 scripts/train_and_evaluate_lfw_by_attribute.py \
    --train_dir $DIR/train/ \
    --test_dir $DIR/test/ \
    --model_type vggface2 \
    --label_name Male \
    --slice_attribute_name Black \
    --anno_fp /Users/jpgard/Documents/research/lfw/lfw_attributes_cleaned.txt \
    --epochs 40 \
    --train_minority \
    --experiment_uid dset_minority

python3 scripts/train_and_evaluate_lfw_by_attribute.py \
    --train_dir $DIR/train/ \
    --test_dir $DIR/test/ \
    --model_type vggface2 \
    --label_name Male \
    --slice_attribute_name Black \
    --anno_fp /Users/jpgard/Documents/research/lfw/lfw_attributes_cleaned.txt \
    --epochs 40 \
    --train_majority \
    --experiment_uid dset_majority
"""

from collections import OrderedDict
import os
import numpy as np

from absl import app
from absl import flags
import pandas as pd
import tensorflow as tf

from dro import keys
from dro.utils.training_utils import make_callbacks, get_train_metrics, \
    make_model_uid_from_flags, make_csv_name

from dro.utils.lfw import make_pos_and_neg_attr_datasets
from dro.utils.flags import define_training_flags, define_eval_flags, \
    extract_dataset_making_parameters_from_flags
from dro.utils.training_utils import get_model_compile_args, get_model_from_flags

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

define_training_flags()
define_eval_flags()

flags.DEFINE_bool("train_majority", False, "whether to train the model on the majority "
                                           "dataset only")
flags.DEFINE_bool("train_minority", False, "whether to train the model on the minority "
                                           "dataset only")

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(argv):
    make_datasets_parameters_train = extract_dataset_making_parameters_from_flags(
        FLAGS, write_samples=False, test=False)
    make_datasets_parameters_test = extract_dataset_making_parameters_from_flags(
        FLAGS, write_samples=False, test=True
    )
    dsets_train = make_pos_and_neg_attr_datasets(
        **make_datasets_parameters_train, include_union=True,
        preprocessing_kwargs={"shuffle": True, "repeat_forever": False,
                              "batch_size": FLAGS.batch_size}
    )
    dsets_test = make_pos_and_neg_attr_datasets(
        **make_datasets_parameters_test, include_union=True,
        preprocessing_kwargs={"shuffle": False, "repeat_forever": False,
                              "batch_size": FLAGS.batch_size}
    )

    # The metrics to optimize during training
    train_metrics_dict = get_train_metrics()
    # .evaluate() automatically prepends the loss(es), so it will always include at
    # least categorical_crossentropy (adversarial also adds the AT loss terms)
    train_metrics_names = ["categorical_crossentropy", ] + list(train_metrics_dict.keys())
    train_metrics = list(train_metrics_dict.values())

    model_compile_args = get_model_compile_args(
        FLAGS, tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics_to_add=train_metrics_dict.values())

    # Shared training arguments for the model fitting.
    train_args = {
        # "steps_per_epoch": steps_per_train_epoch,
        "epochs": FLAGS.epochs,
        # "validation_steps": steps_per_val_epoch
    }

    callbacks = make_callbacks(FLAGS, is_adversarial=False)

    # The base model, trained on both minority and majority
    model = get_model_from_flags(FLAGS)
    model.compile(**model_compile_args)
    model.summary()

    if FLAGS.train_base:
        train_subset_key = "0U1"
    elif FLAGS.train_minority:
        train_subset_key = "1"
    elif FLAGS.train_majority:
        train_subset_key = "0"
    else:
        raise ValueError("Must specify either train_base, train_minority, "
                         "or train_majority flags.")
    train_dset = dsets_train[train_subset_key].dataset
    model.fit_generator(train_dset, callbacks=callbacks, **train_args)
    # After training completes, do the evaluation
    all_test_metrics = list()
    for test_subset_key, test_dset in dsets_test.items():
        test_metrics = model.evaluate_generator(test_dset.dataset)
        assert len(train_metrics_names) == len(test_metrics)
        test_metrics_dict = OrderedDict(zip(train_metrics_names, test_metrics))
        test_metrics_dict["test_subset"] = test_subset_key
        test_metrics_dict["train_subset"] = train_subset_key
        test_metrics_dict["sensitive_attribute"] = FLAGS.slice_attribute_name
        test_metrics_dict["label"] = FLAGS.label_name
        print("test metrics for subset {}:".format(test_subset_key))
        print(test_metrics_dict)
        all_test_metrics.append(test_metrics_dict)
    # Write the results to csv.
    uid = make_model_uid_from_flags(FLAGS, is_adversarial=False)
    uid += "train_subset{}".format(train_subset_key)
    csv_fp = os.path.join(FLAGS.metrics_dir, uid + "-lfw-subsets.csv")
    print("[INFO] writing results to %s" % csv_fp)
    # Coerce dtype to object; otherwise string format of "01" is changed to numeric "1"
    pd.DataFrame(all_test_metrics, dtype=object).to_csv(csv_fp, index=False)
    return

if __name__ == "__main__":
    app.run(main)
