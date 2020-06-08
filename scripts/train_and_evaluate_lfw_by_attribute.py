"""
A script to train and evaluate models on the LFW train/test datasets, respectively,
to compare performance of models trained on the entire dataset vs. those trained only
on minority/majority faces.

This script uses the basic keras interface for VGGFace/OpenFace, instead of cleverhans,
due to its simplicity and because cleverhans is not being used.

Usage:
export DIR="/projects/grail/jpgard/lfw/lfw-deepfunneled-traintest"
export ANNO_FP="/projects/grail/jpgard/lfw/lfw_attributes_cleaned.txt"

export GPU_ID="5"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

export CUDA_HOME=/usr/local/cuda-10.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/

for TRAIN_SUBSET in "0" "1" "0U1"
do
    for LABEL in "Male" "No Eyewear" "Smiling" "Heavy Makeup" "Shiny Skin" "Wearing Earrings"
    do
        for SLICE in "Asian" "Senior" "Male" "Black" "Youth"
        do
            echo $SLICE;
            echo $LABEL;
            echo $TRAIN_SUBSET;
            python3 scripts/train_and_evaluate_lfw_by_attribute.py \
                --train_dir $DIR/train/ \
                --test_dir $DIR/test/ \
                --model_type vggface2 \
                --label_name $LABEL \
                --slice_attribute_name $SLICE \
                --anno_fp $ANNO_FP \
                --epochs 40 \
                --train_subset $TRAIN_SUBSET \
                --experiment_uid ${TRAIN_SUBSET}_EQUAL_SUBGROUP_N \
                --equalize_subgroup_n
        done
        echo ""
    done
done


"""

from collections import OrderedDict
import os

from absl import app
from absl import flags
import pandas as pd
import tensorflow as tf

from dro import keys
from dro.utils.training_utils import make_callbacks, get_train_metrics, \
    make_model_uid_from_flags

from dro.utils.lfw import make_pos_and_neg_attr_datasets
from dro.utils.flags import define_training_flags, define_eval_flags, \
    extract_dataset_making_parameters_from_flags
from dro.utils.training_utils import get_model_compile_args, get_model_from_flags

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

define_training_flags()
define_eval_flags()

flags.DEFINE_enum("train_subset", None, ["0U1", "0", "1"], 
                  "The values of the sensitive attribute to use for training; 0U1 "
                  "indicates the union of groups 0 and 1.")


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
    train_dset = dsets_train[FLAGS.train_subset].dataset
    model.fit_generator(train_dset, callbacks=callbacks, **train_args)
    # After training completes, do the evaluation
    all_test_metrics = list()
    for test_subset_key, test_dset in dsets_test.items():
        test_metrics = model.evaluate_generator(test_dset.dataset)
        assert len(train_metrics_names) == len(test_metrics)
        test_metrics_dict = OrderedDict(zip(train_metrics_names, test_metrics))
        test_metrics_dict["test_subset"] = test_subset_key
        test_metrics_dict["train_subset"] = FLAGS.train_subset
        test_metrics_dict["sensitive_attribute"] = FLAGS.slice_attribute_name
        test_metrics_dict["label"] = FLAGS.label_name
        print("test metrics for subset {}:".format(test_subset_key))
        print(test_metrics_dict)
        all_test_metrics.append(test_metrics_dict)
    # Write the results to csv.
    uid = make_model_uid_from_flags(FLAGS, is_adversarial=False)
    uid += "train_subset{}".format(FLAGS.train_subset)
    csv_fp = os.path.join(FLAGS.metrics_dir, uid + "-lfw-{}-subsets.csv".format(
        FLAGS.slice_attribute_name))
    print("[INFO] writing results to %s" % csv_fp)
    # Coerce dtype to object; otherwise string format of "01" is changed to numeric "1"
    pd.DataFrame(all_test_metrics, dtype=object).to_csv(csv_fp, index=False)
    return

if __name__ == "__main__":
    app.run(main)
