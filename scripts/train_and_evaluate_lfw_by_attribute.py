"""
A script to train and evaluate models on the LFW train/test datasets, respectively,
to compare performance of models trained on the entire dataset vs. those trained only
on minority/majority faces.

This script uses the basic keras interface for VGGFace/OpenFace, instead of cleverhans,
due to its simplicity and because cleverhans is not being used.

Usage:
python3 scripts/train_and_evaluate_lfw_by_attribute.py \
    --train_dir $DIR/train/
    --test_dir $DIR/test/
"""

from collections import OrderedDict
import os
import numpy as np

from absl import app
from absl import flags
import neural_structured_learning as nsl
import pandas as pd
import tensorflow as tf

from dro.keys import LABEL_INPUT_NAME
from dro.training.models import vggface2_model
from dro.utils.training_utils import make_callbacks, \
    write_test_metrics_to_csv, get_train_metrics, make_model_uid_from_flags, \
    add_adversarial_metric_names_to_list, get_n_from_file_pattern, \
    compute_n_train_n_val, \
    steps_per_epoch, load_model_weights_from_flags
from dro.datasets import ImageDataset
from dro.utils.lfw import make_lfw_file_pattern
from dro.utils.testing import assert_shape_equal, assert_file_exists
from dro.datasets.dbs import LabeledBatchGenerator
from dro.utils.flags import define_training_flags, define_adv_training_flags, \
    get_attack_params

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

# the vggface2/training parameters
define_training_flags()

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main(argv):
    train_file_pattern = make_lfw_file_pattern(FLAGS.train_dir)
    test_file_pattern = make_lfw_file_pattern(FLAGS.test_dir)
    n_train_val = get_n_from_file_pattern(train_file_pattern)
    n_train, n_val = compute_n_train_n_val(n_train_val, FLAGS.val_frac)

    train_ds = ImageDataset()
    test_ds = ImageDataset()
    return

if __name__ == "__main__":
    app.run(main)