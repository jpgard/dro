"""
Script to fine-tune pretrained VGGFace2 model.

usage:

# set the gpu
export GPU_ID="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# run the script
export LABEL="Mouth_Open"
export DIR="/projects/grail/jpgard/vggface2/annotated_partitioned_by_label/"
export SS=0.025
export EPOCHS=40
python3 scripts/train_vggface2.py \
    --label_name $LABEL \
    --test_dir ${DIR}/test/${LABEL} \
    --train_dir ${DIR}/train/${LABEL} \
    --adv_step_size $SS --epochs $EPOCHS --notrain_base



"""

from collections import OrderedDict
import glob
import os
import time
import numpy as np

from absl import app
from absl import flags
import neural_structured_learning as nsl
import pandas as pd
import tensorflow as tf

from dro.keys import LABEL_INPUT_NAME
from dro.training.models import vggface2_model
from dro.utils.training_utils import make_callbacks, \
    write_test_metrics_to_csv, get_train_metrics
from dro.datasets import ImageDataset
from dro.utils.vggface import get_key_from_fp, make_annotations_df, image_uid_from_fp
from dro.utils.testing import assert_shape_equal, assert_file_exists, assert_ndims
from dro.datasets.dbs import LabeledBatchGenerator

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

# the vggface2/training parameters
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("epochs", 250, "the number of training epochs")
flags.DEFINE_string("train_dir", None, "directory containing the training images")
flags.DEFINE_string("test_dir", None, "directory containing the test images")
flags.DEFINE_string("ckpt_dir", "./training-logs", "directory to save/load checkpoints "
                                                   "from")
flags.DEFINE_float("learning_rate", 0.01, "learning rate to use")
flags.DEFINE_float("dropout_rate", 0.8, "dropout rate to use in fully-connected layers")
flags.DEFINE_bool("train_adversarial", True, "whether to train an adversarial model.")
flags.DEFINE_bool("train_base", True, "whether to train the base (non-adversarial) "
                                      "model. Otherwise it will be loaded from the "
                                      "default or provided checkpoint.")
flags.DEFINE_string("base_model_ckpt", None,
                    "optional manually-specified checkpoint to use to load the base "
                    "model.")
flags.DEFINE_string("adv_model_ckpt", None,
                    "optional manually-specified checkpoint to use to load the "
                    "adversarial model.")
flags.DEFINE_bool("perturbation_analysis", True, "whether to conduct a perturbation "
                                                 "analysis after completing training.")
flags.DEFINE_float("val_frac", 0.1, "proportion of data to use for validation")
flags.DEFINE_string("label_name", None,
                    "name of the prediction label (e.g. sunglasses, mouth_open)",
                    )
flags.DEFINE_string("experiment_uid", None, "Optional string identifier to be used to "
                                            "uniquely identify this experiment.")
flags.DEFINE_string("precomputed_batches_fp", None,
                    "Optional filepath to a set of precomputed batches; if provided, "
                    "these will be used for training instead of randomly-shuffled "
                    "batches of training data.")
flags.DEFINE_string("anno_dir", None,
                    "path to the directory containing the vggface annotation files.")
flags.mark_flag_as_required("label_name")
flags.mark_flag_as_required("train_dir")
flags.DEFINE_bool("debug", False,
                  "whether to run in debug mode (super short iterations to check for "
                  "bugs)")
flags.DEFINE_bool("use_dbs", False, "whether diverse batch sampling was used; if this is "
                                "set to True, batches will be read from the "
                                "precomputed_batches_fp.")

# the adversarial training parameters
flags.DEFINE_float('adv_multiplier', 0.2,
                   " The weight of adversarial loss in the training objective, relative "
                   "to the labeled loss. e.g. if this is 0.2, The model minimizes "
                   "(mean_crossentropy_loss + 0.2 * adversarial_regularization) ")
flags.DEFINE_float('adv_step_size', 0.2, "The magnitude of adversarial perturbation.")
flags.DEFINE_string('adv_grad_norm', 'infinity',
                    "The norm to measure the magnitude of adversarial perturbation.")

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_n_from_file_pattern(file_pattern):
    return len(glob.glob(file_pattern))


def steps_per_epoch(n):
    return n // FLAGS.batch_size


def compute_n_train_n_val(n_train_val):
    n_val = int(n_train_val * FLAGS.val_frac)
    n_train = n_train_val - n_val
    return n_train, n_val

def replace_parent_img_dirs(fp, label, target_parent_dirs):
    """Replace all directories above the  /person_id/image_id level of fp with
    target_parent_dirs. This function modifies filenames inplace."""
    key = get_key_from_fp(fp)
    return os.path.join(target_parent_dirs, label, key)


def main(argv):
    train_file_pattern = str(FLAGS.train_dir + '/*/*/*.jpg')
    test_file_pattern = str(FLAGS.test_dir + '/*/*/*.jpg')
    n_test = get_n_from_file_pattern(test_file_pattern)

    train_ds = ImageDataset()
    test_ds = ImageDataset()
    if FLAGS.use_dbs:

        attributes_df = make_annotations_df(FLAGS.anno_dir)
        attributes_df = attributes_df[FLAGS.label_name]
        filename_labels_dict = dict(attributes_df)

        # Modify the file paths of the elements of batches to point to the
        # uncropped images in FLAGS.train_dir, not to the original (cropped) images in
        # train_filenames which were used for the embeddings. This assumes that
        # train_dir has the same subdirectory structure as the folder of cropped images
        # used for the embeddings.

        batches = np.load(FLAGS.precomputed_batches_fp)['arr_0']
        print("[INFO] raw input batch files:")
        print(batches[0, :])
        # Extract just the base filepath, discarding the original parent directories.
        batches = pd.DataFrame(batches).applymap(lambda x: image_uid_from_fp(x)[1:])

        # Using the image uids in the filepaths, generate a dataframe with identical
        # structure where the [i,j] element in batch_labels contains the label for the
        # [i,j] element in batches.

        batch_labels = batches.replace(filename_labels_dict)
        assert_shape_equal(batches, batch_labels)
        assert batches.shape[1] == FLAGS.batch_size, \
            "precomputed batches do not match specified batch size; generate a set of " \
            "batches matching this batch size using generate_diverse_batches.py first."

        batches = batches.applymap(
            lambda x: os.path.join(FLAGS.train_dir, str(filename_labels_dict[x]), x)
        )
        print("[INFO] preprocessed input batch files from train_dir:")
        print(batches.values[0, :])

        # We compute the number of training and validation batches,  build the
        # tf.data.Dataset; then preprocess and batch them.

        n_batches = batches.shape[0]
        n_train_batches = int(n_batches * (1 - FLAGS.val_frac))

        train_filenames = batches.values[:n_train_batches, :].flatten()
        train_labels = batch_labels.values[:n_train_batches, :].flatten()
        n_train = len(train_filenames)

        val_filenames = batches.values[n_train_batches:, :].flatten()
        val_labels = batch_labels.values[n_train_batches:, :].flatten()
        n_val = len(val_filenames)

        # Check that the train and validation files exist
        [assert_file_exists(fp) for fp in train_filenames]
        [assert_file_exists(fp) for fp in val_filenames]

        train_generator = LabeledBatchGenerator(train_filenames, train_labels)
        val_generator = LabeledBatchGenerator(val_filenames, val_labels)

        train_ds.from_filename_and_label_generator(train_generator.generator)
        val_ds = ImageDataset()
        val_ds.from_filename_and_label_generator(val_generator.generator)
        preprocess_args = {"repeat_forever": True, "batch_size": FLAGS.batch_size,
                           "shuffle": False}
        train_ds.preprocess(**preprocess_args)
        val_ds.preprocess(**preprocess_args)

    else:
        n_train_val = get_n_from_file_pattern(train_file_pattern)
        n_train, n_val = compute_n_train_n_val(n_train_val)

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

    vgg_model_base = vggface2_model(dropout_rate=FLAGS.dropout_rate)
    print("[INFO] {n_train} training observations; {n_val} validation observations"
          "{n_test} testing observations".format(n_train=n_train,
                                                 n_val=n_val,
                                                 n_test=n_test,
                                                 ))
    if not FLAGS.debug:
        steps_per_train_epoch = steps_per_epoch(n_train)
        steps_per_val_epoch = steps_per_epoch(n_val)
    else:
        print("[INFO] running in debug mode")
        steps_per_train_epoch = 1
        steps_per_val_epoch = 1

    train_ds.write_sample_batch("./debug/sample-batch-train-label{}.png".format(
        FLAGS.label_name))
    test_ds.write_sample_batch("./debug/sample-batch-test-label{}.png".format(
        FLAGS.label_name))

    # The metrics to optimize during training
    train_metrics_dict = get_train_metrics()
    # .evaluate() automatically prepends the loss(es), so it will always include at
    # least categorical_crossentropy (adversarial also adds the AT loss terms)
    train_metrics_names = ["categorical_crossentropy", ] + list(train_metrics_dict.keys())
    train_metrics = list(train_metrics_dict.values())

    model_compile_args = {
        "optimizer": tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate),
        "loss": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        "metrics": train_metrics
    }

    vgg_model_base.compile(**model_compile_args)
    vgg_model_base.summary()

    # Shared training arguments for the model fitting.
    train_args = {"steps_per_epoch": steps_per_train_epoch,
                  "epochs": FLAGS.epochs,
                  "validation_steps": steps_per_val_epoch}
    from dro.utils.training_utils import make_ckpt_filepath

    # Base model training
    if FLAGS.train_base:
        print("[INFO] training base model")
        callbacks_base = make_callbacks(FLAGS, is_adversarial=False)
        vgg_model_base.fit_generator(train_ds.dataset, callbacks=callbacks_base,
                                     validation_data=val_ds.dataset, **train_args)

        # Fetch preds and test labels; these are both numpy arrays of shape [n_test, 2]
        test_metrics = vgg_model_base.evaluate_generator(test_ds.dataset)
        assert len(train_metrics_names) == len(test_metrics)
        test_metrics_dict = OrderedDict(zip(train_metrics_names, test_metrics))
        write_test_metrics_to_csv(test_metrics_dict, FLAGS, is_adversarial=False)

    elif FLAGS.base_model_ckpt:
        # load the model from specified checkpoint path instead of training it
        vgg_model_base.load_weights(filepath=FLAGS.base_model_ckpt)
        # load the model from default checkpoint path instead of training it
    else:
        vgg_model_base.load_weights(filepath=make_ckpt_filepath(FLAGS,
                                                                is_adversarial=False))

    # Adversarial model training
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

    train_ds.convert_to_dictionaries()
    val_ds.convert_to_dictionaries()

    test_ds_adv = ImageDataset()
    test_ds_adv.from_files(test_file_pattern, shuffle=False)
    test_ds_adv.preprocess(repeat_forever=False, shuffle=False,
                           batch_size=FLAGS.batch_size)
    test_ds_adv.convert_to_dictionaries()

    if FLAGS.train_adversarial:
        print("[INFO] training adversarial model")
        callbacks_adv = make_callbacks(FLAGS, is_adversarial=True)
        adv_model.fit_generator(train_ds.dataset, callbacks=callbacks_adv,
                                validation_data=val_ds.dataset,
                                **train_args)
        test_metrics_adv = adv_model.evaluate_generator(test_ds_adv.dataset)
        # The evaluate_generator() function adds the total_loss and adversarial_loss,
        # so here we include those.
        test_metrics_adv_names = \
            ["total_combined_loss", ] + train_metrics_names + ["adversarial_loss", ]
        assert len(test_metrics_adv_names) == len(test_metrics_adv)
        test_metrics_adv = OrderedDict(zip(test_metrics_adv_names, test_metrics_adv))
        write_test_metrics_to_csv(test_metrics_adv, FLAGS, is_adversarial=True)

    elif FLAGS.adv_model_ckpt:  # load the model
        adv_model.load_weights(FLAGS.adv_model_ckpt)
    else:
        adv_model.load_weights(filepath=make_ckpt_filepath(FLAGS, is_adversarial=True))

    if FLAGS.perturbation_analysis:
        # First, create a reference model from the non-adversarially-trained model,
        # which will be used to generate perturbations.

        print("[INFO] generating adversarial samples to compare the models")
        from dro.utils.training_utils import perturb_and_evaluate, \
            make_compiled_reference_model
        from dro.utils.training_utils import make_model_uid
        from dro.utils.viz import show_adversarial_resuts
        reference_model = make_compiled_reference_model(vgg_model_base, adv_config,
                                                        model_compile_args)

        models_to_eval = {
            'base': vgg_model_base,
            'adv-regularized': adv_model.base_model
        }

        perturbed_images, labels, predictions, metrics = perturb_and_evaluate(
            test_ds_adv.dataset, models_to_eval, reference_model)

        adv_image_basename = "./debug/adv-examples-{}".format(make_model_uid(FLAGS))

        show_adversarial_resuts(n_batches=10,
                                perturbed_images=perturbed_images,
                                labels=labels,
                                predictions=predictions,
                                fp_basename=adv_image_basename,
                                batch_size=FLAGS.batch_size)


if __name__ == "__main__":
    app.run(main)
