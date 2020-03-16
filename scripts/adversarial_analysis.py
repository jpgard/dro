"""
A script to conduct adversarial analysis of pre-trained models.

The script applies a range of adversarial perturbations to a set of test images from
the LFW dataset, and evaluates classifier accuracy on those images. Accuracy is
reported  by image
subgroups.

usage:
python3 scripts/adversarial_analysis.py \
--anno_fp /Users/jpgard/Documents/research/lfw/lfw_attributes_cleaned.txt \
--test_dir /Users/jpgard/Documents/research/lfw/lfw-deepfunneled-a \
--label_name Mouth_Open
"""

from absl import app, flags
from functools import partial
import glob
import os.path as osp
import re
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from dro.utils.training_utils import process_path, preprocess_dataset
from dro.training.models import vggface2_model
import neural_structured_learning as nsl
from dro.keys import IMAGE_INPUT_NAME, LABEL_INPUT_NAME
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


FLAGS = flags.FLAGS

flags.DEFINE_string("anno_fp", None, "path to annotations file for evaluation.")
flags.DEFINE_string("test_dir", None, "directory containing the test images")
flags.DEFINE_string("slice_attribute_name", "Black",
                    "attribute name to use from annotations.")
flags.DEFINE_string("label_name", None,
                    "name of the prediction label (e.g. sunglasses, mouth_open) in the "
                    "LFW/test dataset",
                    )
flags.DEFINE_float("confidence_threshold", 0.5, "only predictions with absolute value "
                                                ">= this threshold are used ("
                                                "predictions are centered around zero) "
                                                "in order to ensure high-quality labels.")

# the vggface2/training parameters
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("epochs", 250, "the number of training epochs")
flags.DEFINE_string("ckpt_dir", "./training-logs", "directory to save/load checkpoints "
                                                   "from")
flags.DEFINE_float("learning_rate", 0.01, "learning rate to use")
flags.DEFINE_float("dropout_rate", 0.8, "dropout rate to use in fully-connected layers")
flags.mark_flag_as_required("label_name")

# the adversarial training parameters
flags.DEFINE_float('adv_multiplier', 0.2,
                   " The weight of adversarial loss in the training objective, relative "
                   "to the labeled loss. e.g. if this is 0.2, The model minimizes "
                   "(mean_crossentropy_loss + 0.2 * adversarial_regularization) ")
flags.DEFINE_float('adv_step_size', 0.2, "The magnitude of adversarial perturbation.")
flags.DEFINE_string('adv_grad_norm', 'infinity',
                    "The norm to measure the magnitude of adversarial perturbation.")



# Mapping between names of the features in the different datasets
LFW_TO_VGG_LABEL_MAPPING = {
    "Mouth_Open": "Mouth_Open"
}

# Regex used to parse the LFW filenames
lfw_filename_regex = re.compile("(\w+_\w+)_(\d{4})\.jpg")


def extract_person_from_filename(x):
    """Helper function to extract the person name from a LFW filename."""
    res = re.match(lfw_filename_regex, osp.basename(x))
    try:
        return res.group(1)
    except AttributeError:
        return None


def extract_imagenum_from_filename(x):
    """Helper function to extract the image number from a LFW filename."""
    res = re.match(lfw_filename_regex, osp.basename(x))
    try:
        return res.group(2)
    except AttributeError:
        return None


def get_annotated_data_df():
    """Fetch and preprocess the dataframe of LFW annotations and their corresponding
    filenames."""
    # get the annotated files
    anno_df = pd.read_csv(FLAGS.anno_fp, delimiter="\t")
    anno_df['imagenum_str'] = anno_df['imagenum'].apply(lambda x: f'{x:04}')
    anno_df['person'] = anno_df['person'].apply(lambda x: x.replace(" ", "_"))
    anno_df.set_index(['person', 'imagenum_str'], inplace=True)
    anno_df["Mouth_Open"] = 1 - anno_df["Mouth Closed"]
    # Read the files, dropping any images which cannot be parsed
    files = glob.glob(FLAGS.test_dir + "/*/*.jpg", recursive=True)
    files_df = pd.DataFrame(files, columns=['filename'])
    files_df['person'] = files_df['filename'].apply(extract_person_from_filename)
    files_df['imagenum_str'] = files_df['filename'].apply(extract_imagenum_from_filename)
    files_df.dropna(inplace=True)
    files_df.set_index(['person', 'imagenum_str'], inplace=True)
    annotated_files = anno_df.join(files_df, how='inner')
    return annotated_files


def pred_to_binary(x, thresh=0.):
    """Convert lfw predictions to binary (0.,1.) labels by thresholding based on
    thresh."""
    return int(x > thresh)


def preprocess_path(x, y):
    x = process_path(x, labels=False)
    y = tf.one_hot(y, 2)
    return x, y


def build_dataset_from_dataframe(df):
    # dset starts as tuples of (filename, label_as_float)
    dset = tf.data.Dataset.from_tensor_slices(
        (df['filename'].values,
         df[FLAGS.label_name].values)
    )
    _process_path = partial(process_path, labels=False)
    dset = dset.map(preprocess_path)
    return dset


def apply_thresh(df, colname):
    return df[abs(df[colname]) >= FLAGS.confidence_threshold]


def main(argv):
    # build a labeled dataset from the files
    annotated_files = get_annotated_data_df()
    dset_df = annotated_files.reset_index()[
        ['filename', FLAGS.label_name, FLAGS.slice_attribute_name]]
    # Show histograms of the distributions
    dset_df[FLAGS.label_name].hist(bins=25)
    plt.title(FLAGS.label_name)
    plt.show()
    dset_df[FLAGS.slice_attribute_name].hist(bins=25)
    plt.title(FLAGS.slice_attribute_name)
    plt.show()
    # Apply thresholding. We want observations which have absolute value greater than some
    # threshold (predictions close to zero have low confidence). Need to inspect
    # the distributions a bit to decide a good threshold for each feature.
    dset_df = apply_thresh(dset_df, FLAGS.label_name)
    dset_df = apply_thresh(dset_df, FLAGS.slice_attribute_name)

    dset_df[FLAGS.label_name] = dset_df[FLAGS.label_name].apply(pred_to_binary)
    dset_df[FLAGS.slice_attribute_name] = dset_df[FLAGS.slice_attribute_name].apply(
        pred_to_binary)

    # Break the input dataset into separate tf.Datasets based on the value of the slice
    # attribute.
    dset_attr_pos = build_dataset_from_dataframe(
        dset_df[dset_df[FLAGS.slice_attribute_name] == 1]
    )
    dset_attr_pos = preprocess_dataset(dset_attr_pos, shuffle=False,
                                       repeat_forever=False, batch_size=FLAGS.batch_size)
    dset_attr_neg = build_dataset_from_dataframe(
        dset_df[dset_df[FLAGS.slice_attribute_name] == 0]
    )
    dset_attr_neg = preprocess_dataset(dset_attr_neg, shuffle=False,
                                       repeat_forever=False, batch_size=FLAGS.batch_size)
    from dro.utils.viz import show_batch
    image_batch, label_batch = next(iter(dset_attr_pos))
    show_batch(image_batch.numpy(), label_batch.numpy(),
               fp="./debug/sample_batch_attr{}1-label{}-{}.png".format(
                   FLAGS.slice_attribute_name, FLAGS.label_name, int(time.time()))
               )

    image_batch, label_batch = next(iter(dset_attr_neg))
    show_batch(image_batch.numpy(), label_batch.numpy(),
               fp="./debug/sample_batch_attr{}0-label{}-{}.png".format(
                   FLAGS.slice_attribute_name, FLAGS.label_name, int(time.time()))
               )
    from dro.utils.training_utils import convert_to_dictionaries

    # Convert the datasets into dicts for use in adversarial model.
    dset_attr_neg = dset_attr_neg.map(convert_to_dictionaries)
    dset_attr_pos = dset_attr_pos.map(convert_to_dictionaries)

    # load the models
    from dro.utils.training_utils import get_train_metrics
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
    from dro.utils.training_utils import make_ckpt_filepath
    vgg_model_base.load_weights(filepath=make_ckpt_filepath(FLAGS, is_adversarial=False))

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
    adv_model.load_weights(filepath=make_ckpt_filepath(FLAGS, is_adversarial=True))

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
    print("[INFO] perturbing inputs and evaluating the model")
    for id, dset in zip(["1", "0"], [dset_attr_pos, dset_attr_neg]):
        perturbed_images, labels, predictions, metrics = perturb_and_evaluate(
            dset, models_to_eval, reference_model)

        batch_index = 0
        batch_image = perturbed_images[batch_index]
        batch_label = labels[batch_index]
        batch_pred = predictions[batch_index]

        n_col = 4
        n_row = (FLAGS.batch_size + n_col - 1) / n_col

        print('accuracy in batch %d:' % batch_index)
        for name, pred in batch_pred.items():
            print('%s model: %d / %d' % (
                name, np.sum(batch_label == pred), FLAGS.batch_size))

        adv_image_fp = "./debug/adv-examples-{}-{}-{}.png".format(
            make_model_uid(FLAGS), FLAGS.slice_attribute_name, id)
        show_adversarial_resuts(batch_image, batch_label,
                                batch_pred, adv_image_fp=adv_image_fp, n_row=n_row,
                                n_col=n_col)


if __name__ == "__main__":
    print("running")
    app.run(main)
