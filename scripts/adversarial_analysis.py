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

import pandas as pd
import tensorflow as tf
from dro.utils.training_utils import process_path, preprocess_dataset
from dro.training.models import vggface2_model
import neural_structured_learning as nsl
from dro.keys import IMAGE_INPUT_NAME, LABEL_INPUT_NAME

tf.compat.v1.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_string("anno_fp", None, "path to annotations file for evaluation.")
flags.DEFINE_string("test_dir", None, "directory containing the test images")
flags.DEFINE_string("slice_attribute_name", "Black",
                    "attribute name to use from annotations.")
flags.DEFINE_string("label_name", None,
                    "name of the prediction label (e.g. sunglasses, mouth_open) in the "
                    "LFW/test dataset",
                    )

# the vggface2/training parameters
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("epochs", 250, "the number of training epochs")
flags.DEFINE_string("ckpt_dir", "./training-logs", "directory to save/load checkpoints "
                                                   "from")
flags.DEFINE_float("learning_rate", 0.01, "learning rate to use")
flags.DEFINE_float("dropout_rate", 0.8, "dropout rate to use in fully-connected layers")
flags.mark_flag_as_required("label_name")

# Mapping between names of the features in the different datasets
LFW_TO_VGG_LABEL_MAPPING = {
    "Mouth_Open": "Mouth_Open"
}

# Regex used to parse the LFW filenames
lfw_filename_regex = re.compile("(\w+_\w+)_(\d{4})\.jpg")


def extract_person_from_filename(x):
    res = re.match(lfw_filename_regex, osp.basename(x))
    try:
        return res.group(1)
    except AttributeError:
        return None


def extract_imagenum_from_filename(x):
    res = re.match(lfw_filename_regex, osp.basename(x))
    try:
        return res.group(2)
    except AttributeError:
        return None


def get_annotated_data_df():
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


def filter_neg(x, y, z):
    # import ipdb;
    # ipdb.set_trace()
    label = tf.unstack(z)
    label = label[0]
    result = tf.equal(label, 0)
    return result


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


def main(argv):
    # build a labeled dataset from the files
    annotated_files = get_annotated_data_df()
    dset_df = annotated_files.reset_index()[
        ['filename', FLAGS.label_name, FLAGS.slice_attribute_name]]
    dset_df[FLAGS.label_name] = dset_df[FLAGS.label_name].apply(pred_to_binary)
    dset_df[FLAGS.slice_attribute_name] = dset_df[FLAGS.slice_attribute_name].apply(
        pred_to_binary)

    # Break the input dataset into separate tf.Datasets based on the value of the slice
    # attribute.
    dset_attr_pos = build_dataset_from_dataframe(
        dset_df[dset_df[FLAGS.slice_attribute_name] == 1]
    )
    dset_attr_neg = build_dataset_from_dataframe(
        dset_df[dset_df[FLAGS.slice_attribute_name] == 0]
    )

    # TODO(jpgard): filter the datasets here.
    print("majority group dataset:")
    import matplotlib.pyplot as plt
    for x, y in dset_attr_pos.take(1):
        print("x: ", x.numpy())
        print("y: ", y.numpy())
        plt.imshow(x.numpy())
        plt.show()

    print("minority group dataset:")
    for x, y in dset_attr_neg.take(1):
        print("x: ", x.numpy())
        print("y: ", y.numpy())
        plt.imshow(x.numpy())
        plt.show()

    import ipdb;
    ipdb.set_trace()

    # for testing
    iterator = dset_attr_neg.make_one_shot_iterator()
    sample = iterator.get_next()

    dset = preprocess_dataset(dset, shuffle=False, repeat_forever=False,
                              batch_size=None)

    # TODO(jpgard): convert the (separate) datasets for the demographic/attribute
    #  groups into dicts foruse in adversarial model.

    # TODO: load models and create adversarial examples from them.

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

    perturbed_images, labels, predictions, metrics = perturb_and_evaluate(
        test_ds_adv, models_to_eval, reference_model)


if __name__ == "__main__":
    print("running")
    app.run(main)
