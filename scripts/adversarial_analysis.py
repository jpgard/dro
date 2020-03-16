"""
A script to conduct adversarial analysis of pre-trained models.

The script applies a range of adversarial perturbations to a set of test images from
the LFW dataset, and evaluates classifier accuracy on those images. Accuracy is
reported  by image
subgroups.

usage:
python3 scripts/adversarial_analysis.py \
--anno_fp /Users/jpgard/Documents/research/lfw/lfw_attributes_cleaned.txt \
--test_dir /Users/jpgard/Documents/research/lfw/lfw-deepfunneled-a
"""

from absl import app, flags
from functools import partial
import glob
import os.path as osp
import re

import pandas as pd
import tensorflow as tf
from dro.utils.training_utils import process_path

FLAGS = flags.FLAGS

flags.DEFINE_string("anno_fp", None, "path to annotations file for evaluation.")
flags.DEFINE_string("test_dir", None, "directory containing the test images")
flags.DEFINE_string("slice_attribute_name", "Black",
                    "attribute name to use from annotations.")
flags.DEFINE_string("label_name", "Smiling", "label name to use from annotations.")

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
    return float(x > thresh)


def main(argv):
    # build a labeled dataset from the files
    annotated_files = get_annotated_data_df()
    import ipdb;
    ipdb.set_trace()
    dset_df = annotated_files.reset_index()[
        ['filename', FLAGS.label_name, FLAGS.slice_attribute_name]]
    dset_df[FLAGS.label_name] = dset_df[FLAGS.label_name].apply(pred_to_binary)
    dset_df[FLAGS.slice_attribute_name] = dset_df[FLAGS.slice_attribute_name].apply(
        pred_to_binary)

    # dset starts as tuples of (filename, label_as_float, slice_as_float)
    dset = tf.data.Dataset.from_generator(
        dset_df.itertuples(index=False, name=None),
        output_types=[tf.string, tf.float32, tf.float32])
    _process_path = partial(process_path, labels=False)
    dset = dset.map(lambda x, y, z:
                    (_process_path(x), tf.one_hot(y, 2), z))
    positive_attr_dset = dset.filter(lambda x, y, z: z == 1.)
    negative_attr_dset = dset.filter(lambda x, y, z: z == 0.)

if __name__ == "__main__":
    print("running")
    app.run(main)
