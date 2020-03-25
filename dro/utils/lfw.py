import glob
import re
from functools import partial
from os import path as osp

import pandas as pd
import tensorflow as tf
import numpy as np

from dro.datasets import process_path

# Regex used to parse the LFW filenames
LFW_FILENAME_REGEX = re.compile("(\w+_\w+)_(\d{4})\.jpg")

LABEL_COLNAME = "label"
ATTR_COLNAME = "attr"
FILENAME_COLNAME = "filename"


def extract_person_from_filename(x):
    """Helper function to extract the person name from a LFW filename."""
    res = re.match(LFW_FILENAME_REGEX, osp.basename(x))
    try:
        return res.group(1)
    except AttributeError:
        return None


def extract_imagenum_from_filename(x):
    """Helper function to extract the image number from a LFW filename."""
    res = re.match(LFW_FILENAME_REGEX, osp.basename(x))
    try:
        return res.group(2)
    except AttributeError:
        return None


def preprocess_path(x, y):
    x = process_path(x, labels=False)
    y = tf.one_hot(y, 2)
    return x, y


def build_dataset_from_dataframe(df, label_name):
    # dset starts as tuples of (filename, label_as_float)
    dset = tf.data.Dataset.from_tensor_slices(
        (df['filename'].values,
         df[label_name].values.astype(np.int))
    )
    dset = dset.map(preprocess_path)
    return dset


def apply_thresh(df, colname, thresh: float):
    return df[abs(df[colname]) >= thresh]


def get_annotated_data_df(anno_fp, test_dir, filepattern="/*/*.jpg"):
    """Fetch and preprocess the dataframe of LFW annotations and their corresponding
    filenames.

    Returns: A pd.DataFrame indexed by person_id and image_num, with columns for each
    attribute.
    """
    # get the annotated files
    anno_df = pd.read_csv(anno_fp, delimiter="\t")
    anno_df['imagenum_str'] = anno_df['imagenum'].apply(lambda x: f'{x:04}')
    anno_df['person'] = anno_df['person'].apply(lambda x: x.replace(" ", "_"))
    anno_df.set_index(['person', 'imagenum_str'], inplace=True)
    anno_df["Mouth_Open"] = 1 - anno_df["Mouth Closed"]
    # Read the files, dropping any images which cannot be parsed
    files = glob.glob(test_dir + filepattern, recursive=True)
    assert len(files) > 0, "no files detected with pattern {}".format(test_dir + filepattern)
    files_df = pd.DataFrame(files, columns=['filename'])
    files_df['person'] = files_df['filename'].apply(extract_person_from_filename)
    files_df['imagenum_str'] = files_df['filename'].apply(extract_imagenum_from_filename)
    files_df.dropna(inplace=True)
    files_df.set_index(['person', 'imagenum_str'], inplace=True)
    annotated_files = anno_df.join(files_df, how='inner')
    return annotated_files
