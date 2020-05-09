import glob
import re
import time
from os import path as osp

import pandas as pd

# Regex used to parse the LFW filenames
from dro.datasets import ImageDataset
from dro.keys import FILENAME_COLNAME
from dro.utils.training_utils import pred_to_binary
from dro.utils.viz import show_batch

LFW_FILENAME_REGEX = re.compile("(\D+)_(\d{4})\.jpg")

LABEL_COLNAME = "label"
ATTR_COLNAME = "attr"


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
    assert len(annotated_files) > 0, "no files detected using {} at {}".format(anno_fp,
                                                                               test_dir)
    return annotated_files

def make_lfw_file_pattern(dirname):
    return osp.join(dirname, "*/*.jpg")


def make_pos_and_neg_attr_datasets(anno_fp, test_dir, label_name,
                                   slice_attribute_name,
                                   confidence_threshold, img_shape, batch_size,
                                   write_samples=True
                                   ):
    """Create a dict of datasets where the keys correspond to the binary attribute,
    and the values are tf.data.Datasets of the (image, label) tuples."""
    # build a labeled dataset from the files
    annotated_files = get_annotated_data_df(anno_fp=anno_fp,
                                            test_dir=test_dir)

    # Create a DataFrame with columns for (filename, label, slice_attribute); the columns
    # need to be renamed to generic LABEL_COLNAME and ATTR_COLNAME in order to allow
    # for cases where label and attribute names are the same (e.g. slicing 'Male'
    # prediction by 'Male' attribute).

    dset_df = annotated_files.reset_index()[
        [FILENAME_COLNAME, label_name, slice_attribute_name]]
    dset_df.columns = [FILENAME_COLNAME, LABEL_COLNAME, ATTR_COLNAME]

    # Apply thresholding. We want observations which have absolute value greater than some
    # threshold (predictions close to zero have low confidence).

    dset_df = apply_thresh(dset_df, LABEL_COLNAME,
                           confidence_threshold)
    dset_df = apply_thresh(dset_df, ATTR_COLNAME,
                           confidence_threshold)

    dset_df[LABEL_COLNAME] = dset_df[LABEL_COLNAME].apply(pred_to_binary)
    dset_df[ATTR_COLNAME] = dset_df[ATTR_COLNAME].apply(
        pred_to_binary)

    # Break the input dataset into separate tf.Datasets based on the value of the slice
    # attribute.

    # Create and preprocess the dataset of examples where ATTR_COLNAME == 1
    preprocessing_kwargs = {"shuffle": False, "repeat_forever": False, "batch_size":
        batch_size}

    dset_attr_pos = ImageDataset(img_shape)
    dset_attr_pos.from_dataframe(dset_df[dset_df[ATTR_COLNAME] == 1],
                                 label_name=LABEL_COLNAME)
    dset_attr_pos.preprocess(**preprocessing_kwargs)

    # Create and process the dataset of examples where ATTR_COLNAME == 1
    dset_attr_neg = ImageDataset(img_shape)
    dset_attr_neg.from_dataframe(dset_df[dset_df[ATTR_COLNAME] == 0],
                                 label_name=LABEL_COLNAME)
    dset_attr_neg.preprocess(**preprocessing_kwargs)

    if write_samples:
        print("[INFO] writing sample batches; this will fail if eager execution is "
              "disabled")
        image_batch, label_batch = next(iter(dset_attr_pos.dataset))
        show_batch(image_batch.numpy(), label_batch.numpy(),
                   fp="./debug/sample_batch_attr{}1-label{}-{}.png".format(
                       slice_attribute_name, label_name, int(time.time()))
                   )
        image_batch, label_batch = next(iter(dset_attr_neg.dataset))
        show_batch(image_batch.numpy(), label_batch.numpy(),
                   fp="./debug/sample_batch_attr{}0-label{}-{}.png".format(
                       slice_attribute_name, label_name, int(time.time()))
                   )
    return {"1": dset_attr_pos, "0": dset_attr_neg}