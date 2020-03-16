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
import glob
import os.path as osp
import re

import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string("anno_fp", None, "path to annotations file for evaluation.")
flags.DEFINE_string("test_dir", None, "directory containing the test images")

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

def main(argv):
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
    import ipdb;
    ipdb.set_trace()
    res = anno_df.join(files_df, how='inner')


if __name__ == "__main__":
    app.run(main)
