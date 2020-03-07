"""

A script to find vggface2 faces with annotations, and move them into directories by
label. This fits with existing tensorflow pipelines, which assume data are partitioned
by label.
Usage

python scripts/partition_vggface2_by_label.py \
    --img_dir /Users/jpgard/Documents/research/vggface2/train \
    --anno_fp /Users/jpgard/Documents/research/vggface2/anno/11-Mouth_Open.txt \
    --out_dir /Users/jpgard/Documents/research/vggface2/train_partitioned_by_label/mouth_open
"""

import shutil

import pandas as pd
import glob
from absl import app
from absl import flags
import numpy as np
from dro.datasets import train_test_val_split
import re
import tensorflow as tf
import os

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("epochs", 5, "the number of training epochs")
flags.DEFINE_string("img_dir", None, "directory containing the aligned celeba images")
flags.DEFINE_string("out_dir", None, "directory to output")
flags.DEFINE_string("anno_fp", None, "path to the target annotation file.")


def main(argv):
    attributes_df = pd.read_csv(FLAGS.anno_fp, delimiter="\t", index_col=0)
    img_files = glob.glob(FLAGS.img_dir + "/*/*.jpg")
    missing_file_count = 0
    found_file_count = 0
    for f in img_files:
        img_key = re.match(".*/(.*/.*.jpg)", f).group(1)
        id_num = re.match(".*/(.*)/.*.jpg", f).group(1)
        try:
            label = attributes_df.loc[img_key][0]
            dest_dir = os.path.join(FLAGS.out_dir, str(label), id_num)
            os.makedirs(dest_dir, exist_ok=True)
            dest_fp = os.path.join(dest_dir, os.path.basename(f))
            print("copying from {} to {}".format(f, dest_fp))
            shutil.copy(f, dest_fp)
            found_file_count += 1
        except KeyError:
            # print("key {} not found in annotations; skipping".format(img_key))
            missing_file_count += 1
            continue
    print("found {} files; missing {} files".format(found_file_count, missing_file_count))


if __name__ == "__main__":
    app.run(main)
