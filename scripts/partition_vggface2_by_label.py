"""

A script to find vggface2 faces with annotations, and move them into directories by
label. This fits with existing tensorflow pipelines, which assume data are partitioned
by label.
Usage

python3 scripts/partition_vggface2_by_label.py \
    --img_dir /projects/grail/jpgard/vggface2/train \
    --out_dir /projects/grail/jpgard/vggface2/annotated_partitioned_by_label \
    --anno_dir /projects/grail/jpgard/vggface2/anno

To find the number of (train, test) files, run:
find /projects/grail/jpgard/vggface2/annotated_partitioned_by_label/test -name "*.jpg" | wc -l
find /projects/grail/jpgard/vggface2/annotated_partitioned_by_label/train -name "*.jpg" | wc -l
"""

import shutil

import pandas as pd
import glob
from absl import app
from absl import flags
import re
import os
import random

from sklearn.model_selection import train_test_split

FLAGS = flags.FLAGS
flags.DEFINE_string("img_dir", None, "directory containing the aligned celeba images")
flags.DEFINE_string("out_dir", None, "directory to output")
flags.DEFINE_string("anno_dir", None, "path to the target annotation file.")
flags.DEFINE_float("train_frac", 0.9, "proportion of persons (not image!) to allocate to "
                                      "training; the rest is for testing. All images "
                                      "for a given "
                                      "person are allocated to either train/test to "
                                      "prevent leakage.")

SEED = 29749


def make_annotations_df(anno_dir):
    """Build a single DataFrame with all annotations."""
    annotations = []
    for f in os.listdir(anno_dir):
        if (not f.startswith(".")) and f.endswith(".txt"):
            label = re.match("\d+-(.*)\.txt", f).group(1)
            attributes_df = pd.read_csv(os.path.join(FLAGS.anno_dir, f),
                                        delimiter="\t",
                                        index_col=0)
            attributes_df.columns = [label, ]
            annotations.append(attributes_df)
    return pd.concat(annotations, axis=1)


def get_key_from_fp(fp):
    """Get the key that uniquely identifies the image; user_id/image_id.jpg"""
    try:
        return re.match(".*/(.*/.*.jpg)", fp).group(1)
    except:
        return None


def get_person_id_from_fp(fp):
    """Get the key that uniquely identifies the user in the image."""
    try:
        return re.match(".*/(.*)/.*.jpg", fp).group(1)
    except:
        return None


def process_file(img_file: str, label_name: str, label, source_dir, dest_dir,
                 is_training: bool):
    train_test_dirs = {True: "train", False: "test"}
    source_file = os.path.join(source_dir, img_file)
    person_id = re.match(".*/(.*)/.*.jpg", img_file).group(1)
    img_dest_dir = os.path.join(dest_dir, train_test_dirs[is_training],
                                label_name, str(label), person_id)
    os.makedirs(img_dest_dir, exist_ok=True)
    dest_fp = os.path.join(img_dest_dir, os.path.basename(img_file))
    print("[INFO] copying from {} to {}".format(source_file, dest_fp))
    shutil.copy(source_file, dest_fp)


def main(argv):
    attributes_df = make_annotations_df(FLAGS.anno_dir)
    img_files = glob.glob(FLAGS.img_dir + "/*/*.jpg")
    assert len(img_files) > 0
    annotated_files = [i for i in img_files if get_key_from_fp(i) in attributes_df.index]
    assert len(annotated_files) > 0
    pids = set([get_person_id_from_fp(f) for f in annotated_files])
    n_train = int(FLAGS.train_frac * len(pids))
    random.seed(SEED)
    train_pids = random.sample(pids, n_train)
    test_pids = [pid for pid in pids if pid not in train_pids]
    files_and_labels = [(k, attributes_df.loc[get_key_from_fp(k), :].tolist(),
                         get_person_id_from_fp(k) in train_pids)
                        for k in annotated_files]
    label_names = attributes_df.columns
    for img_file, img_labels, is_training in files_and_labels:
        for label_name, label in zip(label_names, img_labels):
            process_file(img_file, label_name, label, FLAGS.img_dir, FLAGS.out_dir,
                         is_training)


if __name__ == "__main__":
    app.run(main)
