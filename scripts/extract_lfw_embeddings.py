"""

usage:
python3 scripts/extract_lfw_embeddings.py \
    --label_name None \
    --train_dir /projects/grail/jpgard/lfw/lfw-deepfunneled-cropped \
    --model_type facenet
"""

from absl import app
from absl import flags
import glob
import os

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np

from dro.datasets import process_path, preprocess_dataset
from dro.utils.training_utils import get_model_img_shape_from_flags, get_model_from_flags
from dro.utils.flags import define_training_flags
from dro.utils.lfw import make_lfw_file_pattern
from dro.training.models import facenet_model

tf.compat.v1.enable_eager_execution()

define_training_flags()

FLAGS = flags.FLAGS


def fp_generator(image_ids):
    for fp in image_ids:
        yield fp


def label_generator(image_ids):
    for fp in image_ids:
        yield os.path.relpath(fp, start=FLAGS.train_dir)


def main(argv):
    # Generate the datasets
    train_file_pattern = make_lfw_file_pattern(FLAGS.train_dir)
    img_shape = get_model_img_shape_from_flags(FLAGS)
    image_ids = glob.glob(train_file_pattern, recursive=True)
    fp_gen = fp_generator(image_ids)
    id_gen = label_generator(image_ids)
    dset_x = tf.data.Dataset.from_generator(lambda: fp_gen,
                                            output_types=tf.string)
    dset_y = tf.data.Dataset.from_generator(lambda: id_gen,
                                            output_types=tf.string)

    def _process_path(file_path):
        return process_path(file_path, output_shape=img_shape, random_crop=False,
                            labels=False)

    dset_x = dset_x.map(_process_path)
    # Preprocess (batching, etc.)
    preprocess_args = {"repeat_forever": False, "batch_size": FLAGS.batch_size,
                                "shuffle": False}
    dset_x = preprocess_dataset(dset_x, **preprocess_args)
    dset_y = preprocess_dataset(dset_y, **preprocess_args)
    dset_y = tfds.as_numpy(dset_y)
    batch_labels = [i for i in dset_y]
    labels = np.concatenate(batch_labels)

    # Fetch the embeddings
    model = facenet_model(dropout_rate=FLAGS.dropout_rate, fc_sizes=None)
    embeddings = model.predict(dset_x)
    # Read the embeddings and labels into memory
    batch_embeddings = [i for i in embeddings]
    embeddings_df = pd.DataFrame(batch_embeddings, index=labels)
    embeddings_df.to_csv("embeddings/lfw_embeddings.csv", index=True)


if __name__ == "__main__":
    app.run(main)
