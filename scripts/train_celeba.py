"""
Usage:
python scripts/train_celeba.py \
    --img_dir /Users/jpgard/Documents/research/celeba/img/img_align_celeba \
    --attributes_fp /Users/jpgard/Documents/research/celeba/anno/list_attr_celeba.txt
"""

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from absl import flags
from absl import app

from dro.utils.viz import plot_faces
from dro.datasets import make_dataset
from dro.training.models import facenet_model

tf.compat.v1.enable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_string("target_colname", "Smiling",
                    help="The name of the target colname; should match a column name in "
                         "the celeba attributes.txt file.")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("epochs", 5, "the number of training epochs")
flags.DEFINE_string("img_dir", None, "directory containing the aligned celeba images")
flags.DEFINE_string("attributes_fp", None, "path to the celeba attributes file.")


def main(argv):
    # This is the default size for the cropped celeba images, no reduction is applied.
    img_shape = (218, 178, 3)

    n_train = FLAGS.batch_size * 50
    n_val = FLAGS.batch_size * 10
    n_test = FLAGS.batch_size * 5

    # Fetch jpg files for training and testing
    img_files = np.sort(os.listdir(FLAGS.img_dir))
    img_files_train = img_files[:n_train]
    img_files_val = img_files[n_train:n_train + n_val]
    img_files_test = img_files[n_train + n_val: n_train + n_val + n_test]

    attributes_df = pd.read_csv(FLAGS.attributes_fp, delim_whitespace=True, skiprows=0,
                                header=1).sort_index().replace(-1, 0)
    attributes_df[FLAGS.target_colname].copy()
    assert attributes_df.shape[0] == len(img_files), \
        "possible mismatch between training images and attributes."

    # Assemble the datasets. Note that this loads images into memory, which may not be
    # advisable for large datasets.
    train_dataset = make_dataset(img_files_train, FLAGS.batch_size, attributes_df,
                                 FLAGS.target_colname, img_dir=FLAGS.img_dir,
                                 img_shape=img_shape)
    val_dataset = make_dataset(img_files_val, FLAGS.batch_size, attributes_df,
                               FLAGS.target_colname, img_dir=FLAGS.img_dir,
                               img_shape=img_shape)
    test_dataset = make_dataset(img_files_test, FLAGS.batch_size, attributes_df,
                                FLAGS.target_colname, img_dir=FLAGS.img_dir,
                                img_shape=img_shape)

    # show a visualization of the first few faces
    # plot_faces(img_train)

    model = facenet_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.build(input_shape=(None, 218, 178, 3))
    model.summary()
    model.fit(train_dataset, epochs=FLAGS.epochs,
              steps_per_epoch=n_train // FLAGS.batch_size,
              validation_data=val_dataset,
              validation_steps=n_val // FLAGS.batch_size)


if __name__ == "__main__":
    app.run(main)
