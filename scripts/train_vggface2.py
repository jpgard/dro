"""
Script to fine-tune pretrained VGGFace2 model.

usage:
python scripts/train_vggface2.py \
    --img_dir /Users/jpgard/Documents/research/vggface2/train_partitioned_by_label
    /mouth_open
"""

import pandas as pd
import glob
from absl import app
from absl import flags
import numpy as np
from dro.datasets import train_test_val_split
import re
import tensorflow as tf
import os
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("epochs", 5, "the number of training epochs")
flags.DEFINE_string("img_dir", None, "directory containing the aligned celeba images")

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CLASS_NAMES = np.array(["0", "1"])

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(6, 14))
    for n in range(16):
        ax = plt.subplot(8, 2, n + 1)
        plt.imshow(image_batch[n])
        plt.title(str(label_batch[n]))
        plt.axis('off')
    plt.show()


def main(argv):
    list_ds = tf.data.Dataset.list_files(str(FLAGS.img_dir + '/*/*/*.jpg'), shuffle=True,
                                         seed=2974)
    # for f in list_ds.take(3):
    #     print(f.numpy())

    def get_label(file_path):
        # convert the path to a list of path components
        label = tf.strings.substr(file_path, -21, 1)
        # The second to last is the class-directory
        return tf.strings.to_number(label)

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize to a square image of 256 x 256, then crop to random 224 x 224
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_with_crop_or_pad(img, target_height=256, target_width=256)
        img = tf.squeeze(img, axis=0)
        img = tf.image.random_crop(img, (224, 224, 3))
        return img

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # for image, label in labeled_ds.take(10):
    #     print("Image shape: ", image.numpy().shape)
    #     print("Label: ", label.numpy())

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(FLAGS.batch_size)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    # if the data doesn't fit in memory see the example usage at the
    # bottom of the page here:
    # https: // www.tensorflow.org / tutorials / load_data / images

    train_ds = prepare_for_training(labeled_ds)
    # image_batch, label_batch = next(iter(train_ds))
    # show_batch(image_batch.numpy(), label_batch.numpy())
    # import ipdb;ipdb.set_trace()


    # Disable eager
    # tf.compat.v1.disable_eager_execution()

    from tensorflow.keras import Model
    from tensorflow.keras.layers import Flatten, Dense, Input
    from keras_vggface.vggface import VGGFace
    # Convolution Features
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    # set the vgg_model layers to non-trainable
    for layer in vgg_model.layers:
        layer.trainable = False
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(32, activation='relu', name='fc6')(x)
    x = Dense(16, activation='relu', name='fc7')(x)
    out = Dense(1, activation='sigmoid', name='fc8')(x)
    custom_vgg_model = Model(vgg_model.input, out)
    custom_vgg_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                             metrics=['accuracy']
                             )
    custom_vgg_model.summary()
    custom_vgg_model.fit_generator(train_ds, steps_per_epoch=1000/FLAGS.batch_size)


if __name__ == "__main__":
    app.run(main)
