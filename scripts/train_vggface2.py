"""
Script to fine-tune pretrained VGGFace2 model.

usage:

# set the gpu
export GPU_ID="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# run the script
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
import math
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Activation, Dropout
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, \
    FalsePositives, FalseNegatives
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from keras_vggface.vggface import VGGFace

# tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("epochs", 5, "the number of training epochs")
flags.DEFINE_string("img_dir", None, "directory containing the aligned celeba images")
flags.DEFINE_float("learning_rate", 0.001, "learning rate to use")
flags.DEFINE_float("dropout_rate", 0.8, "dropout rate to use in fully-connected layers")

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CLASS_NAMES = np.array(["0", "1"])


class VGGModel:
    def __init__(self, data_X, data_y):
        self._create_architecture(data_X, data_y)

    def _create_architecture(self, data_X, data_y):
        logits = self._create_model(data_X)
        logits = tf.squeeze(logits)
        predictions = tf.round(tf.nn.sigmoid(logits))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=data_y, logits=logits))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, data_y),
                                               tf.float32))

    def _create_model(self, X):
        # Convolution Features
        vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
        # set the vgg_model layers to non-trainable
        for layer in vgg_model.layers:
            layer.trainable = False
        last_layer = vgg_model.get_layer('pool5').output
        # Classification block
        x = Flatten(name='flatten')(last_layer)
        x = Dense(4096, name='fc6')(x)
        x = Activation('relu', name='fc6/relu')(x)
        x = Dropout(rate=FLAGS.dropout_rate)(x)
        x = Dense(256, name='fc7')(x)
        x = Activation('relu', name='fc7/relu')(x)
        x = Dropout(rate=FLAGS.dropout_rate)(x)
        x = Dense(1, name='fc8')(x)
        out = Activation('sigmoid', name='fc8/sigmoid')(x)

        custom_vgg_model = Model(vgg_model.input, out)

        return custom_vgg_model(X)


def main(argv):
    list_ds = tf.data.Dataset.list_files(str(FLAGS.img_dir + '/*/*/*.jpg'), shuffle=True,
                                         seed=2974)

    def make_model_uid():
        model_uid = """bs{batch_size}e{epochs}lr{lr}dropout{dropout_rate}""".format(
            batch_size=FLAGS.batch_size,
            epochs=FLAGS.epochs,
            lr=FLAGS.learning_rate,
            dropout_rate=FLAGS.dropout_rate
        )

    uid = make_model_uid()

    # for f in list_ds.take(3):
    #     print(f.numpy())

    def get_label(file_path):
        # convert the path to a list of path components
        label = tf.strings.substr(file_path, -21, 1)
        # The second to last is the class-directory
        return tf.strings.to_number(label)

    def decode_img(img, normalize_by_channel=False):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize to a square image of 256 x 256, then crop to random 224 x 224
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_images(img, size=(256, 256), preserve_aspect_ratio=True)
        img = tf.image.resize_with_crop_or_pad(img, target_height=256, target_width=256)
        img = tf.squeeze(img, axis=0)
        img = tf.image.random_crop(img, (224, 224, 3))

        # Apply normalization: subtract the channel-wise mean from each image as in
        # https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/utils.py ;
        # divide means by 255.0 since the conversion above restricts to range [0,1].
        if normalize_by_channel:
            ch1mean = tf.constant(91.4953 / 255.0, shape=(224, 224, 1))
            ch2mean = tf.constant(103.8827 / 255.0, shape=(224, 224, 1))
            ch3mean = tf.constant(131.0912 / 255.0, shape=(224, 224, 1))
            channel_norm_tensor = tf.concat([ch1mean, ch2mean, ch3mean], axis=2)
            img -= channel_norm_tensor
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
    # TODO(jpgard): save batch to pdf instead
    # image_batch, label_batch = next(iter(train_ds))
    # from dro.utils.vis import show_batch
    # show_batch(image_batch.numpy(), label_batch.numpy())

    # Disable eager
    # tf.compat.v1.disable_eager_execution()

    # tensorboard_callback = TensorBoard(
    #     log_dir='./training-logs/{}'.format(uid),
    #     batch_size=FLAGS.batch_size,
    #     write_graph=True,
    #     # write_grads=False,
    #     # write_images=False,
    #     # embeddings_freq=0,
    #     # embeddings_layer_names=None,
    #     # embeddings_metadata=None,
    #     # embeddings_data=None,
    #     update_freq='epoch')
    # csv_callback = CSVLogger("./metrics/{}-vggface2-training.log".format(uid))
    #
    # custom_vgg_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    #                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #                          metrics=['accuracy',
    #                                   AUC(name='auc'),
    #                                   TruePositives(name='tp'),
    #                                   FalsePositives(name='fp'),
    #                                   TrueNegatives(name='tn'),
    #                                   FalseNegatives(name='fn')
    #                                   ],
    #                          callbacks=[tensorboard_callback, csv_callback]
    #                          )
    # custom_vgg_model.summary()

    # custom_vgg_model.fit_generator(train_ds, steps_per_epoch=steps_per_epoch,
    #                                epochs=FLAGS.epochs)

    steps_per_epoch = math.floor(1000 / FLAGS.batch_size)
    ds_iterator = train_ds.make_one_shot_iterator()
    batch_x, batch_y = ds_iterator.get_next()
    custom_vgg_model = VGGModel(batch_x, batch_y)

    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )

    with tf.Session(config=config) as sess:
        print("training")
        sess.run(tf.global_variables_initializer())

        tot_accuracy = 0
        iternum = 0
        try:
            while True:
                print("iter %s" % iternum)
                accuracy, loss, _ = sess.run([custom_vgg_model.accuracy,
                                              custom_vgg_model.loss,
                                              custom_vgg_model.optimizer])
                print("iteration %s loss %4f accuracy %4f" % (iternum, loss, accuracy))
                iternum += 1
        except tf.errors.OutOfRangeError:
            pass


if __name__ == "__main__":
    app.run(main)
