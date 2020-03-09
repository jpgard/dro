"""
Script to fine-tune pretrained VGGFace2 model.

usage:

# set the gpu
export GPU_ID="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# run the script
python scripts/train_vggface2.py \
    --img_dir /Users/jpgard/Documents/research/vggface2/train_partitioned_by_label/mouth_open
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
import time
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Activation, Dropout
# from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, \
#     FalsePositives, FalseNegatives
# from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from keras_vggface.vggface import VGGFace

# tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("epochs", 5, "the number of training epochs")
flags.DEFINE_string("img_dir", None, "directory containing the aligned celeba images")
flags.DEFINE_float("learning_rate", 0.01, "learning rate to use")
flags.DEFINE_float("dropout_rate", 0.8, "dropout rate to use in fully-connected layers")

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CLASS_NAMES = np.array(["0", "1"])
LABELS_DTYPE = tf.float32



def main(argv):
    list_ds = tf.data.Dataset.list_files(str(FLAGS.img_dir + '/*/*/*.jpg'), shuffle=True,
                                         seed=2974)
    # TODO(jpgard): read this from dataset or directory.
    n_train = 1000

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
        # extract the label from the path
        label = tf.strings.substr(file_path, -21, 1)
        return tf.strings.to_number(label, out_type=LABELS_DTYPE)

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

        # Repeat the dataset only FLAGS.epochs times. Then, the iterator will
        #  naturally run out at the termination of the desired number of batches and we
        #  don't need to precisely count individual batches.
        ds = ds.repeat(FLAGS.epochs)
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


    # TODO(jpgard): set this to the correct value given the sample size and number of
    #  epochs

    steps_per_epoch = math.floor(n_train / FLAGS.batch_size)
    # TODO(jpgard): instead, make an initializable iterator and re-initizlize at every
    #  epoch, as shown in answer below.
    #  https://stackoverflow.com/questions/47067401/how-to-iterate-a-dataset-several
    #  -times-using-tensorflows-dataset-api
    ds_iterator = train_ds.make_one_shot_iterator()
    next_element = ds_iterator.get_next()

    ########################################################################

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
    model = Model(vgg_model.input, out)

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss(model, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)
        y_ = tf.squeeze(y_)  # Convert shape from (batch_size, 1) to (batch_size,)
        return loss_object(y_true=y, y_pred=y_)

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    # loss_value, grads = grad(model, batch_x, batch_y)
    # optimizer.apply_gradients(zip(grads, model.trainable_variables))
    ########################################################################


    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        # log_device_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )

    with tf.Session(config=config) as sess:
        print("training")
        # sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.epochs):
            epoch_start = time.time()
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
            for step in range(steps_per_epoch):
                print("epoch %s step %s" % (epoch, step))
                # Training loop - using batches of 32
                x, y = sess.run(next_element)
                # Optimize the model
                loss_value, grads = grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                epoch_accuracy.update_state(y_true=y, y_pred=model(x, training=True))
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch, epoch_loss_avg.result().eval(), epoch_accuracy.result().eval()))
            epoch_train_time = int(time.time() - epoch_start)
            print("[INFO] epoch %4s completed in %f seconds" % (epoch, epoch_train_time))

            # End epoch
            # train_loss_results.append(epoch_loss_avg.result())
            # train_accuracy_results.append(epoch_accuracy.result())
        #     epoch_total_accuracy = 0.
        #     epoch_total_loss = 0.
        #     for step in range(steps_per_epoch):
        #         try:
        #             # while True:
        #             # TODO(jpgard): figure out why the loss is not going down.
        #             batch_acc, batch_loss, _ = sess.run([custom_vgg_model.accuracy,
        #                                           custom_vgg_model.loss,
        #                                           custom_vgg_model.optimizer])
        #             epoch_total_accuracy += batch_acc
        #             epoch_total_loss += batch_loss
        #         except tf.errors.OutOfRangeError:
        #             print("[WARNING] reached end of dataset.")
        #             break

        #     print("epoch %s loss %4f accuracy %4f" %
        #           (epoch, epoch_total_loss/float(steps_per_epoch),
        #            epoch_total_accuracy/float(steps_per_epoch)))
        #     print("=" * 80)


if __name__ == "__main__":
    app.run(main)
