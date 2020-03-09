"""
Script to fine-tune pretrained VGGFace2 model.

usage:

# set the gpu
export GPU_ID="3"
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
from dro.sinha.attacks import WassersteinRobustMethod
import keras

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

# the vggface2/training parameters
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("epochs", 5, "the number of training epochs")
flags.DEFINE_string("img_dir", None, "directory containing the aligned celeba images")
flags.DEFINE_float("learning_rate", 0.001, "learning rate to use")
flags.DEFINE_float("dropout_rate", 0.8, "dropout rate to use in fully-connected layers")
flags.DEFINE_bool("adversarial", False, "whether to use adversarial perturbation.")

# the wrm parameters
flags.DEFINE_multi_float('wrm_eps', 1.3,
                         'epsilon value to use for Wasserstein robust method; note that '
                         'original default value is 1.3.')
flags.DEFINE_integer('wrm_ord', 2, 'order of norm to use in Wasserstein robust method')
flags.DEFINE_integer('wrm_steps', 15,
                     'number of steps to use in Wasserstein robus method')


# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



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
        return model_uid

    uid = make_model_uid()

    # for f in list_ds.take(3):
    #     print(f.numpy())

    def get_label(file_path):
        # convert the path to a list of path components
        label = tf.strings.substr(file_path, -21, 1)
        # The second to last is the class-directory
        return tf.strings.to_number(label, out_type=tf.int32)

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
        label = tf.one_hot(label, 2)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # for image, label in labeled_ds.take(10):
    #     print("Image shape: ", image.numpy().shape)
    #     print("Label: ", label.numpy())

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000,
                             repeat_forever=False, batch_size=None):
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
        if repeat_forever:
            ds = ds.repeat()
        if batch_size:
            ds = ds.batch(FLAGS.batch_size)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    # if the data doesn't fit in memory see the example usage at the
    # bottom of the page here:
    # https: // www.tensorflow.org / tutorials / load_data / images

    # TODO(jpgard): save batch to pdf instead
    # image_batch, label_batch = next(iter(train_ds))
    # from dro.utils.vis import show_batch
    # show_batch(image_batch.numpy(), label_batch.numpy())

    # Disable eager
    # tf.compat.v1.disable_eager_execution()
    from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, \
        FalsePositives, FalseNegatives
    from tensorflow.keras.callbacks import TensorBoard, CSVLogger
    from dro.training.models import vggface2_model
    custom_vgg_model = vggface2_model(dropout_rate=FLAGS.dropout_rate)
    steps_per_epoch = math.floor(1000 / FLAGS.batch_size)

    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        # log_device_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )

    if FLAGS.adversarial:
        import tensorflow_datasets as tfds
        train_ds = prepare_for_training(labeled_ds, repeat_forever=True, batch_size=None)
        # train_iter is an iterator which returns X,Y pairs of numpy arrays where
        # X has shape (224, 224, 3) and Y has shape (2,).
        train_iter = tfds.as_numpy(train_ds)
        from dro.sinha.utils_tf import model_train
        from dro.utils.experiment_utils import model_eval_fn
        from functools import partial
        # The adversarial perturbation block
        x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(None, 2))
        wrm_params = {'eps': FLAGS.wrm_eps, 'ord': FLAGS.wrm_ord, 'y': y,
                      'steps': FLAGS.wrm_steps}
        # Create TF session and set as Keras backend session
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
        wrm = WassersteinRobustMethod(custom_vgg_model, sess=sess)
        predictions = custom_vgg_model(x)
        predictions_adv_wrm = custom_vgg_model(wrm.generate(x, **wrm_params))
        # TODO(jpgard): create a separate test dataset.
        eval_params = {'batch_size': FLAGS.batch_size}
        eval_fn = partial(model_eval_fn, sess, x, y, predictions, predictions_adv_wrm,
                          X_test=None, Y_test=None, eval_params=eval_params,
                          dataset_iterator=train_iter)
        model_train_fn = partial(model_train,
                                 sess, x, y, predictions_adv_wrm, X_train=None,
                                 Y_train=None,
                                 evaluate=eval_fn,
                                 args={"nb_epochs": FLAGS.epochs,
                                       "learning_rate": FLAGS.learning_rate,
                                       "batch_size": FLAGS.batch_size},
                                 save=False,
                                 dataset_iterator=train_iter,
                                 nb_batches=steps_per_epoch)
        metrics = model_train_fn()
        print(metrics)
        import ipdb;ipdb.set_trace()



    else:
        train_ds = prepare_for_training(labeled_ds, repeat_forever=True,
                                        batch_size=FLAGS.batch_size)
        tensorboard_callback = TensorBoard(
            log_dir='./training-logs/{}'.format(uid),
            batch_size=FLAGS.batch_size,
            write_graph=True,
            write_grads=True,
            update_freq='epoch')
        csv_callback = CSVLogger("./metrics/{}-vggface2-training.log".format(uid))
        custom_vgg_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                                 loss=tf.keras.losses.CategoricalCrossentropy(
                                     from_logits=True),
                                 metrics=['accuracy',
                                          AUC(name='auc'),
                                          TruePositives(name='tp'),
                                          FalsePositives(name='fp'),
                                          TrueNegatives(name='tn'),
                                          FalseNegatives(name='fn')
                                          ]
                                 )
        custom_vgg_model.summary()
        custom_vgg_model.fit_generator(train_ds, steps_per_epoch=steps_per_epoch,
                                       epochs=FLAGS.epochs, callbacks=[tensorboard_callback, csv_callback])



if __name__ == "__main__":
    app.run(main)
