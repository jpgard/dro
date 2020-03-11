from itertools import islice

import numpy as np
import tensorflow
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_batch(dataset_iterator, batch_size):
    slice = tuple(islice(dataset_iterator, batch_size))
    batch_x = np.stack([i[0] for i in slice], axis=0)
    batch_y = np.stack([i[1] for i in slice], axis=0)
    return batch_x, batch_y


def prepare_dataset_for_training(ds, cache=True, shuffle_buffer_size=1000,
                         repeat_forever=False, batch_size=None,
                                 prefetch_buffer_size=AUTOTUNE):
    """Shuffle, repeat, batch, and prefetch the dataset."""
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.

    # If the data doesn't fit in memory see the example usage at the
    # bottom of the page here:
    # https: // www.tensorflow.org / tutorials / load_data / images
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
        ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=prefetch_buffer_size)
    return ds


def process_path(file_path):
    label = get_label(file_path)
    label = tf.one_hot(label, 2)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


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


def get_label(file_path):
    # convert the path to a list of path components
    label = tf.strings.substr(file_path, -21, 1)
    # The second to last is the class-directory
    return tf.strings.to_number(label, out_type=tf.int32)