"""
Classes for working with datasets.
"""
from abc import ABC
from functools import partial

import tensorflow as tf
import pandas as pd
import numpy as np

from dro.keys import SHUFFLE_RANDOM_SEED
from dro.utils.viz import show_batch
from dro.training.training_utils import convert_to_dictionaries

AUTOTUNE = tf.data.experimental.AUTOTUNE
DEFAULT_IMG_OUTPUT_SHAPE = (224, 224)  # Default (height, width) of the images, in pixels.


def preprocess_dataset(
        ds, cache=False, shuffle_buffer_size=1000,
        repeat_forever=False, batch_size: int = None,
        prefetch_buffer_size=AUTOTUNE, shuffle=True,
        epochs: int = None):
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
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    if repeat_forever:
        ds = ds.repeat()
    elif epochs:
        ds = ds.repeat(epochs)
    if batch_size:
        ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=prefetch_buffer_size)
    return ds


def get_label(file_path):
    # convert the path to a list of path components
    label = tf.strings.substr(file_path, -21, 1)
    # The second to last is the class-directory
    return tf.strings.to_number(label, out_type=tf.int32)


def process_path(file_path, output_shape=DEFAULT_IMG_OUTPUT_SHAPE, random_crop=True,
                 labels=True):
    """Load the data from file_path. Returns either an (x,y) tuplesif
    labels=True, or just x if label=False."""
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, output_shape, random_crop=random_crop)
    if not labels:
        return img
    else:
        label = get_label(file_path)
        label = tf.one_hot(label, 2)
        return img, label


def random_crop_and_resize(img):
    """Resize to a square image of 256 x 256, then crop to random 224 x 224 x 3 image."""
    if len(img.shape) < 4:  # add a batch dimension if one does not exist
        img = tf.expand_dims(img, axis=0)
    img = tf.image.resize_images(img, size=(256, 256), preserve_aspect_ratio=True)
    img = tf.image.resize_with_crop_or_pad(img, target_height=256, target_width=256)
    img = tf.squeeze(img, axis=0)
    img = tf.image.random_crop(img, (224, 224, 3))
    return img


def decode_img(img, output_shape, random_crop=True):
    """

    :param img: the encoded image to decode.
    :param output_shape: a tuple containing the desired height and width of the
    output image (should be square).
    :crop: whether to perform random cropping and resizing.
    """
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    if random_crop:
        img = random_crop_and_resize(img)
    if output_shape != DEFAULT_IMG_OUTPUT_SHAPE:
        img = tf.image.resize_images(img, size=output_shape)
    return img


def preprocess_path_label_tuple(x, y, output_shape, random_crop):
    """Create an (image, one_hot_label) tuple from a (path, label) tuple."""
    x = process_path(x, labels=False, output_shape=output_shape, random_crop=random_crop)
    y = tf.one_hot(y, 2)
    return x, y


class ImageDataset(ABC):
    def __init__(self, img_shape=DEFAULT_IMG_OUTPUT_SHAPE):
        self.n_train = None
        self.n_val = None
        self.n_test = None
        self.dataset = None
        self.img_shape = img_shape

    def from_files(self, file_pattern: str, shuffle: bool,
                   random_seed=SHUFFLE_RANDOM_SEED, labels: bool = True,
                   random_crop: bool = True):
        """Create a dataset from the filepattern and preprocess into (x,y) tuples,
        or just x if labels==False."""
        self.dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle,
                                                  seed=random_seed)
        _process_path = partial(process_path, labels=labels, output_shape=self.img_shape,
                                random_crop=random_crop)
        self.dataset = self.dataset.map(_process_path, num_parallel_calls=AUTOTUNE)
        return

    def from_dataframe(self, df: pd.DataFrame, label_name: str, random_crop: bool = True):
        """Create a dataset from a pd.DataFrame."""
        # dset starts as tuples of (filename, label_as_float)
        dset = tf.data.Dataset.from_tensor_slices(
            (df['filename'].values,
             df[label_name].values.astype(np.int))
        )
        _preprocess_path_label_tuple = partial(preprocess_path_label_tuple,
                                               output_shape=self.img_shape,
                                               random_crop=random_crop)
        self.dataset = dset.map(_preprocess_path_label_tuple)
        return

    def from_filename_and_label_generator(self, generator, random_crop: bool = True,
                                          x_dtype=tf.string, y_dtype=tf.int32):
        """
        Create a dataset from a generator which produces (filename, y) tuples.

        :param generator: callable which yields (filename, y) tuples.
        """
        dset = tf.data.Dataset.from_generator(generator,
                                              output_types=(x_dtype, y_dtype))
        _preprocess_path_label_tuple = partial(preprocess_path_label_tuple,
                                               output_shape=self.img_shape,
                                               random_crop=random_crop)
        self.dataset = dset.map(_preprocess_path_label_tuple)
        return

    def validation_split(self, n_val):
        """Create a validation split by extracting elements from the current dataset.

        Elements in validation dataset will not appear in self.dataset after this
        operation, unless the dataset already contained duplicates or was repeated
        using tf.data.Dataset.repeat() prior to calling this function.

        :param n_val: number of validation samples to extract.
        :return: a new instance of ImageDataset with the validation set.
        """
        val_ds = ImageDataset()
        val_ds.dataset = self.dataset.take(n_val)
        return val_ds

    def preprocess(self, **kwargs):
        self.dataset = preprocess_dataset(self.dataset, **kwargs)

    def write_sample_batch(self, fp: str):
        """Write a sample batch of the current dataset to fp."""
        image_batch, label_batch = next(iter(self.dataset))
        show_batch(image_batch=image_batch.numpy(),
                   label_batch=label_batch.numpy(),
                   fp=fp)

    def convert_to_dictionaries(self):
        """Convert the dataset to a set of key, value pairs as a dictionary."""
        self.dataset = self.dataset.map(convert_to_dictionaries)
