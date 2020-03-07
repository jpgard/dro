"""
Utilities for generating and working with datasets
"""

import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf


def train_test_val_split(img_files, n_train:int, n_val:int, n_test:int, random_seed=2974):
    """Randomly partition img_files into sets of train/val/test images."""
    assert len(img_files) > n_train + n_val + n_test, "insufficient size for given " \
                                                      "splits."
    np.random.seed(random_seed)
    np.random.shuffle(img_files)
    train_files = img_files[:n_train]
    val_files = img_files[n_train:n_train + n_val]
    test_files = img_files[n_train + n_val: n_train + n_val + n_test]
    return train_files, val_files, test_files



def generate_simulated_dataset(n: int = 10 ** 6, p: float = 0.5, shuffle=True):
    """
    Generate a simulated dataset of n examples with proportion p coming from the
    positive class.
    """
    n = int(n)
    n_pos = int(n * p)
    n_neg = int(n) - n_pos
    # The weights and means are set such that the compnents are \sigma apart
    X_pos = np.random.multivariate_normal(mean=(0.5, 0.5), cov=np.eye(2), size=n_pos)
    y_pos = np.ones(n_pos)
    X_neg = np.random.multivariate_normal(mean=(-0.5, 0.5), cov=np.eye(2), size=n_neg)
    y_neg = np.zeros(n_neg)
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    y = y.reshape((len(y), 1))
    y = np.concatenate([np.ones_like(y) - y, y], axis=1)
    if shuffle:
        p = np.random.permutation(n)
        X, y = X[p], y[p]
    return X, y


def load_image_data(img_file_list, img_dir, img_shape):
    img_data = []
    for i, filename in enumerate(img_file_list):
        image = load_img(os.path.join(img_dir, filename),
                         target_size=img_shape[:2])
        image = img_to_array(image) / 255.0
        img_data.append(image)
    img_data = np.array(img_data)
    return img_data


def make_celeba_dataset(img_file_list, batch_size, attributes_df, target_colname, img_dir,
                        img_shape):
    # load the images and the attributes
    img = load_image_data(img_file_list, img_dir, img_shape)
    attr_train = attributes_df.loc[img_file_list, :].values
    labels = attributes_df.loc[img_file_list, target_colname].values
    print("loaded image array with shape = {}".format(img.shape))
    # Build the dataset of images and labels; attributes are not currently used.
    dataset = tf.data.Dataset.from_tensor_slices(
        (img, labels)) \
        .shuffle(1000) \
        .batch(batch_size) \
        .repeat()
    return dataset
