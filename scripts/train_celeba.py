import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pandas as pd

import tensorflow as tf

from dro.utils.viz import plot_faces
from dro.training.models import facenet_model

tf.compat.v1.enable_eager_execution()

img_dir = "/Users/jpgard/Documents/research/celeba/img/img_align_celeba"
attributes_fp = "/Users/jpgard/Documents/research/celeba/anno/list_attr_celeba.txt"

# this is the default size for the cropped images, no reduction
img_shape = (218, 178, 3)
target_colname = "Smiling"
batch_size = 256
epochs = 5
n_train = batch_size * 10
n_test = 100

# Fetch jpg files for training and testing
img_files = np.sort(os.listdir(img_dir))
img_files_train = img_files[:n_train]
img_files_test = img_files[n_train:n_train + n_test]

attributes_df = pd.read_csv(attributes_fp, delim_whitespace=True, skiprows=0,
                            header=1).sort_index().replace(-1, 0)
attributes_df[target_colname].copy()
assert attributes_df.shape[0] == len(img_files), \
    "possible mismatch between training images and attributes."


def load_image_data(img_file_list):
    img_data = []
    for i, filename in enumerate(img_file_list):
        image = load_img(os.path.join(img_dir, filename),
                         target_size=img_shape[:2])
        image = img_to_array(image) / 255.0
        img_data.append(image)
    img_data = np.array(img_data)
    return img_data


# load the images and the attributes
img_train = load_image_data(img_files_train)
attr_train = attributes_df.loc[img_files_train, :].values
labels_train = attributes_df.loc[img_files_train, target_colname].values
print("img_ary.shape = {}".format(img_train.shape))

img_test = load_image_data(img_files_test)
print("img_test.shape = {}".format(img_test.shape))
attr_test = attributes_df.loc[img_files_test, :].values
labels_test = attributes_df.loc[img_files_test, target_colname].values
train_dataset = tf.data.Dataset.from_tensor_slices(
    (img_train, labels_train)) \
    .shuffle(1000) \
    .batch(batch_size) \
    .repeat()

# show a visualization of the first few faces
# plot_faces(img_train)


model = facenet_model()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.build(input_shape=(None, 218, 178, 3))
model.summary()
import ipdb;ipdb.set_trace()
model.fit(train_dataset, epochs=epochs, steps_per_epoch=n_train // batch_size)
