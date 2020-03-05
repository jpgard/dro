import os
import numpy as np
import pandas as pd

import tensorflow as tf

from dro.utils.viz import plot_faces
from dro.datasets import make_dataset
from dro.training.models import facenet_model

tf.compat.v1.enable_eager_execution()

img_dir = "/Users/jpgard/Documents/research/celeba/img/img_align_celeba"
attributes_fp = "/Users/jpgard/Documents/research/celeba/anno/list_attr_celeba.txt"

# this is the default size for the cropped images, no reduction
img_shape = (218, 178, 3)
target_colname = "Smiling"
batch_size = 256
epochs = 5
n_train = batch_size * 50
n_val = batch_size * 10
n_test = batch_size * 5

# Fetch jpg files for training and testing
img_files = np.sort(os.listdir(img_dir))
img_files_train = img_files[:n_train]
img_files_val = img_files[n_train:n_train + n_val]
img_files_test = img_files[n_train + n_val: n_train + n_val + n_test]

attributes_df = pd.read_csv(attributes_fp, delim_whitespace=True, skiprows=0,
                            header=1).sort_index().replace(-1, 0)
attributes_df[target_colname].copy()
assert attributes_df.shape[0] == len(img_files), \
    "possible mismatch between training images and attributes."


train_dataset = make_dataset(img_files_train, batch_size, attributes_df, target_colname)
val_dataset = make_dataset(img_files_val, batch_size, attributes_df, target_colname)
test_dataset = make_dataset(img_files_test, batch_size, attributes_df, target_colname)

# show a visualization of the first few faces
# plot_faces(img_train)

model = facenet_model()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.build(input_shape=(None, 218, 178, 3))
model.summary()
model.fit(train_dataset, epochs=epochs, steps_per_epoch=n_train // batch_size,
          validation_data=val_dataset, validation_steps=n_val // batch_size)
