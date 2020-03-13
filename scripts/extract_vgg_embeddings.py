"""
A script to fetch the VGG embeddings for each image in a directory and write them to a
file.

# set the gpu
export GPU_ID="2"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# run the script
python3 scripts/extract_vgg_embeddings.py \
    --img_dir \
    /Users/jpgard/Documents/research/vggface2/train_partitioned_by_label/mouth_open_tiny
"""

from absl import app
from absl import flags
import glob
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

from keras_vggface.vggface import VGGFace
from dro.utils.training_utils import process_path, preprocess_dataset

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

flags.DEFINE_string("img_dir", None, "directory containing the images")
flags.DEFINE_string("out_dir", "./embeddings", "directory to dump the embeddings and "
                                               "similarity to")
flags.DEFINE_integer("n_embed", 10, "number of embeddings to compute for each sample; "
                                    "these are averaged to account for the random "
                                    "cropping.")
flags.DEFINE_bool("similarity", True, "whether or not to write the similarity matrix; "
                                      "this can be huge for large datasets and it may "
                                      "be easier to just store the embeddings and "
                                      "compute similarity later.")

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(argv):
    filepattern = str(FLAGS.img_dir + '/*/*/*.jpg')  # TODO(jpgard): allow
    # specification of this
    embedding_dim = 512
    image_ids = glob.glob(filepattern)
    N = len(image_ids)
    list_ds = tf.data.Dataset.list_files(filepattern, shuffle=True,
                                         seed=2974)
    from functools import partial
    _process_path = partial(process_path, crop=False)
    input_ds = list_ds.map(_process_path, num_parallel_calls=AUTOTUNE)
    train_ds = preprocess_dataset(input_ds,
                                  batch_size=1,
                                  shuffle=False,
                                  prefetch_buffer_size=AUTOTUNE)
    from dro.utils.training_utils import random_crop_and_resize
    # repeat each element, discaring the labels
    train_ds = train_ds.flat_map(lambda x, y: tf.data.Dataset.from_tensors(x).repeat(
        FLAGS.n_embed))
    train_ds = train_ds.map(lambda x: tf.squeeze(x, axis=0))  # remove this later
    # TODO(jpgard): see if images are being repeated
    import matplotlib.pyplot as plt
    x = 0
    for image in train_ds.take(FLAGS.n_embed):
        ax = plt.subplot(FLAGS.n_embed, 1, x + 1)
        ax.imshow(image.numpy())
        plt.axis('off')
        x += 1
    plt.show()
    import ipdb;
    ipdb.set_trace()
    # apply different random cropping to each iterate of x, and drop the labels.

    # TODO(jpgard): see if random crops are the same or different. Want different.

    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    embeddings = vgg_model.predict_generator(train_ds)

    # embeddings has shape [N * n_embeddings, 512]; take average over each sample due
    # to the random cropping of the inputs.
    # embeddings = tf.reshape(embeddings, (N, embedding_dim, FLAGS.n_embed))
    # embeddings = tf.reduce_mean(embeddings, axis=-1)
    # embeddings = embeddings.numpy()
    embedding_df = pd.DataFrame(embeddings, index=image_ids)
    embedding_df.to_csv(os.path.join(FLAGS.out_dir, "embedding.csv"), index=True)
    if FLAGS.similarity:
        similarities = cosine_similarity(embeddings)
        similarity_df = pd.DataFrame(similarities, index=image_ids, columns=image_ids)
        similarity_df.to_csv(os.path.join(FLAGS.out_dir, "similarity.csv"), index=True)


if __name__ == "__main__":
    app.run(main)
