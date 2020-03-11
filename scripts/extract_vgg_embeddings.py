"""
A script to fetch the VGG embeddings for each image in a directory and write them to a
file.
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

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(argv):
    filepattern = str(FLAGS.img_dir + '/*/*/*.jpg')
    embedding_dim = 512
    image_ids = glob.glob(filepattern)
    N = len(image_ids)
    list_ds = tf.data.Dataset.list_files(filepattern, shuffle=True,
                                         seed=2974)
    input_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = preprocess_dataset(input_ds, epochs=FLAGS.n_embeddings,
                                  batch_size=1,
                                  shuffle=False,
                                  prefetch_buffer_size=AUTOTUNE)
    # drop the labels from train_ds
    train_ds = train_ds.map(lambda x, y: x)
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    embeddings = vgg_model.predict_generator(train_ds)

    # embeddings has shape [N * n_embeddings, 2048]; take average over each sample due
    # to the random cropping of the inputs.
    embeddings = tf.reshape(embeddings, (N, embedding_dim, FLAGS.n_embeddings))
    embeddings = tf.reduce_mean(embeddings, axis=-1)
    embeddings = embeddings.numpy()
    similarities = cosine_similarity(embeddings)
    similarity_df = pd.DataFrame(similarities, index=image_ids, columns=image_ids)
    embedding_df = pd.DataFrame(embeddings, index=image_ids)
    similarity_df.to_csv(os.path.join(FLAGS.out_dir, "similarity.csv"), index=True)
    embedding_df.to_csv(os.path.join(FLAGS.out_dir, "embedding.csv"), index=True)


if __name__ == "__main__":
    app.run(main)
