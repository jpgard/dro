"""
A script to fetch the VGG embeddings for each image in a directory and write them to a
file.

# set the gpu
export GPU_ID="2"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# run the script
python3 scripts/extract_vgg_embeddings.py \
    --img_dir  /projects/grail/jpgard/vggface2/annotated_cropped/train
"""

from absl import app
from absl import flags
from functools import partial
import glob
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

from keras_vggface.vggface import VGGFace
from dro.datasets import preprocess_dataset, process_path
from dro.utils.flags import define_embeddings_flags

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

# Define the flags.
define_embeddings_flags()

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(argv):
    filepattern = str(FLAGS.img_dir + '/*/*.jpg')
    image_ids = glob.glob(filepattern)
    assert len(image_ids) > 0, "no images found"
    image_ids = [os.path.abspath(p) for p in image_ids]
    N = len(image_ids)
    list_ds = tf.data.Dataset.list_files(filepattern, shuffle=False)
    _process_path = partial(process_path, crop=False, labels=False)
    input_ds = list_ds.map(_process_path, num_parallel_calls=AUTOTUNE)
    train_ds = preprocess_dataset(input_ds,
                                  batch_size=FLAGS.batch_size,
                                  shuffle=False,
                                  prefetch_buffer_size=AUTOTUNE)
    # Run the inference
    image_batch = next(iter(train_ds))
    from dro.utils.viz import show_batch
    show_batch(image_batch.numpy(), fp="./debug/example_batch.png")

    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    embeddings = vgg_model.predict_generator(train_ds)

    embedding_df = pd.DataFrame(embeddings, index=image_ids)
    embedding_df.to_csv(os.path.join(FLAGS.out_dir, "embedding.csv"), index=True)
    if FLAGS.similarity:
        similarities = cosine_similarity(embeddings)
        similarity_df = pd.DataFrame(similarities, index=image_ids, columns=image_ids)
        similarity_df.to_csv(os.path.join(FLAGS.out_dir, "similarity.csv"), index=True)


if __name__ == "__main__":
    app.run(main)
