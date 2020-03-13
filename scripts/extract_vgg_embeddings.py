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
import matplotlib.pyplot as plt

from keras_vggface.vggface import VGGFace
from dro.utils.training_utils import process_path, preprocess_dataset, \
    random_crop_and_resize

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

flags.DEFINE_string("img_dir", None, "directory containing the images")
flags.DEFINE_string("out_dir", "./embeddings", "directory to dump the embeddings and "
                                               "similarity to")
flags.DEFINE_bool("similarity", True, "whether or not to write the similarity matrix; "
                                      "this can be huge for large datasets and it may "
                                      "be easier to just store the embeddings and "
                                      "compute similarity later.")
flags.DEFINE_integer("batch_size", 16, "batch size to use for inference")
# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(argv):
    filepattern = str(FLAGS.img_dir + '/*/*.jpg')
    image_ids = glob.glob(filepattern)
    assert len(image_ids) > 0, "no images found"
    image_ids = [os.path.abspath(p) for p in image_ids]
    import ipdb;ipdb.set_trace()
    N = len(image_ids)
    list_ds = tf.data.Dataset.list_files(filepattern, shuffle=False)
    from functools import partial
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
