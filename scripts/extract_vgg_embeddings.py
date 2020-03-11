"""
A script to fetch the VGG embeddings for each image in a directory and write them to a
file.
"""

from absl import app
from absl import flags
import glob
import tensorflow as tf

from keras_vggface.vggface import VGGFace
from dro.utils.training_utils import process_path, preprocess_dataset

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

flags.DEFINE_string("img_dir", None, "directory containing the images")

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(argv):
    n_embeddings = 10  # number of times to randomly crop image and fetch embedding
    filepattern = str(FLAGS.img_dir + '/*/*/*.jpg')
    N = len(glob.glob(filepattern))
    list_ds = tf.data.Dataset.list_files(filepattern, shuffle=True,
                                         seed=2974)
    input_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = preprocess_dataset(input_ds, epochs=n_embeddings,
                                  batch_size=1,
                                  shuffle=False,
                                  prefetch_buffer_size=AUTOTUNE)
    # drop the labels from train_ds
    train_ds = train_ds.map(lambda x, y: x)
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    embeddings = vgg_model.predict_generator(train_ds)
    # embddings has shape [N * n_embeddings, 2048]
    import ipdb;ipdb.set_trace()
    # TODO(jpgard): average the embeddings for each image, then write them to a file.


if __name__ == "__main__":
    app.run(main)
