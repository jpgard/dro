
from absl import app
from absl import flags
import glob
import numpy as np
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from keras_vggface.vggface import VGGFace
from dro.utils.training_utils import process_path, preprocess_dataset, \
    random_crop_and_resize
from mtcnn import MTCNN

# tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

flags.DEFINE_string("img_dir", None, "directory containing the images")
flags.DEFINE_string("out_dir", None, "directory to write new images to")


# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from PIL import Image
import re

face_detector = MTCNN()  # the detector to use below, using default weights


def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = Image.open(filename)
    pixels = np.array(pixels)
    # create the detector, using default weights
    # detect faces in the image
    results = face_detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return image

def main(argv):
    out_dir = FLAGS.out_dir
    assert not out_dir.endswith("/"), "specify out_dir without a trailing /"
    filepattern = str(FLAGS.img_dir + '*/*.jpg')
    image_ids = glob.glob(filepattern)
    N = len(image_ids)
    for f in image_ids:
        face_im = extract_face(f)
        image_id = re.match('.*(.*/.*/.*\\.jpg)', f).group(1)
        out_fp = out_dir + image_id
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        print("writing to %s" % out_fp)
        face_im.save(out_fp)



if __name__ == "__main__":
    app.run(main)