"""
Script to fine-tune pretrained VGGFace2 model.

usage:
python scripts/train_vggface2.py \
    --img_dir /Users/jpgard/Documents/research/vggface2/test \
    --anno_fp /Users/jpgard/Documents/research/vggface2/anno/10-sunglasses.txt
"""

import pandas as pd
import glob
from absl import app
from absl import flags
import numpy as np
from dro.datasets import train_test_val_split
import re



FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("epochs", 5, "the number of training epochs")
flags.DEFINE_string("img_dir", None, "directory containing the aligned celeba images")
flags.DEFINE_string("anno_fp", None, "path to the target annotation file.")


def load_and_preprocess_images(img_file_list, img_dir, img_shape):
    img_data = []
    for i, filename in enumerate(img_file_list):

        image = load_img(os.path.join(img_dir, filename),
                         target_size=img_shape[:2])
        image = img_to_array(image) / 255.0
        img_data.append(image)
    img_data = np.array(img_data)
    return img_data

def make_vggface2_dataset(img_file_list, batch_size, attributes_df,
                          img_dir, img_shape):

    pass

def main(argv):
    img_files = glob.glob(FLAGS.img_dir + "/*/*.jpg")
    # glob will fetch full path; we just extract the subpath from FLAGS.img_dir
    img_files = [re.match(".*/(.*/.*.jpg)", f).group(1) for f in img_files]
    n_train = FLAGS.batch_size * 1
    n_val = FLAGS.batch_size * 1
    n_test = FLAGS.batch_size * 1
    img_files_train, img_files_val, img_files_test = train_test_val_split(
        img_files, n_train, n_val, n_test)
    import ipdb;ipdb.set_trace()
    attributes_df = pd.read_csv(FLAGS.anno_fp, delimiter="\t", index_col=0)
    make_vggface2_dataset(img_files_train, FLAGS.batch_size, attributes_df,
                          )




if __name__ == "__main__":
    app.run(main)