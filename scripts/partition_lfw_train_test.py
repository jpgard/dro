"""
Divide the LFW dataset into a train-test split based on their suggested splits.
"""

from absl import flags, app
import glob
import shutil
import os

import pandas as pd

from dro.utils.lfw import make_lfw_file_pattern, extract_imagenum_from_filename, \
    extract_person_from_filename

flags.DEFINE_string("people_test", None, "Path to the LFW test people file.")
flags.DEFINE_string("people_train", None, "Path to the LFW train people file.")
flags.DEFINE_string("input_dir", None, "Path to directory containing the LFW images.")
flags.DEFINE_string("output_dir", None, "Path to directory to deposit the split results.")
FLAGS = flags.FLAGS


def main(argv):
    train_people = pd.read_csv(FLAGS.people_train,
                               skiprows=1, delimiter="\t").iloc[:, 0].tolist()
    train_people = set(train_people)
    test_people = pd.read_csv(FLAGS.people_test,
                              skiprows=1, delimiter="\t").iloc[:, 0].tolist()
    test_people = set(test_people)
    file_pattern = make_lfw_file_pattern(FLAGS.input_dir)
    images = glob.glob(file_pattern)
    n_train = 0
    n_test = 0
    n_warn = 0
    for image_fp in images:
        try:
            person = extract_person_from_filename(image_fp)
            if person in train_people:
                print("{} in train for fp {}".format(person, image_fp))
                n_train += 1
                dest = "train"
                # TODO:
            elif person in test_people:
                print("{} in test for fp {}".format(person, image_fp))
                n_test += 1
                dest = "test"
                # TODO
            else:
                print("[WARNING] person {} in fp {} is neither in train nor test; "
                      "adding to train set".format(
                    person, image_fp))
                n_warn += 1
                dest = "train"
            dest_dir = os.path.join(FLAGS.output_dir, dest, person)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(image_fp, dest_dir)
        except Exception as e:
            print("[WARNING] Exception {} for fp {}; skipping".format(e, image_fp))
            import ipdb;ipdb.set_trace()
    print("processed {} training images, {} testing images, {} warnings".format(
        n_train, n_test, n_warn
    ))
    return


if __name__ == "__main__":
    app.run(main)
