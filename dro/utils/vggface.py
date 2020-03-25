import os
import re

import pandas as pd
from dro.keys import FILENAME_COLNAME


def image_uid_from_fp(f):
    image_id = re.match('.*(.*/.*/.*\\.jpg)', f)
    if image_id:
        return image_id.group(1)
    else:  # not detected; invalid path
        print("[WARNING] invalid vgg identifier: {}".format(f))
        return None


def make_annotations_df(anno_dir):
    """Build a single DataFrame with all annotations.

    :returns: a pd.DataFrame with index FILENAME_COLNAME and one column per annotation
    file representing the labels associated with that file.
    """
    annotations = []
    for f in os.listdir(anno_dir):
        if (not f.startswith(".")) and f.endswith(".txt"):
            label = re.match("\d+-(.*)\.txt", f).group(1)
            attributes_df = pd.read_csv(
                os.path.join(anno_dir, f),
                delimiter="\t",
                names=[FILENAME_COLNAME, label]) \
                .set_index(FILENAME_COLNAME)
            attributes_df.columns = [label, ]
            annotations.append(attributes_df)
    return pd.concat(annotations, axis=1)


def get_key_from_fp(fp):
    """Get the key that uniquely identifies the image; user_id/image_id.jpg"""
    try:
        return re.match(".*/(.*/.*.jpg)", fp).group(1)
    except:
        return None


def get_person_id_from_fp(fp):
    """Get the key that uniquely identifies the user in the image."""
    try:
        return re.match(".*/(.*)/.*.jpg", fp).group(1)
    except:
        return None
