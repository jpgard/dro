import glob
from collections import defaultdict
from itertools import islice
import json
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from dro.keys import IMAGE_INPUT_NAME, LABEL_INPUT_NAME, TEST_MODE, TRAIN_MODE

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_batch(dataset_iterator, batch_size):
    slice = tuple(islice(dataset_iterator, batch_size))
    batch_x = np.stack([i[0] for i in slice], axis=0)
    batch_y = np.stack([i[1] for i in slice], axis=0)
    return batch_x, batch_y


def get_adversarial_mode(is_adversarial: bool):
    """Get a string indicator for whether the mode is adversarial or not."""
    if is_adversarial:
        return "adversarial"
    else:
        return "base"


def get_training_mode(is_testing: bool):
    """Get a string indicator for whether the mode is training or not."""
    if is_testing:
        return TEST_MODE
    else:
        return TRAIN_MODE


def make_csv_name(uid, mode):
    return "./metrics/{}-vggface2-{}.log".format(uid, mode)


def write_test_metrics_to_csv(metrics, flags, is_adversarial):
    """Write a dict of {metric_name: metric_value} pairs to CSV."""
    uid = make_model_uid_from_flags(flags, is_adversarial=is_adversarial)
    csv_fp = make_csv_name(uid, TEST_MODE)
    print("[INFO] writing test metrics to {}".format(csv_fp))
    pd.DataFrame.from_dict(metrics, orient='index').T.to_csv(csv_fp, index=False)


def make_logdir(flags, uid):
    return os.path.join(flags.ckpt_dir, uid)


def make_ckpt_filepath():
    pass


def make_ckpt_filepath(ext, uid, logdir):
    """Make the checkpoint filepath for a model."""
    assert ext.startswith("."), "provide a valid extension"
    return os.path.join(logdir, uid + ext)


def make_ckpt_filepath_from_flags(flags, is_adversarial: bool, ext: str = ".h5"):
    """A utility function to make the checkpoint filepath from a set of flags."""
    uid = make_model_uid_from_flags(flags, is_adversarial=is_adversarial)
    logdir = make_logdir(flags, uid)
    ckpt_filepath = make_ckpt_filepath(ext=ext, uid=uid, logdir=logdir)
    return ckpt_filepath


def make_model_uid(label_name: str, model_type: str, batch_size: int, epochs: int,
                   learning_rate: float, dropout_rate: float, attack: str,
                   attack_params: str, adv_multiplier: float, experiment_uid: str,
                   use_dbs: bool, is_adversarial: bool):
    """Create a unique identifier for the model specified by the attributes."""
    model_uid = """{label}-{model}-bs{bs}e{epochs}lr{lr}dropout{dr}""".format(
        label=label_name,
        model=model_type,
        bs=batch_size,
        epochs=epochs,
        lr=learning_rate,
        dr=dropout_rate
    )
    if is_adversarial:
        model_uid += "-" + attack
        if attack_params is not None:
            attack_params = json.loads(attack_params)
            for k, v in sorted(attack_params.items()):
                model_uid += "-{}{}".format(k[0], str(v))
            model_uid += "-m{}".format(adv_multiplier)
    if use_dbs:
        model_uid += "dbs"
    if experiment_uid:
        model_uid += experiment_uid
    return model_uid


def make_model_uid_from_flags(flags, is_adversarial=False):
    """Utility function to create the unique model identifier from a set of flags."""

    if is_adversarial:
        attack = flags.attack
        attack_params = flags.attack_params
        adv_multiplier = flags.adv_multiplier
    else:
        attack = None
        attack_params = None
        adv_multiplier = None
    uid = make_model_uid(label_name=flags.label_name, model_type=flags.model_type,
                         batch_size=flags.batch_size, epochs=flags.epochs,
                         learning_rate=flags.learning_rate,
                         dropout_rate=flags.dropout_rate, attack=attack,
                         attack_params=attack_params,
                         adv_multiplier=adv_multiplier,
                         experiment_uid=flags.experiment_uid, use_dbs=flags.use_dbs,
                         is_adversarial=is_adversarial)
    return uid


def convert_to_dictionaries(image, label):
    """Convert a set of x,y tuples to a dict for use in adversarial training."""
    return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}


def pred_to_binary(x, thresh=0.):
    """Convert lfw predictions to binary (0.,1.) labels by thresholding based on
    thresh."""
    return int(x > thresh)


def add_keys_to_dict(input_dict, **kwargs):
    """Modify input_dict in place by adding the key-value pairs."""
    for k, v in kwargs.items():
        input_dict[k] = v
    return input_dict


def get_n_from_file_pattern(file_pattern):
    return len(glob.glob(file_pattern))


def compute_n_train_n_val(n_train_val, val_frac):
    n_val = int(n_train_val * val_frac)
    n_train = n_train_val - n_val
    return n_train, n_val


def steps_per_epoch(n, batch_size, debug=False):
    if debug:
        print("[INFO] running in debug mode")
        return 1
    else:
        return n // batch_size


def load_model_weights_from_flags(model, flags, is_adversarial: bool):
    """Load weights for a pretrained model, either from a manually-specified checkpoint 
    or from the default path."""
    if is_adversarial:
        model_ckpt = flags.adv_model_ckpt
    else:
        model_ckpt = flags.base_model_ckpt

    if model_ckpt:  # load from the manually-specified checkpoint
        print("[INFO] loading from specified checkpoint {}".format(
            model_ckpt
        ))
        model.load_weights(filepath=model_ckpt)
    else:  # Load from the default checkpoint path
        filepath = make_ckpt_filepath_from_flags(flags, is_adversarial=is_adversarial)
        print("[INFO] loading weights from{}".format(filepath))
        model.load_weights(filepath=filepath)
    return


def metrics_to_dict(metrics):
    """Convert metrics to a dictionary of key:float pairs."""
    results = defaultdict(dict)
    for model_name, metrics_list in metrics.items():
        for metric in metrics_list:
            res = metric.result().numpy()
            results[model_name][metric.name] = res
    return results