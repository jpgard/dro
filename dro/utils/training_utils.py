from collections import defaultdict
import glob
from itertools import islice
import json
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import neural_structured_learning as nsl
from tensorflow import keras

from tensorflow_core.python.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, \
    FalsePositives, FalseNegatives

from dro.keys import IMAGE_INPUT_NAME, LABEL_INPUT_NAME, ACC, CE
from dro import keys
from dro.training.models import vggface2_model, facenet_model

AUTOTUNE = tf.data.experimental.AUTOTUNE
TEST_MODE = "test"
TRAIN_MODE = "train"


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


def cross_entropy_sigma(y_true, y_pred):
    """Custom metric to compute the standard deviation of the loss."""
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    element_wise_loss = loss_object(y_true=y_true, y_pred=y_pred)
    loss_std = tf.math.reduce_std(element_wise_loss)
    return loss_std


def cross_entropy_max(y_true, y_pred):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    element_wise_loss = loss_object(y_true=y_true, y_pred=y_pred)
    loss_max = tf.math.reduce_max(element_wise_loss)
    return loss_max


def get_train_metrics():
    """Fetch an OrderedDict of train metrics."""
    train_metrics_dict = {ACC: ACC,
                          'auc': AUC(name='auc'),
                          'tp': TruePositives(name='tp'),
                          'fp': FalsePositives(name='fp'),
                          'tn': TrueNegatives(name='tn'),
                          'fn': FalseNegatives(name='fn'),
                          CE: tf.keras.losses.CategoricalCrossentropy(),
                          'sigma_ce': cross_entropy_sigma,
                          'max_ce': cross_entropy_max
                          }
    return train_metrics_dict


def write_test_metrics_to_csv(metrics, flags, is_adversarial):
    """Write a dict of {metric_name: metric_value} pairs to CSV."""
    uid = make_model_uid(flags, is_adversarial=is_adversarial)
    csv_fp = make_csv_name(uid, TEST_MODE)
    print("[INFO] writing test metrics to {}".format(csv_fp))
    pd.DataFrame.from_dict(metrics, orient='index').T.to_csv(csv_fp, index=False)


def make_csv_callback(flags, is_adversarial: bool):
    callback_uid = make_model_uid(flags, is_adversarial=is_adversarial)
    csv_fp = make_csv_name(callback_uid, mode=TRAIN_MODE)
    return CSVLogger(csv_fp)


def make_logdir(flags, uid):
    return os.path.join(flags.ckpt_dir, uid)


def make_ckpt_filepath(flags, is_adversarial: bool, ext: str = ".h5"):
    assert ext.startswith("."), "provide a valid extension"
    uid = make_model_uid(flags, is_adversarial=is_adversarial)
    logdir = make_logdir(flags, uid)
    return os.path.join(logdir, uid + ext)


def make_callbacks(flags, is_adversarial: bool):
    """Create the callbacks for training, including properly naming files."""
    callback_uid = make_model_uid(flags, is_adversarial=is_adversarial)
    logdir = make_logdir(flags, callback_uid)
    tensorboard_callback = TensorBoard(
        log_dir=logdir,
        batch_size=flags.batch_size,
        write_graph=True,
        write_grads=True,
        update_freq='epoch')
    csv_callback = make_csv_callback(flags, is_adversarial)
    ckpt_fp = make_ckpt_filepath(flags, is_adversarial=is_adversarial)
    ckpt_callback = ModelCheckpoint(ckpt_fp,
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=False,
                                    save_freq='epoch',
                                    verbose=1,
                                    mode='auto')
    return [tensorboard_callback, csv_callback, ckpt_callback]


def make_model_uid(flags, is_adversarial=False):
    """Create a unique identifier for the model."""
    model_uid = """{label}-{model}-bs{bs}e{epochs}lr{lr}dropout{dr}""".format(
        label=flags.label_name,
        model=flags.model_type,
        bs=flags.batch_size,
        epochs=flags.epochs,
        lr=flags.learning_rate,
        dr=flags.dropout_rate
    )
    if is_adversarial:
        model_uid += "-" + flags.attack
        if flags.attack_params is not None:
            attack_params = json.loads(flags.attack_params)
            for k, v in sorted(attack_params.items()):
                model_uid += "-{}{}".format(k[0], str(v))
    if flags.use_dbs:
        model_uid += "dbs"
    if flags.experiment_uid:
        model_uid += flags.experiment_uid
    return model_uid


def metrics_to_dict(metrics):
    """Convert metrics to a dictionary of key:float pairs."""
    results = defaultdict(dict)
    for model_name, metrics_list in metrics.items():
        for metric in metrics_list:
            res = metric.result().numpy()
            results[model_name][metric.name] = res
    return results


def perturb_and_evaluate(test_ds_adv, models_to_eval, reference_model):
    """Perturbs the entire test set using adversarial training and computes metrics
    over that set.

    :returns: A tuple for four elements: a Tensor of the perturbed images; a List of
    the labels; a List of the predictions for the models; and a nested dict of
    per-model metrics.
    """
    print("[INFO] perturbing images and evaluating models on perturbed data...")
    perturbed_images, labels, predictions = [], [], []

    # TODO(jpgard): implement additional metrics as tf.keras.Metric subclasses; see
    #  https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/metrics/Metric

    metrics = {model_name: [tf.keras.metrics.SparseCategoricalAccuracy(name=ACC),
                            tf.keras.metrics.SparseCategoricalCrossentropy(name=CE),
                            ]
               for model_name in models_to_eval.keys()
               }
    for batch in test_ds_adv:
        perturbed_batch = reference_model.perturb_on_batch(batch)
        # Clipping makes perturbed examples have the same range as regular ones.
        perturbed_batch[IMAGE_INPUT_NAME] = tf.clip_by_value(
            perturbed_batch[IMAGE_INPUT_NAME], 0.0, 1.0)
        y_true = tf.argmax(perturbed_batch.pop(LABEL_INPUT_NAME), axis=-1)
        perturbed_images.append(perturbed_batch[IMAGE_INPUT_NAME].numpy())
        labels.append(y_true.numpy())
        predictions.append({})
        for model_name, model in models_to_eval.items():
            y_pred = model(perturbed_batch)
            predictions[-1][model_name] = tf.argmax(y_pred, axis=-1).numpy()
            for i in range(len(metrics[model_name])):
                metrics[model_name][i](y_true, y_pred)
    print("[INFO] perturbation evaluation complete.")
    metrics = metrics_to_dict(metrics)
    return perturbed_images, labels, predictions, metrics


def make_compiled_reference_model(model_base, adv_config, model_compile_args):
    reference_model = nsl.keras.AdversarialRegularization(
        model_base,
        label_keys=[LABEL_INPUT_NAME],
        adv_config=adv_config)
    reference_model.compile(**model_compile_args)
    return reference_model


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


def add_adversarial_metric_names_to_list(metrics_list):
    """The adversarial training has different metrics -- add these to metrics_list."""
    return ["total_combined_loss", ] + metrics_list + ["adversarial_loss", ]


def get_n_from_file_pattern(file_pattern):
    return len(glob.glob(file_pattern))


def compute_n_train_n_val(n_train_val, val_frac):
    n_val = int(n_train_val * val_frac)
    n_train = n_train_val - n_val
    return n_train, n_val


def steps_per_epoch(n, batch_size):
    return n // batch_size


def load_model_weights_from_flags(model: keras.Model, flags, is_adversarial: bool):
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
        filepath = make_ckpt_filepath(flags, is_adversarial=is_adversarial)
        print("[INFO] loading weights from{}".format(filepath))
        model.load_weights(filepath=filepath)
    return


def get_model_from_flags(flags):
    """Parse the flags to construct a model of the appropriate type with the specified
    architecture and hyperparamters."""
    if flags.model_type == keys.VGGFACE_2_MODEL:
        model = vggface2_model(dropout_rate=flags.dropout_rate,
                               activation=flags.model_activation)
    elif flags.model_type == keys.FACENET_MODEL:
        model = facenet_model(dropout_rate=flags.dropout_rate,
                              activation=flags.model_activation)
    else:
        raise NotImplementedError("The model type {} has not been implemented".format(
            flags.model_type))
    return model


def get_model_img_shape_from_flags(flags):
    """Fetch the (height, width) of the default model image shape."""
    if flags.model_type == keys.VGGFACE_2_MODEL:
        return keys.VGGFACE_2_IMG_SHAPE
    elif flags.model_type == keys.FACENET_MODEL:
        return keys.FACENET_IMG_SHAPE
    else:
        raise NotImplementedError("The model type {} has not been implemented".format(
            flags.model_type))
