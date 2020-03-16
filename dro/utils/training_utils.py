from collections import OrderedDict
from itertools import islice
import os

import pandas as pd
import numpy as np
import tensorflow
import tensorflow as tf
import neural_structured_learning as nsl


from tensorflow_core.python.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, \
    FalsePositives, FalseNegatives

from dro.keys import IMAGE_INPUT_NAME, LABEL_INPUT_NAME

AUTOTUNE = tf.data.experimental.AUTOTUNE
TEST_MODE = "test"
TRAIN_MODE = "train"


def get_batch(dataset_iterator, batch_size):
    slice = tuple(islice(dataset_iterator, batch_size))
    batch_x = np.stack([i[0] for i in slice], axis=0)
    batch_y = np.stack([i[1] for i in slice], axis=0)
    return batch_x, batch_y


def preprocess_dataset(
        ds, cache=True, shuffle_buffer_size=1000,
        repeat_forever=False, batch_size: int = None,
        prefetch_buffer_size=AUTOTUNE, shuffle=True,
        epochs: int = None):
    """Shuffle, repeat, batch, and prefetch the dataset."""
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.

    # If the data doesn't fit in memory see the example usage at the
    # bottom of the page here:
    # https: // www.tensorflow.org / tutorials / load_data / images
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    if repeat_forever:
        ds = ds.repeat()
    elif epochs:
        ds = ds.repeat(epochs)
    if batch_size:
        ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=prefetch_buffer_size)
    return ds


def process_path(file_path, crop=True, labels=True):
    """Load the data from file_path. Returns either an (x,y) tuplesif
    labels=True, or just x if label=False."""
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, crop=crop)
    if not labels:
        return img
    else:
        label = get_label(file_path)
        label = tf.one_hot(label, 2)
        return img, label


def random_crop_and_resize(img):
    # resize to a square image of 256 x 256, then crop to random 224 x 224
    if len(img.shape) < 4:  # add a batch dimension if one does not exist
        img = tf.expand_dims(img, axis=0)
    img = tf.image.resize_images(img, size=(256, 256), preserve_aspect_ratio=True)
    img = tf.image.resize_with_crop_or_pad(img, target_height=256, target_width=256)
    img = tf.squeeze(img, axis=0)
    img = tf.image.random_crop(img, (224, 224, 3))
    return img


def decode_img(img, normalize_by_channel=False, crop=True):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    if crop:
        img = random_crop_and_resize(img)

    # Apply normalization: subtract the channel-wise mean from each image as in
    # https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/utils.py ;
    # divide means by 255.0 since the conversion above restricts to range [0,1].
    if normalize_by_channel:
        ch1mean = tf.constant(91.4953 / 255.0, shape=(224, 224, 1))
        ch2mean = tf.constant(103.8827 / 255.0, shape=(224, 224, 1))
        ch3mean = tf.constant(131.0912 / 255.0, shape=(224, 224, 1))
        channel_norm_tensor = tf.concat([ch1mean, ch2mean, ch3mean], axis=2)
        img -= channel_norm_tensor
    return img


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


def get_label(file_path):
    # convert the path to a list of path components
    label = tf.strings.substr(file_path, -21, 1)
    # The second to last is the class-directory
    return tf.strings.to_number(label, out_type=tf.int32)


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
    train_metrics_dict = {'accuracy': 'accuracy',
                          'auc': AUC(name='auc'),
                          'tp': TruePositives(name='tp'),
                          'fp': FalsePositives(name='fp'),
                          'tn': TrueNegatives(name='tn'),
                          'fn': FalseNegatives(name='fn'),
                          'ce': tf.keras.losses.CategoricalCrossentropy(),
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


def make_callbacks(flags, is_adversarial: bool):
    """Create the callbacks for training, including properly naming files."""
    callback_uid = make_model_uid(flags, is_adversarial=is_adversarial)
    logdir = './training-logs/{}'.format(callback_uid)
    tensorboard_callback = TensorBoard(
        log_dir=logdir,
        batch_size=flags.batch_size,
        write_graph=True,
        write_grads=True,
        update_freq='epoch')
    csv_callback = make_csv_callback(flags, is_adversarial)
    ckpt_fp = os.path.join(logdir, callback_uid + ".ckpt")
    ckpt_callback = ModelCheckpoint(ckpt_fp,
                                    monitor='val_loss', verbose=0,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    save_freq='epoch',
                                    mode='auto')
    return [tensorboard_callback, csv_callback, ckpt_callback]


def make_model_uid(flags, is_adversarial=False):
    """Create a unique identifier for the model."""
    model_uid = """{label_name}bs{batch_size}e{epochs}lr{lr}dropout{dropout_rate}""" \
        .format(label_name=flags.label_name,
                batch_size=flags.batch_size,
                epochs=flags.epochs,
                lr=flags.learning_rate,
                dropout_rate=flags.dropout_rate
                )
    if is_adversarial:
        model_uid = "{model_uid}-{adv}-m{mul}-s{step}-n{norm}".format(
            adv=get_adversarial_mode(True),
            model_uid=model_uid, mul=flags.adv_multiplier,
            step=flags.adv_step_size, norm=flags.adv_grad_norm)
    return model_uid


def perturb_and_evaluate(test_ds_adv, models_to_eval, reference_model):
    perturbed_images, labels, predictions = [], [], []
    metrics = {name: tf.keras.metrics.SparseCategoricalAccuracy()
               for name in models_to_eval.keys()
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
        for name, model in models_to_eval.items():
            y_pred = model(perturbed_batch)
            metrics[name](y_true, y_pred)
            predictions[-1][name] = tf.argmax(y_pred, axis=-1).numpy()

    return perturbed_images, labels, predictions


def make_compiled_reference_model(model_base, adv_config, model_compile_args):
    reference_model = nsl.keras.AdversarialRegularization(
        model_base,
        label_keys=[LABEL_INPUT_NAME],
        adv_config=adv_config)
    reference_model.compile(**model_compile_args)
    return reference_model
