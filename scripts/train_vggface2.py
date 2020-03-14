"""
Script to fine-tune pretrained VGGFace2 model.

usage:

# set the gpu
export GPU_ID="2"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# run the script
python scripts/train_vggface2.py \
    --img_dir /Users/jpgard/Documents/research/vggface2/train_partitioned_by_label
    /mouth_open
    --train_base --train_adversarial
"""

import glob
import math
import numpy as np
import time

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, \
    FalsePositives, FalseNegatives
import tensorflow_datasets as tfds
import neural_structured_learning as nsl

from dro.training.models import vggface2_model
from dro.utils.training_utils import preprocess_dataset, process_path, make_callbacks, \
    make_model_uid
from dro.utils.viz import show_batch

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

# the vggface2/training parameters
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("epochs", 5, "the number of training epochs")
flags.DEFINE_string("img_dir", None, "directory containing the aligned celeba images")
flags.DEFINE_float("learning_rate", 0.001, "learning rate to use")
flags.DEFINE_float("dropout_rate", 0.8, "dropout rate to use in fully-connected layers")
flags.DEFINE_bool("train_adversarial", False, "whether to train an adversarial model.")
flags.DEFINE_bool("train_base", True, "whether to train the base (non-adversarial) "
                                      "model.")
flags.DEFINE_float("val_frac", 0.1, "proportion of data to use for validation")
flags.DEFINE_float("test_frac", 0.1, "proportion of data to use for testing")
flags.DEFINE_string("label_name", None,
                    "name of the prediction label (e.g. sunglasses, mouth_open)",
                    )
flags.mark_flag_as_required("label_name")
flags.DEFINE_bool("debug", False,
                  "whether to run in debug mode (super short iterations to check for "
                  "bugs)")

# the wrm parameters
flags.DEFINE_multi_float('wrm_eps', 1.3,
                         'epsilon value to use for Wasserstein robust method; note that '
                         'original default value is 1.3.')
flags.DEFINE_integer('wrm_ord', 2, 'order of norm to use in Wasserstein robust method')
flags.DEFINE_integer('wrm_steps', 15,
                     'number of steps to use in Wasserstein robus method')

# the adversarial training parameters
flags.DEFINE_float('adv_multiplier', 0.2,
                   " The weight of adversarial loss in the training objective, relative "
                   "to the labeled loss")
flags.DEFINE_float('adv_step_size', 0.2, "The magnitude of adversarial perturbation.")
flags.DEFINE_string('adv_grad_norm', 'infinity',
                    "The norm to measure the magnitude of adversarial perturbation.")

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'


def convert_to_dictionaries(image, label):
    """Convert a set of x,y tuples to a dict for use in adversarial training."""
    return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}


def compute_element_wise_loss(preds, labels):
    test_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds)
    element_wise_test_loss = tf.reduce_sum(test_loss, axis=1)
    return element_wise_test_loss


def main(argv):
    filepattern = str(FLAGS.img_dir + '/*/*/*.jpg')
    N = len(glob.glob(filepattern))
    list_ds = tf.data.Dataset.list_files(filepattern, shuffle=True,
                                         seed=2974)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    input_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # Save a sample batch to png for debugging
    image_batch, label_batch = next(iter(input_ds))
    show_batch(image_batch.numpy(), label_batch.numpy(),
               fp="./debug/sample_batch{}.png".format(int(time.time())))

    custom_vgg_model = vggface2_model(dropout_rate=FLAGS.dropout_rate)
    n_val = int(N * FLAGS.val_frac)
    n_test = int(N * FLAGS.test_frac)
    n_train = N - n_val - n_test
    if not FLAGS.debug:
        steps_per_train_epoch = math.floor(n_train / FLAGS.batch_size)
        steps_per_val_epoch = math.floor(n_val / FLAGS.batch_size)
        steps_per_test_epoch = math.floor(n_test / FLAGS.batch_size)
    else:
        print("[INFO] running in debug mode")
        steps_per_train_epoch = 1
        steps_per_val_epoch = 1
        steps_per_test_epoch = 1

    # build the datasets
    val_ds_pre = input_ds.take(n_val)
    test_ds_pre = input_ds.take(n_test)
    val_ds = preprocess_dataset(val_ds_pre, repeat_forever=True,
                                batch_size=FLAGS.batch_size,
                                prefetch_buffer_size=AUTOTUNE)
    test_ds = preprocess_dataset(test_ds_pre, repeat_forever=False,
                                 batch_size=FLAGS.batch_size,
                                 prefetch_buffer_size=AUTOTUNE)
    test_ds_inputs = test_ds.map(lambda x, y: x)
    test_ds_labels = test_ds.map(lambda x, y: y)
    train_ds = preprocess_dataset(input_ds, repeat_forever=True,
                                  batch_size=FLAGS.batch_size,
                                  prefetch_buffer_size=AUTOTUNE)
    # The metrics to optimize during training
    train_metrics = ['accuracy',
                     AUC(name='auc'),
                     TruePositives(name='tp'),
                     FalsePositives(name='fp'),
                     TrueNegatives(name='tn'),
                     FalseNegatives(name='fn')
                     ]
    custom_vgg_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                             loss=tf.keras.losses.CategoricalCrossentropy(
                                 from_logits=True),
                             metrics=train_metrics
                             )
    custom_vgg_model.summary()
    if FLAGS.train_base:
        print("[INFO] training base model")
        callbacks = make_callbacks(FLAGS, is_adversarial=False)
        custom_vgg_model.fit_generator(train_ds,
                                       steps_per_epoch=steps_per_train_epoch,
                                       epochs=FLAGS.epochs,
                                       callbacks=callbacks,
                                       validation_data=val_ds,
                                       validation_steps=steps_per_val_epoch)
        # Fetch preds and test labels; these are both numpy arrays of shape [n_test, 2]
        preds = custom_vgg_model.predict_generator(test_ds_inputs)
        labels = np.concatenate([y for y in tfds.as_numpy(test_ds_labels)])
        element_wise_test_loss = compute_element_wise_loss(preds=preds, labels=labels)
        print("Final non-adversarial test loss: mean {} std ({})".format(
            tf.reduce_mean(element_wise_test_loss),
            tf.math.reduce_std(element_wise_test_loss))
        )
        loss_filename = "./metrics/{}-test_loss.txt".format(
            make_model_uid(FLAGS,
                           is_adversarial=False))
        np.savetxt(loss_filename, element_wise_test_loss)

    if FLAGS.train_adversarial:
        print("[INFO] training adversarial model")
        # the adversarial training block

        adv_config = nsl.configs.make_adv_reg_config(
            multiplier=FLAGS.adv_multiplier,
            adv_step_size=FLAGS.adv_step_size,
            adv_grad_norm=FLAGS.adv_grad_norm
        )
        base_adv_model = vggface2_model(dropout_rate=FLAGS.dropout_rate)
        adv_model = nsl.keras.AdversarialRegularization(
            base_adv_model,
            label_keys=[LABEL_INPUT_NAME],
            adv_config=adv_config
        )
        train_ds_adv = train_ds.map(convert_to_dictionaries)
        val_ds_adv = val_ds.map(convert_to_dictionaries)

        # The test dataset can be initialized from test_ds_pre(); the prepare_...
        # function will re-initialize it as a fresh generator from the same elements.

        test_ds_adv = preprocess_dataset(
            test_ds_pre,
            repeat_forever=False,
            batch_size=FLAGS.batch_size,
            prefetch_buffer_size=AUTOTUNE)
        test_ds_adv_labels = test_ds_adv.map(lambda x, y: y)
        test_ds_adv = test_ds_adv.map(convert_to_dictionaries)

        # We only need the labels for the adversarial model; the prediction function
        # takes the combined inputs in AT.

        adv_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                          metrics=train_metrics)
        callbacks = make_callbacks(FLAGS, is_adversarial=True)
        adv_model.fit_generator(train_ds_adv,
                                steps_per_epoch=steps_per_train_epoch,
                                epochs=FLAGS.epochs,
                                callbacks=callbacks,
                                validation_data=val_ds_adv,
                                validation_steps=steps_per_val_epoch
                                )

        # Fetch preds and test labels; these are both numpy arrays of shape [n_test, 2]
        preds = adv_model.predict_generator(test_ds_adv)
        labels = np.concatenate([y for y in tfds.as_numpy(test_ds_adv_labels)])
        element_wise_test_loss = compute_element_wise_loss(preds=preds, labels=labels)
        print("Final adversarial test loss: mean {} std ({})".format(
            tf.reduce_mean(element_wise_test_loss),
            tf.math.reduce_std(element_wise_test_loss))
        )
        loss_filename = "./metrics/{}-test_loss.txt".format(
            make_model_uid(FLAGS, is_adversarial=True))
        np.savetxt(loss_filename, element_wise_test_loss)


if __name__ == "__main__":
    app.run(main)
