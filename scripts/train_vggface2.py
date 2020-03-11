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

import math
import os
import numpy as np

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, \
    FalsePositives, FalseNegatives
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
import tensorflow_datasets as tfds
import neural_structured_learning as nsl

from dro.training.models import vggface2_model
from dro.utils.training_utils import prepare_dataset_for_training

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

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'
N = 1000  # the number of training observations


def convert_to_dictionaries(image, label):
    """Convert a set of x,y tuples to a dict for use in adversarial training."""
    return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}


def make_model_uid(is_adversarial=False):
    """Create a unique identifier for the model."""
    model_uid = """bs{batch_size}e{epochs}lr{lr}dropout{dropout_rate}""".format(
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        lr=FLAGS.learning_rate,
        dropout_rate=FLAGS.dropout_rate
    )
    if is_adversarial:
        model_uid = "{model_uid}-adv-m{mul}-s{step}-n{norm}".format(
            model_uid=model_uid, mul=FLAGS.adv_multiplier,
            step=FLAGS.adv_step_size, norm=FLAGS.adv_grad_norm)
    return model_uid


def make_callbacks(is_adversarial: bool):
    """Create the callbacks for training, including properly naming files."""
    callback_uid = make_model_uid(is_adversarial=is_adversarial)
    logdir = './training-logs/{}'.format(callback_uid)
    tensorboard_callback = TensorBoard(
        log_dir=logdir,
        batch_size=FLAGS.batch_size,
        write_graph=True,
        write_grads=True,
        update_freq='epoch')
    csv_fp = "./metrics/{}-vggface2-training.log".format(callback_uid)
    csv_callback = CSVLogger(csv_fp)
    ckpt_fp = os.path.join(logdir, callback_uid + ".ckpt")
    ckpt_callback = ModelCheckpoint(ckpt_fp,
                                    monitor='val_loss', verbose=0,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    save_freq='epoch',
                                    mode='auto')
    return [tensorboard_callback, csv_callback, ckpt_callback]


def compute_element_wise_loss(preds, labels):
    test_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds)
    element_wise_test_loss = tf.reduce_sum(test_loss, axis=1)
    return element_wise_test_loss


def main(argv):
    list_ds = tf.data.Dataset.list_files(str(FLAGS.img_dir + '/*/*/*.jpg'), shuffle=True,
                                         seed=2974)

    # for f in list_ds.take(3):
    #     print(f.numpy())

    def get_label(file_path):
        # convert the path to a list of path components
        label = tf.strings.substr(file_path, -21, 1)
        # The second to last is the class-directory
        return tf.strings.to_number(label, out_type=tf.int32)

    def decode_img(img, normalize_by_channel=False):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize to a square image of 256 x 256, then crop to random 224 x 224
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_images(img, size=(256, 256), preserve_aspect_ratio=True)
        img = tf.image.resize_with_crop_or_pad(img, target_height=256, target_width=256)
        img = tf.squeeze(img, axis=0)
        img = tf.image.random_crop(img, (224, 224, 3))

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

    def process_path(file_path):
        label = get_label(file_path)
        label = tf.one_hot(label, 2)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label
        # return {IMAGE_INPUT_NAME: img, LABEL_INPUT_NAME: label}

    # def tuple_to_dict(element):
    #     x,y = element
    #     return {"image": x, "label": y}

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # for image, label in labeled_ds.take(10):
    #     print("Image shape: ", image.numpy().shape)
    #     print("Label: ", label.numpy())

    # TODO(jpgard): save batch to pdf instead
    # image_batch, label_batch = next(iter(train_ds))
    # from dro.utils.vis import show_batch
    # show_batch(image_batch.numpy(), label_batch.numpy())

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
        steps_per_train_epoch = 5
        steps_per_val_epoch = 5
        steps_per_test_epoch = 5

    # build the datasets
    val_ds = labeled_ds.take(n_val)
    test_ds = labeled_ds.take(n_test)
    val_ds = prepare_dataset_for_training(val_ds, repeat_forever=True,
                                          batch_size=FLAGS.batch_size,
                                          prefetch_buffer_size=AUTOTUNE)
    test_ds = prepare_dataset_for_training(test_ds, repeat_forever=False,
                                           batch_size=FLAGS.batch_size,
                                           prefetch_buffer_size=AUTOTUNE)
    test_ds_inputs = test_ds.map(lambda x, y: x)
    test_ds_labels = test_ds.map(lambda x, y: y)
    train_ds = prepare_dataset_for_training(labeled_ds, repeat_forever=True,
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
        callbacks = make_callbacks(is_adversarial=False)
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
        loss_filename = "./metrics/{}-test_loss.txt".format(make_model_uid(
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
        # The test dataset can be a copy of the original test set; the prepare_...
        # function re-initializes it as a fresh generator.
        test_ds_adv = prepare_dataset_for_training(
            test_ds,
            repeat_forever=False,
            batch_size=FLAGS.batch_size,
            prefetch_buffer_size=AUTOTUNE)
        test_ds_adv_inputs = test_ds.map(lambda x, y: x)
        test_ds_adv_labels = test_ds.map(lambda x, y: y)
        adv_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                          metrics=train_metrics)
        callbacks = make_callbacks(is_adversarial=True)
        adv_model.fit_generator(train_ds_adv,
                                steps_per_epoch=steps_per_train_epoch,
                                epochs=FLAGS.epochs,
                                callbacks=callbacks,
                                validation_data=val_ds_adv,
                                validation_steps=steps_per_val_epoch
                                )
        # Fetch preds and test labels; these are both numpy arrays of shape [n_test, 2]
        preds = adv_model.predict_generator(test_ds_adv_inputs)
        labels = np.concatenate([y for y in tfds.as_numpy(test_ds_adv_labels)])
        element_wise_test_loss = compute_element_wise_loss(preds=preds, labels=labels)
        print("Final adversarial test loss: mean {} std ({})".format(
            tf.reduce_mean(element_wise_test_loss),
            tf.math.reduce_std(element_wise_test_loss))
        )
        loss_filename = "./metrics/{}-test_loss.txt".format(make_model_uid(
            is_adversarial=False))
        np.savetxt(loss_filename, element_wise_test_loss)


if __name__ == "__main__":
    app.run(main)
