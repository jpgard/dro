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
"""

import math

from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, \
    FalsePositives, FalseNegatives
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
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


def convert_to_dictionaries(image, label):
    """Convert a set of x,y tuples to a dict for use in adversarial training."""
    return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}


def make_model_uid():
    """Create a unique identifier for the model."""
    model_uid = """bs{batch_size}e{epochs}lr{lr}dropout{dropout_rate}""".format(
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        lr=FLAGS.learning_rate,
        dropout_rate=FLAGS.dropout_rate
    )
    return model_uid


def make_callbacks(adversarial_training: bool):
    """Create the callbacks for training, including properly naming files."""
    callback_uid = make_model_uid()
    if adversarial_training:
        callback_uid = "{callback_uid}-adv-m{mul}-s{step}-n{norm}".format(
            callback_uid=callback_uid, mul=FLAGS.adv_multiplier,
            step=FLAGS.adv_step_size, norm=FLAGS.adv_grad_norm)
    tensorboard_callback = TensorBoard(
        log_dir='./training-logs/{}'.format(callback_uid),
        batch_size=FLAGS.batch_size,
        write_graph=True,
        write_grads=True,
        update_freq='epoch')
    csv_callback = CSVLogger("./metrics/{}-vggface2-training.log".format(callback_uid))
    return [tensorboard_callback, csv_callback]


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
    N = 1000
    n_val = int(N * FLAGS.val_frac)
    n_train = N - n_val
    steps_per_train_epoch = math.floor(n_train / FLAGS.batch_size)
    steps_per_val_epoch = math.floor(n_val / FLAGS.batch_size)

    # build the datasets
    val_ds = labeled_ds.take(n_val)
    val_ds = prepare_dataset_for_training(val_ds, repeat_forever=True,
                                          batch_size=FLAGS.batch_size,
                                          prefetch_buffer_size=AUTOTUNE)
    # val_ds = val_ds.make_one_shot_iterator()
    train_ds = prepare_dataset_for_training(labeled_ds, repeat_forever=True,
                                            batch_size=FLAGS.batch_size,
                                            prefetch_buffer_size=AUTOTUNE)
    # train_ds = train_ds.make_one_shot_iterator()

    custom_vgg_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                             loss=tf.keras.losses.CategoricalCrossentropy(
                                 from_logits=True),
                             metrics=['accuracy',
                                      AUC(name='auc'),
                                      TruePositives(name='tp'),
                                      FalsePositives(name='fp'),
                                      TrueNegatives(name='tn'),
                                      FalseNegatives(name='fn')
                                      ]
                             )
    custom_vgg_model.summary()
    if FLAGS.train_base:
        print("[INFO] training base model")
        callbacks = make_callbacks(adversarial_training=False)
        custom_vgg_model.fit_generator(train_ds, steps_per_epoch=steps_per_train_epoch,
                                       epochs=FLAGS.epochs,
                                       callbacks=callbacks)
    if FLAGS.train_adversarial:
        print("[INFO] training adversarial model")
        # the adversarial training block
        import neural_structured_learning as nsl
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
        train_set_for_adv_model = train_ds.map(convert_to_dictionaries)
        test_set_for_adv_model = val_ds.map(convert_to_dictionaries)
        adv_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                          loss=tf.keras.losses.CategoricalCrossentropy(
                              from_logits=True),
                          metrics=['accuracy',
                                   AUC(name='auc'),
                                   TruePositives(name='tp'),
                                   FalsePositives(name='fp'),
                                   TrueNegatives(name='tn'),
                                   FalseNegatives(name='fn')
                                   ])
        callbacks = make_callbacks(adversarial_training=True)
        adv_model.fit_generator(train_set_for_adv_model,
                                steps_per_epoch=steps_per_train_epoch,
                                epochs=FLAGS.epochs,
                                callbacks=callbacks
                                )


if __name__ == "__main__":
    app.run(main)
