"""
Script to fine-tune pretrained VGGFace2 model.

usage:

# set the gpu
export GPU_ID="3"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# run the script
export LABEL="Sunglasses"
export DIR="/projects/grail/jpgard/vggface2/annotated_partitioned_by_label/"
python3 scripts/train_vggface2.py \
    --label_name $LABEL \
    --test_dir ${DIR}/test/${LABEL} \
    --train_dir ${DIR}/train/${LABEL} \
    --adv_step_size 0.05 --epochs 40



"""

from collections import OrderedDict
import glob
import time
import numpy as np

from absl import app
from absl import flags
import tensorflow as tf
import neural_structured_learning as nsl

from dro.keys import LABEL_INPUT_NAME
from dro.training.models import vggface2_model
from dro.utils.training_utils import preprocess_dataset, process_path, make_callbacks, \
    write_test_metrics_to_csv, get_train_metrics, convert_to_dictionaries
from dro.utils.viz import show_batch

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

# the vggface2/training parameters
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("epochs", 250, "the number of training epochs")
flags.DEFINE_string("train_dir", None, "directory containing the training images")
flags.DEFINE_string("test_dir", None, "directory containing the test images")
flags.DEFINE_string("ckpt_dir", "./training-logs", "directory to save/load checkpoints "
                                                   "from")
flags.DEFINE_float("learning_rate", 0.01, "learning rate to use")
flags.DEFINE_float("dropout_rate", 0.8, "dropout rate to use in fully-connected layers")
flags.DEFINE_bool("train_adversarial", True, "whether to train an adversarial model.")
flags.DEFINE_bool("train_base", True, "whether to train the base (non-adversarial) "
                                      "model.")
flags.DEFINE_string("base_model_ckpt", None,
                    "optional manually-specified checkpoint to use to load the base "
                    "model.")
flags.DEFINE_string("adv_model_ckpt", None,
                    "optional manually-specified checkpoint to use to load the "
                    "adversarial model.")
flags.DEFINE_bool("perturbation_analysis", True, "whether to conduct a perturbation "
                                                 "analysis after completing training.")
flags.DEFINE_float("val_frac", 0.1, "proportion of data to use for validation")
flags.DEFINE_string("label_name", None,
                    "name of the prediction label (e.g. sunglasses, mouth_open)",
                    )
flags.mark_flag_as_required("label_name")
flags.DEFINE_bool("debug", False,
                  "whether to run in debug mode (super short iterations to check for "
                  "bugs)")

# the adversarial training parameters
flags.DEFINE_float('adv_multiplier', 0.2,
                   " The weight of adversarial loss in the training objective, relative "
                   "to the labeled loss. e.g. if this is 0.2, The model minimizes "
                   "(mean_crossentropy_loss + 0.2 * adversarial_regularization) ")
flags.DEFINE_float('adv_step_size', 0.2, "The magnitude of adversarial perturbation.")
flags.DEFINE_string('adv_grad_norm', 'infinity',
                    "The norm to measure the magnitude of adversarial perturbation.")

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_n_from_file_pattern(file_pattern):
    return len(glob.glob(file_pattern))


def steps_per_epoch(n):
    return n // FLAGS.batch_size


def main(argv):
    train_file_pattern = str(FLAGS.train_dir + '/*/*/*.jpg')
    test_file_pattern = str(FLAGS.test_dir + '/*/*/*.jpg')
    n_train_val = get_n_from_file_pattern(train_file_pattern)
    n_test = get_n_from_file_pattern(test_file_pattern)

    # Create the datasets and process files to create (x,y) tuples. Set
    # `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_val_input_ds = tf.data.Dataset.list_files(
        train_file_pattern, shuffle=True, seed=2974) \
        .map(process_path, num_parallel_calls=AUTOTUNE)
    test_input_ds = tf.data.Dataset.list_files(test_file_pattern, shuffle=False) \
        .map(process_path, num_parallel_calls=AUTOTUNE)

    vgg_model_base = vggface2_model(dropout_rate=FLAGS.dropout_rate)
    n_val = int(n_train_val * FLAGS.val_frac)
    n_train = n_train_val - n_val
    print("[INFO] {n_train} training observations; {n_val} validation observations"
          "{n_test} testing observations".format(n_train=n_train,
                                                 n_val=n_val,
                                                 n_test=n_test,
                                                 ))
    if not FLAGS.debug:
        steps_per_train_epoch = steps_per_epoch(n_train)
        steps_per_val_epoch = steps_per_epoch(n_val)
    else:
        print("[INFO] running in debug mode")
        steps_per_train_epoch = 1
        steps_per_val_epoch = 1

    # Build the datasets. Take the validation samples from the training data prior to
    # doing any preprocessing or repeating; this ensures validation and train sets do
    # not overlap. Note that we create new variables (instead of reassigning the same
    # variable) because the original, unprocessed versions are needed for re-processing
    # prior to the adversarial training below.

    val_ds_pre = train_val_input_ds.take(n_val)
    val_ds = preprocess_dataset(val_ds_pre, repeat_forever=True,
                                batch_size=FLAGS.batch_size,
                                prefetch_buffer_size=AUTOTUNE)
    test_ds = preprocess_dataset(test_input_ds, repeat_forever=False,
                                 shuffle=False,
                                 batch_size=FLAGS.batch_size,
                                 prefetch_buffer_size=AUTOTUNE)

    train_ds = preprocess_dataset(train_val_input_ds, repeat_forever=True,
                                  batch_size=FLAGS.batch_size,
                                  prefetch_buffer_size=AUTOTUNE)

    # Save a sample batch to png for debugging
    image_batch, label_batch = next(iter(train_ds))
    show_batch(image_batch.numpy(), label_batch.numpy(),
               fp="./debug/sample_batch_label-train-{}-{}.png".format(FLAGS.label_name,
                                                                      int(time.time()))
               )

    image_batch_test, label_batch_test = next(iter(test_ds))
    show_batch(image_batch_test.numpy(), label_batch_test.numpy(),
               fp="./debug/sample_batch_label-test-{}-{}.png".format(FLAGS.label_name,
                                                                     int(time.time()))
               )
    # The metrics to optimize during training
    train_metrics_dict = get_train_metrics()
    # .evaluate() automatically prepends the loss(es), so it will always include at
    # least categorical_crossentropy (adversarial also adds the AT loss terms)
    train_metrics_names = ["categorical_crossentropy", ] + list(train_metrics_dict.keys())
    train_metrics = list(train_metrics_dict.values())

    model_compile_args = {
        "optimizer": tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate),
        "loss": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        "metrics": train_metrics
    }

    vgg_model_base.compile(**model_compile_args)
    vgg_model_base.summary()

    # Shared training arguments for the model fitting.
    train_args = {"steps_per_epoch": steps_per_train_epoch,
                  "epochs": FLAGS.epochs,
                  "validation_steps": steps_per_val_epoch}
    from dro.utils.training_utils import make_ckpt_filepath

    # Base model training
    if FLAGS.train_base:
        print("[INFO] training base model")
        callbacks_base = make_callbacks(FLAGS, is_adversarial=False)
        vgg_model_base.fit_generator(train_ds, callbacks=callbacks_base,
                                     validation_data=val_ds, **train_args)

        # Fetch preds and test labels; these are both numpy arrays of shape [n_test, 2]
        test_metrics = vgg_model_base.evaluate_generator(test_ds)
        assert len(train_metrics_names) == len(test_metrics)
        test_metrics_dict = OrderedDict(zip(train_metrics_names, test_metrics))
        write_test_metrics_to_csv(test_metrics_dict, FLAGS, is_adversarial=False)

    elif FLAGS.base_model_ckpt:
        # load the model from specified checkpoint path instead of training it
        vgg_model_base.load_weights(filepath=FLAGS.base_model_ckpt)
        # load the model from default checkpoint path instead of training it
    else:
        vgg_model_base.load_weights(filepath=make_ckpt_filepath(FLAGS,
                                                                is_adversarial=False))

    # Adversarial model training
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

    adv_model.compile(**model_compile_args)

    train_ds_adv = train_ds.map(convert_to_dictionaries)
    val_ds_adv = val_ds.map(convert_to_dictionaries)

    # The test dataset can be initialized from test_input_ds; preprocess_dataset()
    # will re-initialize it as a fresh generator from the same elements.

    test_ds_adv = preprocess_dataset(
        test_input_ds,
        repeat_forever=False,
        shuffle=False,
        batch_size=FLAGS.batch_size,
        prefetch_buffer_size=AUTOTUNE)
    test_ds_adv = test_ds_adv.map(convert_to_dictionaries)

    if FLAGS.train_adversarial:
        print("[INFO] training adversarial model")
        callbacks_adv = make_callbacks(FLAGS, is_adversarial=True)
        adv_model.fit_generator(train_ds_adv, callbacks=callbacks_adv,
                                validation_data=val_ds_adv,
                                **train_args)
        test_metrics_adv = adv_model.evaluate_generator(test_ds_adv)
        # The evaluate_generator() function adds the total_loss and adversarial_loss,
        # so here we include those.
        test_metrics_adv_names = \
            ["total_combined_loss", ] + train_metrics_names + ["adversarial_loss", ]
        assert len(test_metrics_adv_names) == len(test_metrics_adv)
        test_metrics_adv = OrderedDict(zip(test_metrics_adv_names, test_metrics_adv))
        write_test_metrics_to_csv(test_metrics_adv, FLAGS, is_adversarial=True)

    elif FLAGS.adv_model_ckpt:  # load the model
        adv_model.load_weights(FLAGS.adv_model_ckpt)
    else:
        adv_model.load_weights(filepath=make_ckpt_filepath(FLAGS, is_adversarial=True))

    if FLAGS.perturbation_analysis:
        # First, create a reference model from the non-adversarially-trained model,
        # which will be used to generate perturbations.

        print("[INFO] generating adversarial samples to compare the models")
        from dro.utils.training_utils import perturb_and_evaluate, \
            make_compiled_reference_model
        from dro.utils.training_utils import make_model_uid
        from dro.utils.viz import show_adversarial_resuts
        reference_model = make_compiled_reference_model(vgg_model_base, adv_config,
                                                        model_compile_args)

        models_to_eval = {
            'base': vgg_model_base,
            'adv-regularized': adv_model.base_model
        }

        perturbed_images, labels, predictions, metrics = perturb_and_evaluate(
            test_ds_adv, models_to_eval, reference_model)

        adv_image_basename = "./debug/adv-examples-{}".format(make_model_uid(FLAGS))

        show_adversarial_resuts(n_batches=10,
                                perturbed_images=perturbed_images,
                                labels=labels,
                                predictions=predictions,
                                fp_basename=adv_image_basename,
                                batch_size=FLAGS.batch_size)


if __name__ == "__main__":
    app.run(main)
