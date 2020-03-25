"""
A script to conduct adversarial analysis of pre-trained models.

The script applies a range of adversarial perturbations to a set of test images from
the LFW dataset, and evaluates classifier accuracy on those images. Accuracy is
reported  by image
subgroups.

usage:
export LABEL="Mouth_Open"
export LABEL="Sunglasses"
export LABEL="Male"
export SS=0.05
export EPOCHS=40

export DIR="/projects/grail/jpgard/lfw"
python3 scripts/adversarial_analysis.py \
    --anno_fp ${DIR}/lfw_attributes_cleaned.txt \
    --test_dir ${DIR}/lfw-deepfunneled \
    --label_name $LABEL \
    --slice_attribute_name $SLICE_ATTR \
    --adv_step_size $SS \
    --epochs $EPOCHS --experiment_uid TMP

for SLICE_ATTR in "Asian" "Senior" "Male" "Black"
do
    python3 scripts/adversarial_analysis.py \
    --anno_fp ${DIR}/lfw_attributes_cleaned.txt \
    --test_dir ${DIR}/lfw-deepfunneled \
    --label_name $LABEL \
    --slice_attribute_name $SLICE_ATTR \
    --adv_step_size $SS \
    --epochs $EPOCHS
done
"""

from absl import app, flags
import time

import tensorflow as tf
import pandas as pd

from dro.utils.lfw import apply_thresh, \
    get_annotated_data_df, LABEL_COLNAME, ATTR_COLNAME
from dro.utils.training_utils import pred_to_binary
from dro.training.models import vggface2_model
import neural_structured_learning as nsl
from dro.keys import LABEL_INPUT_NAME, FILENAME_COLNAME
from dro.utils.viz import show_batch
from dro.utils.training_utils import get_train_metrics
from dro.utils.training_utils import make_ckpt_filepath
from dro.utils.training_utils import perturb_and_evaluate, \
    make_compiled_reference_model
from dro.utils.training_utils import make_model_uid
from dro.utils.viz import show_adversarial_resuts
from dro.datasets import ImageDataset

tf.compat.v1.enable_eager_execution()

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS

flags.DEFINE_string("anno_fp", None, "path to annotations file for evaluation.")
flags.DEFINE_string("test_dir", None, "directory containing the test images")
flags.DEFINE_string("base_model_ckpt", None,
                    "optional manually-specified checkpoint to use to load the base "
                    "model.")
flags.DEFINE_string("adv_model_ckpt", None,
                    "optional manually-specified checkpoint to use to load the "
                    "adversarial model.")
flags.DEFINE_string("slice_attribute_name", None,
                    "attribute name to use from annotations.")
flags.DEFINE_string("label_name", None,
                    "name of the prediction label (e.g. sunglasses, mouth_open) in the "
                    "LFW/test dataset",
                    )
flags.mark_flag_as_required("label_name")
flags.mark_flag_as_required("slice_attribute_name")
flags.DEFINE_float("confidence_threshold", 0.5, "only predictions with absolute value "
                                                ">= this threshold are used ("
                                                "predictions are centered around zero) "
                                                "in order to ensure high-quality labels.")

# the vggface2/training parameters
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("epochs", 250, "the number of training epochs")
flags.DEFINE_string("ckpt_dir", "./training-logs", "directory to save/load checkpoints "
                                                   "from")
flags.DEFINE_float("learning_rate", 0.01, "learning rate to use")
flags.DEFINE_float("dropout_rate", 0.8, "dropout rate to use in fully-connected layers")
flags.mark_flag_as_required("label_name")
flags.DEFINE_string("experiment_uid", None, "Optional string identifier to be used to "
                                            "uniquely identify this experiment.")

# the adversarial training parameters
flags.DEFINE_float('adv_multiplier', 0.2,
                   " The weight of adversarial loss in the training objective, relative "
                   "to the labeled loss. e.g. if this is 0.2, The model minimizes "
                   "(mean_crossentropy_loss + 0.2 * adversarial_regularization) ")
flags.DEFINE_float('adv_step_size', 0.2, "The magnitude of adversarial perturbation.")
flags.DEFINE_string('adv_grad_norm', 'infinity',
                    "The norm to measure the magnitude of adversarial perturbation.")


def main(argv):
    # build a labeled dataset from the files
    annotated_files = get_annotated_data_df(anno_fp=FLAGS.anno_fp,
                                            test_dir=FLAGS.test_dir)
    assert len(annotated_files) > 0, "no files detected"

    # Create a DataFrame with columns for (filename, label, slice_attribute); the columns
    # need to be renamed to generic LABEL_COLNAME and ATTR_COLNAME in order to allow
    # for cases where label and attribute names are the same (e.g. slicing 'Male'
    # prediction by 'Male' attribute).

    dset_df = annotated_files.reset_index()[
        [FILENAME_COLNAME, FLAGS.label_name, FLAGS.slice_attribute_name]]
    dset_df.columns = [FILENAME_COLNAME, LABEL_COLNAME, ATTR_COLNAME]

    # Apply thresholding. We want observations which have absolute value greater than some
    # threshold (predictions close to zero have low confidence).

    dset_df = apply_thresh(dset_df, LABEL_COLNAME,
                           FLAGS.confidence_threshold)
    dset_df = apply_thresh(dset_df, ATTR_COLNAME,
                           FLAGS.confidence_threshold)

    dset_df[LABEL_COLNAME] = dset_df[LABEL_COLNAME].apply(pred_to_binary)
    dset_df[ATTR_COLNAME] = dset_df[ATTR_COLNAME].apply(
        pred_to_binary)

    # Break the input dataset into separate tf.Datasets based on the value of the slice
    # attribute.

    # Create and preprocess the dataset of examples where ATTR_COLNAME == 1
    preprocessing_kwargs = {"shuffle": False, "repeat_forever": False, "batch_size":
        FLAGS.batch_size}
    dset_attr_pos = ImageDataset()
    dset_attr_pos.from_dataframe(dset_df[dset_df[ATTR_COLNAME] == 1],
                                 label_name=LABEL_COLNAME)
    dset_attr_pos.preprocess(**preprocessing_kwargs)

    # Create and process the dataset of examples where ATTR_COLNAME == 1
    dset_attr_neg = ImageDataset()
    dset_attr_neg.from_dataframe(dset_df[dset_df[ATTR_COLNAME] == 0],
                                 label_name=LABEL_COLNAME)
    dset_attr_neg.preprocess(**preprocessing_kwargs)

    image_batch, label_batch = next(iter(dset_attr_pos.dataset))
    show_batch(image_batch.numpy(), label_batch.numpy(),
               fp="./debug/sample_batch_attr{}1-label{}-{}.png".format(
                   FLAGS.slice_attribute_name, FLAGS.label_name, int(time.time()))
               )
    image_batch, label_batch = next(iter(dset_attr_neg.dataset))
    show_batch(image_batch.numpy(), label_batch.numpy(),
               fp="./debug/sample_batch_attr{}0-label{}-{}.png".format(
                   FLAGS.slice_attribute_name, FLAGS.label_name, int(time.time()))
               )

    # Convert the datasets into dicts for use in adversarial model.
    dset_attr_neg.convert_to_dictionaries()
    dset_attr_pos.convert_to_dictionaries()
    attr_dsets = {"1": dset_attr_pos, "0": dset_attr_neg}

    # load the models
    train_metrics_dict = get_train_metrics()
    train_metrics_names = ["categorical_crossentropy", ] + list(train_metrics_dict.keys())
    train_metrics = list(train_metrics_dict.values())
    model_compile_args = {
        "optimizer": tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate),
        "loss": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        "metrics": train_metrics
    }
    vgg_model_base = vggface2_model(dropout_rate=FLAGS.dropout_rate)
    vgg_model_base.compile(**model_compile_args)
    if FLAGS.base_model_ckpt:  # load from the manually-specified checkpoint
        print("[INFO] loading from specified checkpoint {}".format(
            FLAGS.base_model_ckpt
        ))
        vgg_model_base.load_weights(filepath=FLAGS.base_model_ckpt)
    else:  # Load from the default checkpoint path
        vgg_model_base.load_weights(filepath=make_ckpt_filepath(
            FLAGS, is_adversarial=False))

    # Adversarial model
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
    if FLAGS.adv_model_ckpt:
        adv_model.load_weights(FLAGS.adv_model_ckpt)
    else:
        adv_model.load_weights(filepath=make_ckpt_filepath(FLAGS, is_adversarial=True))
    # List to store the results of the experiment
    metrics_list = list()

    for adv_step_size_to_eval in (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.25):
        print("adv_step_size_to_eval %f" % adv_step_size_to_eval)
        reference_model = make_compiled_reference_model(
            model_base=vgg_model_base,
            adv_config=nsl.configs.make_adv_reg_config(
                multiplier=FLAGS.adv_multiplier,
                adv_step_size=adv_step_size_to_eval,
                adv_grad_norm=FLAGS.adv_grad_norm
            ),
            model_compile_args=model_compile_args)
        models_to_eval = {
            'base': vgg_model_base,
            'adv-regularized': adv_model.base_model
        }
        for attr_val, dset in attr_dsets.items():
            # Perturb the images and get the metrics
            perturbed_images, labels, predictions, metrics = perturb_and_evaluate(
                dset.dataset, models_to_eval, reference_model)
            # Add other identifiers to the metrics dict and save to metrics_list
            metrics['attr_val'] = attr_val
            metrics['attr_name'] = FLAGS.slice_attribute_name
            metrics['uid'] = make_model_uid(FLAGS, is_adversarial=True)
            metrics['adv_step_size'] = adv_step_size_to_eval
            metrics_list.append(metrics)
            # Write the results for 3 batches to a file for inspection.
            adv_image_basename = \
                "./debug/adv-examples-{uid}-{attr}-{val}-step{ss}".format(
                    uid=make_model_uid(FLAGS, is_adversarial=True),
                    attr=FLAGS.slice_attribute_name,
                    val=attr_val,
                    ss=adv_step_size_to_eval
                )

            show_adversarial_resuts(n_batches=3,
                                    perturbed_images=perturbed_images,
                                    labels=labels,
                                    predictions=predictions,
                                    fp_basename=adv_image_basename,
                                    batch_size=FLAGS.batch_size)

    metrics_fp = "./metrics/{}-{}-adversarial-analysis.csv".format(
        make_model_uid(FLAGS, is_adversarial=True), FLAGS.slice_attribute_name)
    print("[INFO] writing results to {}".format(metrics_fp))
    pd.DataFrame(metrics_list).to_csv(metrics_fp)


if __name__ == "__main__":
    print("running")
    app.run(main)
