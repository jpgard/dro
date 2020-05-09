"""
A script to evaluate the impact of adversarial examples drawn from classification
models, on a set of face verification models. This is a form of transfer attack.

export SS=0.025
export EPOCHS=40
export ATTACK="FastGradientMethod"

export DIR="/projects/grail/jpgard/lfw"

# set the gpu
export GPU_ID="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MODEL_TYPE="facenet"

python3 scripts/verification_analysis.py \
    --label_name "" \
    --anno_fp ${DIR}/lfw_attributes_cleaned.txt \
    --test_dir ${DIR}/lfw-deepfunneled \
    --slice_attribute_name "" \
    --attack FastGradientMethod \
    --attack_params "{\"eps\": 0.025, \"clip_min\": null, \"clip_max\": null}" \
    --adv_multiplier 0.2 \
    --epochs 40 \
    --metrics_dir ./metrics \
    --model_type "vggface2" \
    --label_names "Mouth_Open Sunglasses Male Eyeglasses" \
    --slice_attribute_names "Asian Black Male Senior"

"""
import os
from absl import flags, app

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp

from dro.utils.lfw import make_pos_and_neg_attr_datasets, \
    extract_dataset_making_parameters
from keras_vggface.vggface import VGGFace
from dro.utils.flags import define_training_flags, define_eval_flags, \
    define_adv_training_flags, define_verification_analysis_flags
from dro.utils.training_utils import get_model_from_flags, get_model_img_shape_from_flags, \
    get_model_compile_args
from dro.utils.cleverhans import get_attack, \
    attack_params_from_flags
from dro.utils.reports import Report
from dro.utils.training_utils import make_ckpt_filepath, make_model_uid, make_logdir

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS

define_training_flags()
define_eval_flags()
define_adv_training_flags(cleverhans=True)
define_verification_analysis_flags()


def embedding_analysis(dset_generator, model, sess, attack):
    """Utility function to get the embeddings on clean and perturbed inputs,
    using generate_attack_op."""
    embeddings_clean = list()
    embeddings_adv = list()
    batch_index = 0

    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
    attack_params = attack_params_from_flags(FLAGS)
    # TODO(jpgard): handle the case of randomized attacks!
    generate_attack_op = attack.generate(x, **attack_params)

    for batch_x, _ in dset_generator:
        # Get the adversarial x
        x_adv = sess.run(generate_attack_op, feed_dict={x: batch_x})
        # Get the embeddings
        x_embed = model.predict(batch_x)
        x_adv_embed_base = model.predict(x_adv)
        embeddings_clean.append(x_embed)
        embeddings_adv.append(x_adv_embed_base)
        batch_index += 1
        if FLAGS.debug:
            print("[INFO] running in debug mode")
            break
    print("processed {} batches".format(batch_index))
    return np.concatenate(embeddings_clean), np.concatenate(embeddings_adv)


def compute_perturbation_l2_distance(embeddings_clean, embeddings_adv):
    shift_matrix = embeddings_clean - embeddings_adv
    perturbation_distance = np.linalg.norm(shift_matrix, ord=2, axis=1)
    # Expect array of size (n,) where each element is the L_p distance for that
    # observation
    # between the embeddings of the clean and the perturbed image.
    return perturbation_distance


def compute_perturbation_cosine_distance(embeddings_clean, embeddings_adv):
    assert embeddings_clean.shape == embeddings_adv.shape, \
        "embeddings arrays must have same shape"
    n = embeddings_clean.shape[0]
    cosine_distances = [cosine(embeddings_clean[i, :], embeddings_adv[i, :]) for i in
                        range(n)]
    return np.array(cosine_distances)


def plot_histograms(data_group_0: np.array, data_group_1: np.array,
                    slice_attribute_name, attack_name, label_name, filename,
                    metric_name, nbins=50):
    test_result = ks_2samp(data_group_0, data_group_1)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    axs[0].hist(data_group_0, nbins)
    axs[0].axvline(np.median(data_group_0), color="red", linestyle="--")
    axs[0].set_title("%s == 0" % slice_attribute_name)
    axs[1].hist(data_group_1, nbins)
    axs[1].axvline(np.median(data_group_1), color="red", linestyle="--")
    axs[1].set_title("%s == 1" % slice_attribute_name)
    fig.suptitle(
        "{metric} Embedding Shifts Under Adversarial Perturbation\n"
        "Attack {attack} With Model for {label}\nK-S Test Statistic {ks} (p={p})".format(
            metric=metric_name,
            attack=attack_name,
            label=label_name,
            ks=round(test_result.statistic, 1),
            p=round(test_result.pvalue, 4)))
    fig.tight_layout(rect=[0, 0.03, 1, 0.8])  # [left, bottom, right, top]
    fig.set_size_inches(6, 4)
    print("[INFO] saving figure to {}".format(filename))
    fig.savefig(filename, dpi=100)
    plt.cla()
    return


def main(argv):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    keras.backend.set_session(sess)
    # Set the learning phase to False, following the issue here:
    # https://github.com/tensorflow/cleverhans/issues/1052
    K.set_learning_phase(False)

    report = Report(FLAGS)

    img_shape = get_model_img_shape_from_flags(FLAGS)

    # Load the pretrained embedding model; this is used to extract the
    # face embeddings and compute the representation disparities.
    model = VGGFace(include_top=False, input_shape=(img_shape[0], img_shape[1], 3),
                    pooling='avg')

    model_compile_args = get_model_compile_args(
        FLAGS,
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics_to_add=None)

    for label_name in FLAGS.label_names:

        # Load the trained base classification model and its weights;
        # we need this model in order to compute the adversarial inputs
        model_base = get_model_from_flags(FLAGS)
        model_base.compile(**model_compile_args)
        uid = make_model_uid(label_name=label_name, model_type=FLAGS.model_type,
                             batch_size=FLAGS.batch_size, epochs=FLAGS.epochs,
                             learning_rate=FLAGS.learning_rate,
                             dropout_rate=FLAGS.dropout_rate, attack=FLAGS.attack,
                             attack_params=FLAGS.attack_params,
                             adv_multiplier=FLAGS.adv_multiplier,
                             experiment_uid=FLAGS.experiment_uid,
                             use_dbs=FLAGS.use_dbs, is_adversarial=False)
        ckpt_filepath = make_ckpt_filepath(".h5", uid, make_logdir(FLAGS, uid))
        print("[INFO] loading weights from {}".format(ckpt_filepath))
        model_base.load_weights(ckpt_filepath)
        # Get the attack.
        attack = get_attack(FLAGS.attack, model_base, sess)

        for slice_attribute_name in FLAGS.slice_attribute_names:

            # Make the datasets for both values of the binary attribute
            dataset_params = extract_dataset_making_parameters(
                anno_fp=FLAGS.anno_fp, data_dir=FLAGS.test_dir, label_name=label_name,
                slice_attribute_name=slice_attribute_name,
                confidence_threshold=FLAGS.confidence_threshold, img_shape=img_shape,
                batch_size=FLAGS.batch_size, write_samples=False
            )
            eval_dsets = make_pos_and_neg_attr_datasets(**dataset_params)

            cosine_distances = {}
            l2_distances = {}

            for attr_val in ("0", "1"):
                eval_dset_numpy = tfds.as_numpy(eval_dsets[attr_val].dataset)

                embeddings_clean, embeddings_adv = embedding_analysis(
                    eval_dset_numpy, model, sess, attack)

                # Compare the L2 distances.
                perturbation_l2_dist = compute_perturbation_l2_distance(
                    embeddings_clean, embeddings_adv)
                l2_distances[attr_val] = perturbation_l2_dist
                median_l2_perturbation_dist = np.median(perturbation_l2_dist)

                # Compare the cosine distances.
                perturbation_cosine_dist = compute_perturbation_cosine_distance(
                    embeddings_clean, embeddings_adv)
                cosine_distances[attr_val] = perturbation_cosine_dist
                median_cosine_perturbation_dist = np.median(perturbation_cosine_dist)

                report.add_result(
                    {"label_name": label_name,
                     "slice_attribute_name": slice_attribute_name,
                     "attr_val": attr_val,
                     "metric": "median_l2_distance_after_perturbation",
                     "value": median_l2_perturbation_dist,
                     }
                )

                report.add_result(
                    {"label_name": label_name,
                     "slice_attribute_name": slice_attribute_name,
                     "attr_val": attr_val,
                     "metric": "median_cosine_distance_after_perturbation",
                     "value": median_cosine_perturbation_dist,
                     }
                )
            # Run a KS test for the cosine distance and the L2 distances

            # Add results for the KS test p-values
            report.add_result(
                {"label_name": label_name,
                 "slice_attribute_name": slice_attribute_name,
                 "attr_val": np.nan,
                 "metric": "ks_test_pval_cosine_dist",
                 "value": ks_2samp(cosine_distances["0"], cosine_distances["1"]).pvalue,
                 }
            )

            report.add_result(
                {"label_name": label_name,
                 "slice_attribute_name": slice_attribute_name,
                 "attr_val": np.nan,
                 "metric": "ks_test_pval_l2_dist",
                 "value": ks_2samp(l2_distances["0"], l2_distances["1"]).pvalue,
                 }
            )
            l2_filename = "-".join([label_name, slice_attribute_name, "l2"]) + ".png"
            plot_histograms(l2_distances["0"], l2_distances["1"],
                            slice_attribute_name=slice_attribute_name,
                            attack_name=FLAGS.attack, label_name=label_name,
                            metric_name="L2 Distance",
                            filename=os.path.join("./img", l2_filename))

            cosine_filename = "-".join([label_name, slice_attribute_name, "cos"]) + ".png"
            plot_histograms(cosine_distances["0"], cosine_distances["1"],
                            slice_attribute_name=slice_attribute_name,
                            attack_name=FLAGS.attack, label_name=label_name,
                            metric_name="Cosine Distance",
                            filename=os.path.join("./img", cosine_filename))
        # Delete the model before proceeding to the next one, to ensure the weights do
        # not carry over.
        del model_base

    report.to_csv(FLAGS.metrics_dir)


if __name__ == "__main__":
    app.run(main)
