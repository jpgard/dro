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
    --label_name "Mouth_Open" \
    --epochs 1 \
    --anno_fp ${DIR}/lfw_attributes_cleaned.txt \
    --test_dir ${DIR}/lfw-deepfunneled \
    --label_name "Mouth_Open" \
    --slice_attribute_name "Asian" \
    --attack FastGradientMethod \
    --attack_params "{\"eps\": 0.025, \"clip_min\": null, \"clip_max\": null}" \
    --adv_multiplier 0.2 \
    --epochs 1 \
    --metrics_dir ./metrics \
    --model_type "vggface2"

"""
from absl import flags, app

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from scipy.spatial.distance import cosine

from dro.utils.evaluation import make_pos_and_neg_attr_datasets, \
    extract_dataset_making_parameters
from keras_vggface.vggface import VGGFace
from dro.utils.flags import define_training_flags, define_eval_flags, \
    define_adv_training_flags
from dro.utils.training_utils import load_model_weights_from_flags, get_model_from_flags
from dro.utils.cleverhans import get_model_compile_args, get_attack, \
    attack_params_from_flags

# Suppress the annoying tensorflow 1.x deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS

define_training_flags()
define_eval_flags()
define_adv_training_flags(cleverhans=True)


def embedding_analysis(dset_generator, model, sess, attack, n_batches=1):
    """Utility function to get the embeddings on clean and perturbed inputs,
    using generate_attack_op."""
    embeddings_clean = list()
    embeddings_adv = list()
    batch_index = 0

    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
    attack_params = attack_params_from_flags(FLAGS)
    generate_attack_op = attack.generate(x, **attack_params)

    for batch_x, _ in dset_generator:
        print("batch %3s" % batch_index)
        # Get the adversarial x
        x_adv = sess.run(generate_attack_op, feed_dict={x: batch_x})
        # Get the embeddings
        x_embed = model.predict(batch_x)
        x_adv_embed_base = model.predict(x_adv)
        embeddings_clean.append(x_embed)
        embeddings_adv.append(x_adv_embed_base)
        batch_index += 1
        if batch_index - 1 >= n_batches:
            return np.concatenate(embeddings_clean), np.concatenate(embeddings_adv)


def compute_perturbation_l2_distance(embeddings_clean, embeddings_adv):
    shift_matrix = embeddings_clean - embeddings_adv
    perturbation_distance = np.linalg.norm(shift_matrix, ord=2, axis=1)
    # Expect array of size (n,) where each element is the L_p distance for that
    # observation
    # between the embeddings of the clean and the perturbed image.
    return perturbation_distance


def compute_perturbation_cosine_distance(embeddings_clean, embeddings_adv):
    assert embeddings_clean.shape == embeddings_adv.shape, "embeddings arrays must have " \
                                                           "" \
                                                           "" \
                                                           "same shape"
    n = embeddings_clean.shape[0]
    cosine_distances = [cosine(embeddings_clean[i, :], embeddings_adv[i, :]) for i in
                        range(n)]
    return np.array(cosine_distances)


def main(argv):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    keras.backend.set_session(sess)
    # Set the learning phase to False, following the issue here:
    # https://github.com/tensorflow/cleverhans/issues/1052
    K.set_learning_phase(False)

    # Make the datasets for both values of the binary attribute
    dataset_params = extract_dataset_making_parameters(FLAGS, write_samples=False)
    eval_dsets = make_pos_and_neg_attr_datasets(**dataset_params)

    # Load the embedding model; this is used to extract the
    # face embeddings and compute the representation disparities.
    model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # Load the trained base model and its weights;
    # we need the full model with classification layers
    # in order to compute the adversarial inputs
    model_compile_args = get_model_compile_args(
        FLAGS,
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics_to_add=None)
    model_base = get_model_from_flags(FLAGS)
    model_base.compile(**model_compile_args)
    load_model_weights_from_flags(model_base, FLAGS, is_adversarial=False)

    attack = get_attack(FLAGS.attack, model_base, sess)

    # Do the analysis for both values of the slicing attribute

    for attr_val in ("0", "1"):
        eval_dset_numpy = tfds.as_numpy(eval_dsets[attr_val].dataset)

        embeddings_clean, embeddings_adv = embedding_analysis(
            eval_dset_numpy, model, sess, attack)

        # Compare the L2 distances.

        perturbation_l2_dist = compute_perturbation_l2_distance(
            embeddings_clean, embeddings_adv)

        print("mean perturbation distance for attribute group {slice}=={val}: {d}".format(
            slice=FLAGS.slice_attribute_name, val=attr_val,
            d=perturbation_l2_dist.mean()
        ))

        # Compare the cosine distances
        perturbation_cosine_dist = compute_perturbation_cosine_distance(embeddings_clean,
                                                                        embeddings_adv)
        print("mean cosine distance for attribute group {slice}=={val}: {c}".format(
            slice=FLAGS.slice_attribute_name, val=attr_val,
            c=perturbation_cosine_dist.mean()
        ))

    if __name__ == "__main__":
        app.run(main)
