from absl import flags
import json

from dro.utils.lfw import extract_dataset_making_parameters
from dro.utils.training_utils import get_model_img_shape_from_flags

FLAGS = flags.FLAGS


def get_attack_params(flags):
    return json.loads(flags.attack_params)


def define_training_flags():
    """Defines the flags used for model training."""
    flags.DEFINE_integer("batch_size", 16, "batch size")
    flags.DEFINE_integer("epochs", 250, "the number of training epochs")
    flags.DEFINE_string("train_dir", None, "directory containing the training images")
    flags.DEFINE_string("test_dir", None, "directory containing the test images")
    flags.DEFINE_string("ckpt_dir", "./training-logs",
                        "directory to save/load checkpoints "
                        "from")
    flags.DEFINE_float("learning_rate", 0.01, "learning rate to use")
    flags.DEFINE_float("dropout_rate", 0.8,
                       "dropout rate to use in fully-connected layers")
    flags.DEFINE_bool("train_adversarial", True, "whether to train an adversarial model.")
    flags.DEFINE_bool("train_base", True, "whether to train the base (non-adversarial) "
                                          "model. Otherwise it will be loaded from the "
                                          "default or provided checkpoint.")
    flags.DEFINE_string("base_model_ckpt", None,
                        "optional manually-specified checkpoint to use to load the base "
                        "model.")
    flags.DEFINE_string("adv_model_ckpt", None,
                        "optional manually-specified checkpoint to use to load the "
                        "adversarial model.")
    flags.DEFINE_float("val_frac", 0.1, "proportion of data to use for validation")
    flags.DEFINE_string("label_name", None,
                        "name of the prediction label (e.g. sunglasses, mouth_open)",
                        )
    flags.DEFINE_string("experiment_uid", None,
                        "Optional string identifier to be used to "
                        "uniquely identify this experiment.")
    flags.DEFINE_string("precomputed_batches_fp", None,
                        "Optional filepath to a set of precomputed batches; if provided, "
                        "these will be used for training instead of randomly-shuffled "
                        "batches of training data.")
    flags.DEFINE_string("anno_dir", None,
                        "path to the directory containing the vggface annotation files.")
    flags.mark_flag_as_required("label_name")
    flags.DEFINE_bool("debug", False,
                      "whether to run in debug mode (super short iterations to check for "
                      "bugs)")
    flags.DEFINE_bool("use_dbs", False,
                      "whether diverse batch sampling was used; if this is "
                      "set to True, batches will be read from the "
                      "precomputed_batches_fp.")
    flags.DEFINE_enum("model_type", "vggface2", ["vggface2", "facenet"],
                      "the type of model to use")
    flags.DEFINE_string("model_activation", "softmax", "the activation to use in the "
                                                       "final layer. Cleverhans "
                                                       "requires the use of a softmax "
                                                       "layer.")
    flags.DEFINE_string("metrics_dir", "./metrics", "directory to write metrics to")
    return


def define_adv_training_flags(cleverhans: bool):
    """Defines the flags used for adversarial training."""

    flags.DEFINE_string('attack', 'FastGradientMethod',
                        'the cleverhans attack class name to use.')
    flags.DEFINE_string(
        'attack_params',
        None,
        """JSON string representing the dictionary of arguments to pass to the attack 
        constructor. 
        
        Example for usage with cleverhans:
        "{\"eps\": 0.025}" 
        Example for usage with nsl.AdversarialRegularization:
        "{\"adv_multiplier\": 0.2, \"adv_step_size\": 0.025, \"adv_grad_norm\": 
        \"infinity\"}"
        """)
    if cleverhans:
        flags.DEFINE_float(
            "adv_multiplier", 0.2,
            "adversarial multiplier. This is defined as a separate flag for cleverhans, "
            "since it is not a parameter provided to the Attack.generate() method. "
            "The loss will be computed as:"
            "loss_on_clean_inputs + adv_multiplier * loss_on_perturbed_inputs")
    return


def define_embeddings_flags():
    """Defines the flags used for generating embeddings."""
    flags.DEFINE_string("img_dir", None, "directory containing the images")
    flags.DEFINE_string("out_dir", "./embeddings", "directory to dump the embeddings and "
                                                   "similarity to")
    flags.DEFINE_bool("similarity", False,
                      "whether or not to write the similarity matrix; "
                      "this can be huge for large datasets and it may "
                      "be easier to just store the embeddings and "
                      "compute similarity later.")
    flags.DEFINE_integer("batch_size", 16, "batch size to use for inference")
    return


def define_eval_flags():
    """Defines the flags used for evaluation."""
    flags.DEFINE_string("anno_fp", None, "path to annotations file for evaluation.")
    flags.DEFINE_string("slice_attribute_name", None,
                        "attribute name to use from annotations.")
    flags.mark_flag_as_required("slice_attribute_name")
    flags.DEFINE_float("confidence_threshold", 0.5,
                       "only predictions with absolute value "
                       ">= this threshold are used ("
                       "predictions are centered around zero) "
                       "in order to ensure high-quality labels.")
    flags.DEFINE_bool("equalize_subgroup_n", False,
                      "If true, make the subsamples for the majority and minority "
                      "groups the same size by downsampling to the smaller subsample "
                      "size.")
    return


def define_dbs_flags():
    """Define flags for use with diverse batch sampling."""
    flags.DEFINE_string("embeddings_fp", None, "path to the embeddings.csv file.")
    flags.DEFINE_integer("random_state", 95120,
                         "Random seed to use in generating batches.")
    flags.DEFINE_bool("use_precomputed_eigs", False,
                      "whether to use a set of precomputed "
                      "eigenvalues/vectors.")
    flags.DEFINE_string("eigen_vals_fp", None, "path to eigenvalues; if "
                                               "use_recomputed_eigs is True, this is "
                                               "where "
                                               "the eigenvalues will be loaded from.")
    flags.DEFINE_string("eigen_vecs_fp", None, "path to eigenvectors; if "
                                               "use_recomputed_eigs is True, this is "
                                               "where "
                                               "the eigenvectors will be loaded from.")
    flags.DEFINE_string("batches_fp", None,
                        "Path to write batches array (should be .npz)")
    flags.DEFINE_integer("n_batches", 100, "Number of batches to generate.")
    flags.DEFINE_integer("batch_size", 16, "Size of batches to generate")
    flags.mark_flags_as_required(["embeddings_fp", "eigen_vals_fp", "eigen_vecs_fp"])
    return


def define_verification_analysis_flags():
    """Define flags for use with the adversarial transfer attack experiments."""
    flags.DEFINE_spaceseplist("label_names", None, "the list of labels to evaluate")
    flags.DEFINE_spaceseplist("slice_attribute_names", None, "the list of slice "
                                                             "attributes to evaluate")
    return


def extract_dataset_making_parameters_from_flags(flags, write_samples: bool, test=True):
    """A helper function to extract a dict of parameters from flags, which can then be
    unpacked to make_pos_and_neg_attr_datasets."""
    if test:
        data_dir = flags.test_dir
    else:
        data_dir = flags.train_dir
    make_datasets_parameters = extract_dataset_making_parameters(
        anno_fp=flags.anno_fp, data_dir=data_dir, label_name=flags.label_name,
        slice_attribute_name=flags.slice_attribute_name,
        confidence_threshold=flags.confidence_threshold,
        img_shape=get_model_img_shape_from_flags(flags),
        batch_size=flags.batch_size, write_samples=write_samples,
        equalize_subgroup_n=flags.equalize_subgroup_n
    )
    return make_datasets_parameters
