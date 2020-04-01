from absl import flags

FLAGS = flags.FLAGS

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
    flags.mark_flag_as_required("train_dir")
    flags.DEFINE_bool("debug", False,
                      "whether to run in debug mode (super short iterations to check for "
                      "bugs)")
    flags.DEFINE_bool("use_dbs", False,
                      "whether diverse batch sampling was used; if this is "
                      "set to True, batches will be read from the "
                      "precomputed_batches_fp.")
    return

def define_adv_training_flags():
    """Defines the flags used for adversarial training."""
    flags.DEFINE_float('adv_multiplier', 0.2,
                       " The weight of adversarial loss in the training objective, "
                       "relative "
                       "to the labeled loss. e.g. if this is 0.2, The model minimizes "
                       "(mean_crossentropy_loss + 0.2 * adversarial_regularization) ")
    flags.DEFINE_float('adv_step_size', 0.2, "The magnitude of adversarial perturbation.")
    flags.DEFINE_string('adv_grad_norm', 'infinity',
                        "The norm to measure the magnitude of adversarial perturbation.")
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
    flags.DEFINE_string("metrics_dir", "./metrics", "directory to write metrics to")
    flags.DEFINE_string("slice_attribute_name", None,
                        "attribute name to use from annotations.")
    flags.mark_flag_as_required("slice_attribute_name")
    flags.DEFINE_float("confidence_threshold", 0.5,
                       "only predictions with absolute value "
                       ">= this threshold are used ("
                       "predictions are centered around zero) "
                       "in order to ensure high-quality labels.")
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

