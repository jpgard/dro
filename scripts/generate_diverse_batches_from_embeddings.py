"""
Script to generate DPP-based batches from a set of precomputed embeddings.

Writes the results to a file, where each line in the file is a batch of the specified
batch_size, and each element is an index into embeddings.csv.

usage:
export DIR="./embeddings"
python3 scripts/generate_diverse_batches_from_embeddings.py \
--embeddings_fp $DIR/embedding.csv \
--eigen_vals_fp $DIR/eigen_vals.npz \
--eigen_vecs_fp $DIR/eigen_vecs.npz \
--batches_fp $DIR/batches.npz \
--use_precomputed_eigs --n_batches 100 --batch_size 16
"""

import time

from absl import app, flags
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dppy.finite_dpps import FiniteDPP

FLAGS = flags.FLAGS

# TODO(jpgard): just load all files from a shared directory. Also, the batches file
#  should have a uid which includes the parameters in its name to uniquely identify it.

flags.DEFINE_string("embeddings_fp", None, "path to the embeddings.csv file.")
flags.DEFINE_integer("random_state", 95120, "Random seed to use in generating batches.")
flags.DEFINE_bool("use_precomputed_eigs", False, "whether to use a set of precomputed "
                                                 "eigenvalues/vectors.")
flags.DEFINE_string("eigen_vals_fp", None, "path to eigenvalues; if "
                                           "use_recomputed_eigs is True, this is where "
                                           "the eigenvalues will be loaded from.")
flags.DEFINE_string("eigen_vecs_fp", None, "path to eigenvectors; if "
                                           "use_recomputed_eigs is True, this is where "
                                           "the eigenvectors will be loaded from.")
flags.DEFINE_string("batches_fp", None, "Path to write batches array (should be .npz)")
flags.DEFINE_integer("n_batches", 100, "Number of batches to generate.")
flags.DEFINE_integer("batch_size", 16, "Size of batches to generate")
flags.mark_flags_as_required(["embeddings_fp", "eigen_vals_fp", "eigen_vecs_fp"])


def load_or_compute_eigs(similarities: np.ndarray, use_precomputed_eigs: bool,
                         eigen_vals_fp: str, eigen_vecs_fp: str):
    """
    Either load or compute eigenvalues/eigenvectors.

    :param similarities: symmetric, real-valued similarity matrix.
    :param use_precomputed_eigs: indicator for whether to load precomputed eigs.
    :param eigen_vecs_fp: filepath to either write or load the eigenvalues.
    :param eigen_vecs_fp: filepath to either write or load the eigenvectors.
    """
    assert similarities.max() == 1, "Unexpected values > 1 in similarity matrix."
    if use_precomputed_eigs:
        print("[INFO] loading eigenvalues from {},{}".format(eigen_vals_fp,
                                                             eigen_vecs_fp))
        eigen_vals = np.load(eigen_vals_fp)["eigen_vals"]
        eigen_vecs = np.load(eigen_vecs_fp)["eigen_vecs"]
    else:
        # Compute the eigendecomposition of the similarity matrix; this is
        # expensive and should be cached. Takes ~36mins for 26,000-dimensional matrix,
        # and the resulting file is ~5GB.
        start = time.time()
        print("[INFO] computing eigendecomposition of similarity matrix")
        eigen_vals, eigen_vecs = np.linalg.eigh(similarities)
        print("[INFO] computed eigendecomposition of similarity matrix in {} sec".format(
            int(time.time() - start)
        ))
        np.savez_compressed(eigen_vals_fp, eigen_vals=eigen_vals)
        np.savez_compressed(eigen_vecs_fp, eigen_vecs=eigen_vecs)
    return eigen_vals, eigen_vecs


def batch_indices_to_values(list_of_samples: list, index_vals: list) -> list:
    """Convert a nested list of batch indices to a list of values with the same
    structure."""
    batch_values = [[index_vals[x] for x in batch] for batch in list_of_samples]
    return batch_values


def main(argv):
    embeddings = pd.read_csv(FLAGS.embeddings_fp, index_col=0)
    print("[INFO] loaded embeddings of shape {}".format(embeddings.shape))
    similarities = cosine_similarity(embeddings.values)
    # sets similarities > 1 to 1 to correct for numerical imprecision in similarity
    np.fill_diagonal(similarities, 1)
    # Get the eigenvalues for the DPP kernel.
    eigen_vals, eigen_vecs = load_or_compute_eigs(similarities,
                                                  FLAGS.use_precomputed_eigs,
                                                  FLAGS.eigen_vals_fp,
                                                  FLAGS.eigen_vecs_fp)
    # Generate the batches.
    rng = np.random.RandomState(FLAGS.random_state)
    start = time.time()
    DPP = FiniteDPP(kernel_type='likelihood', L_eig_dec=(eigen_vals, eigen_vecs))
    for _ in range(FLAGS.n_batches):
        DPP.sample_exact_k_dpp(size=FLAGS.batch_size, random_state=rng)
    end = time.time()
    print("generated %s samples in %s secs" % (FLAGS.n_batches, end - start))

    # Transform batches of index numbers into batches of index values and write to file.
    index_vals = embeddings.index.values
    batch_values = batch_indices_to_values(DPP.list_of_samples, index_vals)
    batch_values = np.array(batch_values)
    np.savez_compressed(FLAGS.batches_fp, batch_values)
    return


if __name__ == "__main__":
    app.run(main)
