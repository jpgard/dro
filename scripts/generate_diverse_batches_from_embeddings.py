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

from dro.datasets.dbs import load_or_compute_eigs, batch_indices_to_values
from dro.utils.flags import define_dbs_flags

FLAGS = flags.FLAGS

# TODO(jpgard): just load all files from a shared directory. Also, the batches file
#  should have a uid which includes the parameters in its name to uniquely identify it.

define_dbs_flags()


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
    print("[INFO] writing output to {}".format(FLAGS.batches_fp))
    np.savez_compressed(FLAGS.batches_fp, batch_values)
    return


if __name__ == "__main__":
    app.run(main)
