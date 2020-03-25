"""
Utilities for diverse batch sampling.
"""
import time

import numpy as np


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