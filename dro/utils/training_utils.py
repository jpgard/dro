from itertools import islice

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_batch(dataset_iterator, batch_size):
    slice = tuple(islice(dataset_iterator, batch_size))
    batch_x = np.stack([i[0] for i in slice], axis=0)
    batch_y = np.stack([i[1] for i in slice], axis=0)
    return batch_x, batch_y


def prepare_dataset_for_training(ds, cache=True, shuffle_buffer_size=1000,
                         repeat_forever=False, batch_size=None,
                                 prefetch_buffer_size=AUTOTUNE):
    """Shuffle, repeat, batch, and prefetch the dataset."""
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.

    # If the data doesn't fit in memory see the example usage at the
    # bottom of the page here:
    # https: // www.tensorflow.org / tutorials / load_data / images
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    if repeat_forever:
        ds = ds.repeat()
    if batch_size:
        ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=prefetch_buffer_size)
    return ds