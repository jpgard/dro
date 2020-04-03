import time

from dro.datasets import ImageDataset
from dro.keys import FILENAME_COLNAME
from dro.utils.lfw import get_annotated_data_df, LABEL_COLNAME, ATTR_COLNAME, apply_thresh
from dro.utils.training_utils import pred_to_binary
from dro.utils.viz import show_batch


def make_pos_and_neg_attr_datasets(flags):
    # build a labeled dataset from the files
    annotated_files = get_annotated_data_df(anno_fp=flags.anno_fp,
                                            test_dir=flags.test_dir)
    assert len(annotated_files) > 0, "no files detected"

    # Create a DataFrame with columns for (filename, label, slice_attribute); the columns
    # need to be renamed to generic LABEL_COLNAME and ATTR_COLNAME in order to allow
    # for cases where label and attribute names are the same (e.g. slicing 'Male'
    # prediction by 'Male' attribute).

    dset_df = annotated_files.reset_index()[
        [FILENAME_COLNAME, flags.label_name, flags.slice_attribute_name]]
    dset_df.columns = [FILENAME_COLNAME, LABEL_COLNAME, ATTR_COLNAME]

    # Apply thresholding. We want observations which have absolute value greater than some
    # threshold (predictions close to zero have low confidence).

    dset_df = apply_thresh(dset_df, LABEL_COLNAME,
                           flags.confidence_threshold)
    dset_df = apply_thresh(dset_df, ATTR_COLNAME,
                           flags.confidence_threshold)

    dset_df[LABEL_COLNAME] = dset_df[LABEL_COLNAME].apply(pred_to_binary)
    dset_df[ATTR_COLNAME] = dset_df[ATTR_COLNAME].apply(
        pred_to_binary)

    # Break the input dataset into separate tf.Datasets based on the value of the slice
    # attribute.

    # Create and preprocess the dataset of examples where ATTR_COLNAME == 1
    preprocessing_kwargs = {"shuffle": False, "repeat_forever": False, "batch_size":
        flags.batch_size}
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
                   flags.slice_attribute_name, flags.label_name, int(time.time()))
               )
    image_batch, label_batch = next(iter(dset_attr_neg.dataset))
    show_batch(image_batch.numpy(), label_batch.numpy(),
               fp="./debug/sample_batch_attr{}0-label{}-{}.png".format(
                   flags.slice_attribute_name, flags.label_name, int(time.time()))
               )
    return {"1": dset_attr_pos, "0": dset_attr_neg}


ADV_STEP_SIZE_GRID = (0.005, 0.01, 0.025, 0.05, 0.1, 0.125, 0.2, 0.25)