import time

from dro.datasets import ImageDataset
from dro.keys import FILENAME_COLNAME
from dro.utils.lfw import get_annotated_data_df, LABEL_COLNAME, ATTR_COLNAME, apply_thresh
from dro.utils.training_utils import pred_to_binary, get_model_img_shape_from_flags
from dro.utils.viz import show_batch


def extract_dataset_making_parameters(flags, write_samples: bool):
    """A helper function to extract a dict of parameters from flags, which can then be 
    unpacked to make_pos_and_neg_attr_datasets."""
    make_datasets_parameters = {
        "anno_fp": flags.anno_fp,
        "test_dir": flags.test_dir,
        "label_name": flags.label_name,
        "slice_attribute_name": flags.slice_attribute_name,
        "confidence_threshold": flags.confidence_threshold,
        "img_shape": get_model_img_shape_from_flags(flags),
        "batch_size": flags.batch_size,
        "write_samples": write_samples
    }
    return make_datasets_parameters


def make_pos_and_neg_attr_datasets(anno_fp, test_dir, label_name, 
                                   slice_attribute_name,
                                   confidence_threshold, img_shape, batch_size, 
                                   write_samples=True
                                   ):
    """Create a dict of datasets where the keys correspond to the binary attribute,
    and the values are tf.data.Datasets of the (image, label) tuples."""
    # build a labeled dataset from the files
    annotated_files = get_annotated_data_df(anno_fp=anno_fp,
                                            test_dir=test_dir)
    assert len(annotated_files) > 0, "no files detected"

    # Create a DataFrame with columns for (filename, label, slice_attribute); the columns
    # need to be renamed to generic LABEL_COLNAME and ATTR_COLNAME in order to allow
    # for cases where label and attribute names are the same (e.g. slicing 'Male'
    # prediction by 'Male' attribute).

    dset_df = annotated_files.reset_index()[
        [FILENAME_COLNAME, label_name, slice_attribute_name]]
    dset_df.columns = [FILENAME_COLNAME, LABEL_COLNAME, ATTR_COLNAME]

    # Apply thresholding. We want observations which have absolute value greater than some
    # threshold (predictions close to zero have low confidence).

    dset_df = apply_thresh(dset_df, LABEL_COLNAME,
                           confidence_threshold)
    dset_df = apply_thresh(dset_df, ATTR_COLNAME,
                           confidence_threshold)

    dset_df[LABEL_COLNAME] = dset_df[LABEL_COLNAME].apply(pred_to_binary)
    dset_df[ATTR_COLNAME] = dset_df[ATTR_COLNAME].apply(
        pred_to_binary)

    # Break the input dataset into separate tf.Datasets based on the value of the slice
    # attribute.

    # Create and preprocess the dataset of examples where ATTR_COLNAME == 1
    preprocessing_kwargs = {"shuffle": False, "repeat_forever": False, "batch_size":
        batch_size}

    dset_attr_pos = ImageDataset(img_shape)
    dset_attr_pos.from_dataframe(dset_df[dset_df[ATTR_COLNAME] == 1],
                                 label_name=LABEL_COLNAME)
    dset_attr_pos.preprocess(**preprocessing_kwargs)

    # Create and process the dataset of examples where ATTR_COLNAME == 1
    dset_attr_neg = ImageDataset(img_shape)
    dset_attr_neg.from_dataframe(dset_df[dset_df[ATTR_COLNAME] == 0],
                                 label_name=LABEL_COLNAME)
    dset_attr_neg.preprocess(**preprocessing_kwargs)

    if write_samples:
        print("[INFO] writing sample batches; this will fail if eager execution is "
              "disabled")
        image_batch, label_batch = next(iter(dset_attr_pos.dataset))
        show_batch(image_batch.numpy(), label_batch.numpy(),
                   fp="./debug/sample_batch_attr{}1-label{}-{}.png".format(
                       slice_attribute_name, label_name, int(time.time()))
                   )
        image_batch, label_batch = next(iter(dset_attr_neg.dataset))
        show_batch(image_batch.numpy(), label_batch.numpy(),
                   fp="./debug/sample_batch_attr{}0-label{}-{}.png".format(
                       slice_attribute_name, label_name, int(time.time()))
                   )
    return {"1": dset_attr_pos, "0": dset_attr_neg}


ADV_STEP_SIZE_GRID = (0.005, 0.01, 0.025, 0.05, 0.1, 0.125)