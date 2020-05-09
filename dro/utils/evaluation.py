def extract_dataset_making_parameters(
        anno_fp: str, data_dir: str, label_name: str,
        slice_attribute_name: str, confidence_threshold: float, img_shape: tuple,
        batch_size: int, write_samples: bool):
    make_datasets_parameters = {
        "anno_fp": anno_fp,
        "data_dir": data_dir,
        "label_name": label_name,
        "slice_attribute_name": slice_attribute_name,
        "confidence_threshold": confidence_threshold,
        "img_shape": img_shape,
        "batch_size": batch_size,
        "write_samples": write_samples
    }
    return make_datasets_parameters


ADV_STEP_SIZE_GRID = (0.005, 0.01, 0.025, 0.05, 0.1, 0.125)
