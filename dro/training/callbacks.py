from dro.keys import TRAIN_MODE
from dro.training.training_utils import make_model_uid_from_flags, make_csv_name, \
    make_logdir, make_ckpt_filepath_from_flags


def make_csv_callback(flags, is_adversarial: bool):
    callback_uid = make_model_uid_from_flags(flags, is_adversarial=is_adversarial)
    csv_fp = make_csv_name(callback_uid, mode=TRAIN_MODE)
    return CSVLogger(csv_fp)


def make_callbacks(flags, is_adversarial: bool):
    """Create the callbacks for training, including properly naming files."""
    callback_uid = make_model_uid_from_flags(flags, is_adversarial=is_adversarial)
    logdir = make_logdir(flags, callback_uid)
    tensorboard_callback = TensorBoard(
        log_dir=logdir,
        batch_size=flags.batch_size,
        write_graph=True,
        write_grads=True,
        update_freq='epoch')
    csv_callback = make_csv_callback(flags, is_adversarial)
    ckpt_fp = make_ckpt_filepath_from_flags(flags, is_adversarial=is_adversarial)
    ckpt_callback = ModelCheckpoint(ckpt_fp,
                                    monitor='val_loss',
                                    save_best_only=True,
                                    save_weights_only=False,
                                    save_freq='epoch',
                                    verbose=1,
                                    mode='auto')
    return [tensorboard_callback, csv_callback, ckpt_callback]