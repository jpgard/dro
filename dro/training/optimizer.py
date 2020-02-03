import tensorflow as tf

def optimizer_from_flags(flags):
    if flags.loss_name == "sgd":
        return tf.keras.optimizers.SGD()
    elif flags.loss_name == "adam":
        return tf.keras.optimizers.Adam()
    else:
        raise NotImplementedError
