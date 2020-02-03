import tensorflow as tf

def optimizer_from_flags(flags):
    if flags.optimizer == "sgd":
        return tf.keras.optimizers.SGD()
    elif flags.optimizer == "adam":
        return tf.keras.optimizers.Adam()
    else:
        raise NotImplementedError
