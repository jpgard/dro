
import tensorflow as tf
from tensorflow import keras


from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent, Noise, Attack
from cleverhans.utils_keras import KerasModelWrapper



def get_attack(flags, wrap: KerasModelWrapper, sess: tf.Session):
    """Load and instantiate the cleverhans object for the specified attack."""
    """Creates an instance of the attack method specified in flags."""
    return globals()[flags.attack](wrap, sess=sess)

def get_adversarial_acc_metric(model: keras.Model, attack: Attack, fgsm_params: dict):
    """Get a callable which can be used to compute the adversarial accuracy during
    training."""
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = attack.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        print(x_adv)
        print(model.input)
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)
    return adv_acc


def get_adversarial_loss(model: keras.Model, attack: Attack,
                         fgsm_params: dict,
                         adv_multiplier: float):
    """Get a callable which can be used to compute the adversarial loss metric during
    Keras model training."""
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = attack.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return cross_ent + adv_multiplier * cross_ent_adv

    return adv_loss