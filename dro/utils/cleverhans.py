import json
import tensorflow as tf
from tensorflow import keras

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent, Noise, \
    Attack, BasicIterativeMethod, MadryEtAl
from dro.utils.attacks import IterativeFastGradientMethod, \
    RandomizedFastGradientMethod, RandomizedFastGradientMethodBeta
from dro.staib.attacks import FastDistributionallyRobustMethod, \
    FrankWolfeDistributionallyRobustMethod
from cleverhans.utils_keras import KerasModelWrapper


def generate_attack(attack: Attack, x: tf.Tensor, attack_params):
    """Helper function to generate an attack."""
    if attack_params is not None:
        x_adv = attack.generate(x, **attack_params)
        if isinstance(x_adv, tuple):
            # Some of the attacks, notably those from Staib et al, return a tuple where
            # the first element is adv_x and the second element is the set of epsilons.
            # In this case, we discard the epsilons.
            x_adv = x_adv[0]
    else:
        x_adv = attack.generate(x)
    return x_adv


def get_attack(flags, model: keras.Model, sess: tf.Session):
    """Load and instantiate the cleverhans object for the specified attack."""
    """Creates an instance of the attack method specified in flags."""
    wrap = KerasModelWrapper(model)
    return globals()[flags.attack](wrap, sess=sess)


def attack_params_from_flags(flags, override_eps_value: float = None):
    """Build a dict of attack params to be passed to Attack.generate().

    :param flags: the flags object.
    :param override_eps_value: optional value to use to override the epsilon in the
    flags; for example, when generating perturbations at an epsilon different from the
    epsilon used to train the model.
    :return: a dict of {parameter_name:parameter_value} pairs to be passed to
    Attack.generate() if flags.attack_params is not None; otherwise return None.
    """
    assert flags.attack_params is not None, "please provide attack_params"
    attack_params = json.loads(flags.attack_params)
    # These are default parameter values we do not want to change.
    if override_eps_value is not None:
        attack_params["eps"] = override_eps_value
    return attack_params


def get_model_compile_args(flags, loss, metrics_to_add: list = None):
    """Builds a dict of the args for compilation containing callables for loss and
    metrics. Accuracy is added by default; other metrics can also be added."""
    metrics = ['accuracy']
    if metrics_to_add:
        metrics.extend(metrics_to_add)
    compile_args = {
        "optimizer": tf.keras.optimizers.SGD(learning_rate=flags.learning_rate),
        "loss": loss,
        "metrics": metrics
    }
    return compile_args


def get_adversarial_acc_metric(model: keras.Model, attack: Attack, attack_params: dict):
    """Get a callable which can be used to compute the adversarial accuracy during
    training."""

    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = generate_attack(attack, model.input, attack_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)
        # Accuracy on the adversarial examples
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
        x_adv = generate_attack(attack, model.input, fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return cross_ent + adv_multiplier * cross_ent_adv

    return adv_loss
