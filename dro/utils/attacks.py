import warnings

import numpy as np
import tensorflow as tf

from cleverhans.attacks import Attack, optimize_linear, clip_eta
from cleverhans import utils
from cleverhans.attacks_tf import SPSAAdam, margin_logit_loss, TensorAdam
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.model import wrapper_warning, wrapper_warning_logits
from cleverhans.compat import reduce_sum, reduce_mean
from cleverhans.compat import reduce_max
from cleverhans.compat import softmax_cross_entropy_with_logits
from cleverhans.utils_tf import clip_eta
from cleverhans import utils_tf


def fgm(model, x, y, ord, eps: float, nb_iter=1, clip_min=0., clip_max=1.):
    """
    Fast gradient method; adapted from
    https://github.com/gongzhitaao/tensorflow-adversarial

    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
    for details.  This implements the revised version since the original FGM
    has label leaking problem (https://arxiv.org/abs/1611.01236).

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param nb_iter: The number of iterations to conduct.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        logits = model.get_logits(x)
        preds_max = reduce_max(logits, 1, keepdims=True)
        y = tf.to_float(tf.equal(logits, preds_max))
        y = tf.stop_gradient(y)
    y = y / reduce_sum(y, 1, keepdims=True)

    eps = tf.abs(eps)

    adv_x = x
    for i in range(nb_iter):
        logits = model.get_logits(adv_x)
        loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
        grad, = tf.gradients(loss, adv_x)
        optimal_perturbation = optimize_linear(grad, eps, ord)
        adv_x = x + optimal_perturbation
        # Following cleverhans, we only support clipping when both clip_min and clip_max
        # are specified.
        if (clip_min is not None) and (clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

        # Clipping perturbation eta to self.ord norm ball
        eta = adv_x - x
        eta = clip_eta(eta, ord, eps)
        adv_x = x + eta

        # Following cleverhans, redo the clipping: subtracting and re-adding eta can
        # add some small numerical error.
        if (clip_min is not None) and (clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

        adv_x = tf.stop_gradient(adv_x)

    return adv_x


def clip(x: tf.Tensor, eps: float):
    """Implements the CLIP function described in Kurakin et al. (sec 2.1)"""
    # These Tensors give the elemnt-wise upper and lower bounds for the matrix
    upper_bound = x + eps
    lower_bound = x - eps
    clipped_upper = tf.minimum(upper_bound, x)
    clipped = tf.maximum(clipped_upper, lower_bound)
    return clipped


class IterativeFastGradientMethod(Attack):

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        """
        Create a FastGradientMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """

        super(IterativeFastGradientMethod, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs = ('eps', 'y', 'y_target', 'clip_min', 'clip_max', 'nb_iter')
        self.structural_kwargs = ['ord', 'sanity_checks']

    def generate(self, x, **kwargs):
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, _nb_classes = self.get_or_guess_labels(x, kwargs)

        return fgm(self.model,
                   x=x,
                   y=labels,
                   ord=self.ord,
                   eps=self.eps,
                   nb_iter=self.nb_iter,
                   clip_min=self.clip_min,
                   clip_max=self.clip_max
                   )

    def parse_params(self,
                     eps=0.3,
                     ord=np.inf,
                     y=None,
                     y_target=None,
                     clip_min=None,
                     clip_max=None,
                     nb_iter=1,
                     sanity_checks=True,
                     **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the true labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param nb_iter: number of iterations to perform.
        :param sanity_checks: bool, if True, include asserts
          (Turn them off to use less runtime / memory or for unit tests that
          intentionally pass strange input)
        """
        # Save attack-specific parameters

        self.eps = eps
        self.ord = ord
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.nb_iter = nb_iter
        self.sanity_checks = sanity_checks

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        if len(kwargs.keys()) > 0:
            warnings.warn("kwargs is unused and will be removed on or after "
                          "2019-04-26.")

        return True
