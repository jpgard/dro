from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
from six.moves import xrange
import tensorflow as tf
import warnings
import logging

from cleverhans import utils_tf
from cleverhans import utils

_logger = utils.create_logger("cleverhans-extensions.attacks.tf")



def fgm_distributional(x, preds, y=None, eps=0.3, point_ord=np.inf, wasserstein_ord=2,
        clip_min=None, clip_max=None,
        targeted=False):
    """
    TensorFlow implementation of the Fast Gradient Method (Distributionally Robust variant).
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param point_ord: (optional) Order of the norm to compare points (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param wasserstein_ord: (optional) Order of the norm to define Wasserstein distance (mimics Numpy).
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if point_ord == np.inf and wasserstein_ord > 1:
        red_ind = tf.range(1, tf.rank(grad))
        grad_1_norms = tf.reduce_sum(tf.abs(grad),
                                     axis=red_ind,
                                     keep_dims=True)

        # if wasserstein_ord is p, and q is the dual norm, compute q/p = 1/(p-1)
        exponent = 1.0 / (wasserstein_ord - 1.0)
        per_example_weights = tf.pow(grad_1_norms, exponent)
        per_example_weights_normalized = per_example_weights / tf.norm(per_example_weights, ord=wasserstein_ord)

        batch_size = tf.to_float(tf.shape(x)[0])
        eps_batch = eps * per_example_weights_normalized

        #epsilon has to scale with batch size, depending on p
        # eps_batch = eps_batch * tf.pow(batch_size, 1.0 / wasserstein_ord)

        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)

        # adversarial direction
        scaled_grad = eps_batch * normalized_grad

    # if point_ord == np.inf and wasserstein_ord == 2:
    #     red_ind = tf.range(1, tf.rank(grad))
    #     grad_1_norms = tf.reduce_sum(tf.abs(grad),
    #                                  axis=red_ind,
    #                                  keep_dims=True)

    #     norms_normalized = grad_1_norms / tf.norm(grad_1_norms, ord=2)

    #     batch_size = tf.to_float(tf.shape(x)[0])
    #     eps_batch = eps * norms_normalized

    #     #epsilon has to scale with batch size, depending on p
    #     # eps_batch = eps_batch * tf.pow(batch_size, 1.0 / wasserstein_ord)

    #     # Take sign of gradient
    #     normalized_grad = tf.sign(grad)
    #     # The following line should not change the numerical results.
    #     # It applies only because `normalized_grad` is the output of
    #     # a `sign` op, which has zero derivative anyway.
    #     # It should not be applied for the other norms, where the
    #     # perturbation has a non-zero derivative.
    #     normalized_grad = tf.stop_gradient(normalized_grad)

    #     # adversarial direction
    #     scaled_grad = eps_batch * normalized_grad
    elif point_ord == 2 and wasserstein_ord == 2:
        scaled_grad = eps * grad / tf.norm(grad, ord=2)

        red_ind = tf.range(1, tf.rank(scaled_grad))
        eps_batch = tf.reduce_sum(tf.pow(scaled_grad, 2),
                                  axis=red_ind,
                                  keep_dims=False)

        eps_batch = tf.pow(eps_batch, 0.5)

    else:
        raise NotImplementedError("Only L_{2,inf} norm is currently implemented.")

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    epsilons_out = eps_batch
    return adv_x, epsilons_out


def distributional_gradient_step(x, start_x, preds, y=None, eps_iter=0.3, point_ord=np.inf, gamma=0.3,
        clip_min=None, clip_max=None,
        targeted=False):
    """
    TODO: update this description
    TensorFlow implementation of the Fast Gradient Method (Distributionally Robust variant).
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param point_ord: (optional) Order of the norm to compare points (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param wasserstein_ord: (optional) Order of the norm to define Wasserstein distance (mimics Numpy).
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """
    def _cost(v, keep_dims=False):
        red_ind = tf.range(1, tf.rank(v))

        if point_ord == np.inf:
            return tf.reduce_max(tf.abs(v),
                                 axis=red_ind,
                                 keep_dims=keep_dims)
        else:
            powers = tf.pow(v, point_ord)
            power_of_norm = tf.reduce_sum(tf.abs(powers),
                                          axis=red_ind,
                                          keep_dims=keep_dims)
            return power_of_norm

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)
    if targeted:
        loss = -loss

    # add in the cost of moving away
    loss -= gamma * _cost(x - start_x)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)
    grad = tf.stop_gradient(grad)

    # take gradient step
    return x + eps_iter * grad


def compute_gradients(x, preds, y=None, targeted=False):
    """
    TensorFlow implementation of a single gradient step.
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor containing the gradient
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    grad = tf.stop_gradient(grad)

    return grad



def grad(x, preds, y=None, targeted=False):
    """
    Produces Gradients of loss with respect to x
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    return grad

