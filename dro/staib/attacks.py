from abc import ABCMeta
import numpy as np
from six.moves import xrange
import warnings
import collections

import tensorflow as tf

import cleverhans.utils as utils
from cleverhans.model import Model, CallableModelWrapper
from cleverhans import utils_tf
from dro.staib.attacks_tf import fgm_distributional, distributional_gradient_step

import logging

_logger = utils.create_logger("cleverhans.attacks")


from cleverhans.attacks import Attack, FastGradientMethod



class NullAttack(Attack):

    """
    Attack which does nothing
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a NullAttack instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(NullAttack, self).__init__(model, back, sess)
        self.feedable_kwargs = {}
        self.structural_kwargs = []

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        """
        return x


class FastDistributionallyRobustMethod(Attack):

    """
    TODO: fill in explanation
    """

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        """
        Create a FastDistributionallyRobustMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(FastDistributionallyRobustMethod, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs = {'eps': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['point_ord', 'wasserstein_ord']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
        :param point_ord: (optional) Order of the norm to compare points (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param wasserstein_ord: (optional) Order of the norm to define Wasserstein distance (mimics Numpy).
        :param y: (optional) A tensor with the model labels. Only provide
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
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        adv_x, eps_batch = fgm_distributional(x, self.model.get_probs(x), y=labels, eps=self.eps,
                   point_ord=self.point_ord, wasserstein_ord=self.wasserstein_ord,
                   clip_min=self.clip_min,
                   clip_max=self.clip_max,
                   targeted=(self.y_target is not None))

        return adv_x, eps_batch

    def parse_params(self, eps=0.3, point_ord=np.inf, wasserstein_ord=2, y=None, y_target=None,
                     clip_min=None, clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
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
        """
        # Save attack-specific parameters

        self.eps = eps
        self.point_ord = point_ord
        self.wasserstein_ord = wasserstein_ord
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.point_ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.wasserstein_ord not in [np.inf] and self.wasserstein_ord <= 1:
            raise ValueError("Norm order must be either np.inf, or >1.")

        return True


class FrankWolfeDistributionallyRobustMethod(Attack):

    """
    Heuristic which tries to optimize the Distributionally Robust loss
    """

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        """
        Create a FrankWolfeDistributionallyRobustMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(FrankWolfeDistributionallyRobustMethod, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['point_ord', 'wasserstein_ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param point_ord: (optional) Order of the norm to compare points (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param wasserstein_ord: (optional) Order of the norm to define Wasserstein distance (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param retract: abandon projection/Frank-Wolfe in favor of just dividing by the norm
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        # at each iteration, eta is the 'current' perturbation
        eta = 0
        adv_x = x

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        grad_params = {y_kwarg: y}

        fdrm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max,
                      'point_ord': self.point_ord, 'wasserstein_ord': self.wasserstein_ord}

        for i in range(self.nb_iter):
            FDRM = FastDistributionallyRobustMethod(self.model,
                                                    back=self.back,
                                                    sess=self.sess)

            # Compute this step's perturbation
            adv_x_candidate, _ = FDRM.generate(adv_x, **fdrm_params)
            # eta = eta_plus_x - x
            # eta, epsilons_out = FDRM.generate(adv_x, **fdrm_params) - x

            #adv_x = adv_x_candidate
            # adv_x = eta_plus_x

            bound_total_eps_traveled = self.eps_iter * (i+1)
            bound_is_vacuous = bound_total_eps_traveled > self.eps

            col_norms = self._norm(adv_x - x)
            total_norm = tf.norm(col_norms, ord=self.wasserstein_ord)

            should_not_update = tf.logical_and(bound_is_vacuous, total_norm > self.eps)

            adv_x = tf.cond(should_not_update, lambda: adv_x, lambda: adv_x_candidate)

            # # if the bound on total distance traveled is no longer vacuous...
            # if bound_total_eps_traveled > self.eps:
            #     # check that we stayed feasible. If we didn't, revert and break
            #     col_norms = self._norm(adv_x - x)
            #     total_norm = tf.norm(col_norms, ord=self.wasserstein_ord)
            #     adv_x = tf.cond(total_norm < self.eps, adv_x_candidate, adv_x)
            # else:
            #     adv_x = adv_x_candidate



            # adv_x = adv_x + eta #self.eps_iter * eta

            # project back onto mixed norm ball
            # batch_size = tf.to_float(tf.shape(x)[0])
            # constraint = self.eps #* tf.pow(batch_size, 1.0 / self.wasserstein_ord)

            # can only comment this out if we are guaranteed to never leave the ball,
            # which happens if we choose eps_iter = eps / nb_iter
            # adv_x = x + proj_mixed_norm(self.sess, adv_x - x, self.wasserstein_ord, self.point_ord, constraint)

        # The final iterate of adv_x is the adversarial example (clip if necessary)
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        diff = adv_x - x

        epsilons_out = self._norm(diff)

        return adv_x, epsilons_out # are these actually the right epsilons?


    def _norm(self, v, keep_dims=False):

        red_ind = tf.range(1, tf.rank(v))

        if self.point_ord == np.inf:
            return tf.reduce_max(tf.abs(v),
                                 axis=red_ind,
                                 keep_dims=keep_dims)
        else:
            powers = tf.pow(v, self.point_ord)
            power_of_norm = tf.reduce_sum(tf.abs(powers),
                                          axis=red_ind,
                                          keep_dims=keep_dims)
            return tf.pow(power_of_norm, 1.0/self.point_ord)


    def parse_params(self, eps=0.3, point_ord=np.inf, wasserstein_ord=2, nb_iter=5, y=None,
                     ord=np.inf, clip_min=None, clip_max=None, eps_iter=0.1,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (formerly required float) step size for each attack iteration.
                         Currently set automatically to equal eps / nb_iter so
                         feasibility is always maintained
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.point_ord = point_ord
        self.wasserstein_ord = wasserstein_ord
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        # hard-coded:
        # self.eps_iter = self.eps / self.nb_iter

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.point_ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.wasserstein_ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        return True


class UnconstrainedDistributionallyRobustMethod(Attack):

    """
    Heuristic which tries to optimize the Distributionally Robust loss
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a FrankWolfeDistributionallyRobustMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(UnconstrainedDistributionallyRobustMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'gamma': np.float32,
                                'eps_iter': np.float32,
                                'decay_stepsize': np.bool,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['point_ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param point_ord: (optional) Order of the norm to compare points (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param wasserstein_ord: (optional) Order of the norm to define Wasserstein distance (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param retract: abandon projection/Frank-Wolfe in favor of just dividing by the norm
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        adv_x = x

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'

        start_x = x + 0

        for i in range(self.nb_iter):
            if self.decay_stepsize:
                stepsize = self.eps_iter / np.sqrt(i + 1)
            else:
                stepsize = self.eps_iter
            adv_x = distributional_gradient_step(adv_x,
                                                 start_x,
                                                 model_preds,
                                                 eps_iter=stepsize,
                                                 point_ord=self.point_ord,
                                                 gamma=self.gamma,
                                                 clip_min=self.clip_min,
                                                 clip_max=self.clip_max,
                                                 targeted=targeted)

        # The final iterate of adv_x is the adversarial example (clip if necessary)
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        diff = adv_x - start_x
        epsilons_out = self._norm(diff)

        return adv_x, epsilons_out


    def _norm(self, v, keep_dims=False):

        red_ind = tf.range(1, tf.rank(v))

        if self.point_ord == np.inf:
            return tf.reduce_max(tf.abs(v),
                                 axis=red_ind,
                                 keep_dims=keep_dims)
        else:
            powers = tf.pow(v, self.point_ord)
            power_of_norm = tf.reduce_sum(tf.abs(powers),
                                          axis=red_ind,
                                          keep_dims=keep_dims)
            return tf.pow(power_of_norm, 1.0/self.point_ord)


    def parse_params(self, gamma=0.3, point_ord=np.inf, nb_iter=5, y=None,
                     ord=np.inf, clip_min=None, clip_max=None, eps_iter=0.1,
                     decay_stepsize=True,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (formerly required float) step size for each attack iteration.
                         Currently set automatically to equal eps / nb_iter so
                         feasibility is always maintained
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.gamma = gamma
        self.point_ord = point_ord
        self.eps_iter = eps_iter
        self.decay_stepsize = decay_stepsize
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.point_ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True