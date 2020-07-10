import tensorflow as tf
from tensorflow.python.keras.metrics import AUC, TruePositives, FalsePositives, \
    TrueNegatives, FalseNegatives

from dro.keys import ACC, CE


def get_train_metrics():
    """Fetch an OrderedDict of train metrics."""
    train_metrics_dict = {ACC: ACC,
                          'auc': AUC(name='auc'),
                          'tp': TruePositives(name='tp'),
                          'fp': FalsePositives(name='fp'),
                          'tn': TrueNegatives(name='tn'),
                          'fn': FalseNegatives(name='fn'),
                          CE: tf.keras.losses.CategoricalCrossentropy(),
                          'sigma_ce': cross_entropy_sigma,
                          'max_ce': cross_entropy_max
                          }
    return train_metrics_dict


def add_adversarial_metric_names_to_list(metrics_list):
    """The adversarial training has different metrics -- add these to metrics_list."""
    return ["total_combined_loss", ] + metrics_list + ["adversarial_loss", ]


def cross_entropy_sigma(y_true, y_pred):
    """Custom metric to compute the standard deviation of the loss."""
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    element_wise_loss = loss_object(y_true=y_true, y_pred=y_pred)
    loss_std = tf.math.reduce_std(element_wise_loss)
    return loss_std


def cross_entropy_max(y_true, y_pred):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    element_wise_loss = loss_object(y_true=y_true, y_pred=y_pred)
    loss_max = tf.math.reduce_max(element_wise_loss)
    return loss_max