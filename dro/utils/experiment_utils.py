from dro import keys
from dro.sinha.utils_tf import model_eval


def model_eval_fn(sess, x, y, predictions, predictions_adv, X_test, Y_test, eval_params,
                  dataset_iterator=None, nb_batches=None):
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    acc, acc_std, ce, ce_std = model_eval(sess, x, y, predictions, X_test, Y_test,
                                          args=eval_params,
                                          dataset_iterator=dataset_iterator,
                                          nb_batches=nb_batches,
                                          return_extended_metrics=True)
    print('Test accuracy %0.4f (%0.4f) and cross-entropy %0.4f (%0.4f) on legitimate '
          'test examples' %
          (acc, acc_std, ce, ce_std))

    # Accuracy of the model on Wasserstein adversarial examples
    acc_adv_wass, acc_std_adv_wass, ce_adv_wass, ce_std_adv_wass = model_eval(
        sess, x, y, predictions_adv, X_test, Y_test,
        args=eval_params,
        dataset_iterator=dataset_iterator,
        nb_batches=nb_batches,
        return_extended_metrics=True)
    print('Test accuracy %0.4f (%0.4f) and cross-entropy %0.4f (%0.4f) on Wasserstein '
          'examples' %
          (acc_adv_wass, acc_std_adv_wass, ce_adv_wass, ce_std_adv_wass))
    metrics_dict = {keys.ACC: acc,
                    keys.ACC_SIGMA: acc_std,
                    keys.CE: ce,
                    keys.CE_SIGMA: ce_std,
                    keys.CE_ADV_W: ce_adv_wass,
                    keys.CE_ADV_W_SIGMA: ce_std_adv_wass,
                    keys.ACC_ADV_W: acc_adv_wass,
                    keys.ACC_ADV_W_SIGMA: acc_std_adv_wass}
    return metrics_dict
