from dro import keys
from dro.sinha.utils_tf import model_eval


def model_eval_fn(sess, x, y, predictions, predictions_adv, X_test, Y_test, eval_params,
                  dataset_iterator=None, nb_batches=None):
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy, cross_entropy = model_eval(sess, x, y, predictions, X_test, Y_test,
                                         args=eval_params,
                                         dataset_iterator=dataset_iterator,
                                         nb_batches=nb_batches,
                                         return_extended_metrics=True)
    print('Test accuracy on legitimate test examples: %0.4f' % accuracy)

    # Accuracy of the model on Wasserstein adversarial examples
    accuracy_adv_wass, ce_adv_wass = model_eval(sess, x, y, predictions_adv, X_test,
                                                Y_test, args=eval_params,
                                                dataset_iterator=dataset_iterator,
                                                nb_batches=nb_batches)
    print('Test accuracy on Wasserstein examples: %0.4f\n' % accuracy_adv_wass)
    metrics_dict = {keys.ACC: accuracy,
                    keys.CE: cross_entropy,
                    keys.CE_ADV_W: ce_adv_wass
                    keys.ACC_ADV_W: accuracy_adv_wass}
    return metrics_dict
