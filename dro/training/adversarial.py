import neural_structured_learning as nsl
import tensorflow as tf

from dro.keys import ACC, CE, IMAGE_INPUT_NAME, LABEL_INPUT_NAME
from dro.training.training_utils import metrics_to_dict


def perturb_and_evaluate(test_ds_adv, models_to_eval, reference_model):
    """Perturbs the entire test set using adversarial training and computes metrics
    over that set.

    :returns: A tuple for four elements: a Tensor of the perturbed images; a List of
    the labels; a List of the predictions for the models; and a nested dict of
    per-model metrics.
    """
    print("[INFO] perturbing images and evaluating models on perturbed data...")
    perturbed_images, labels, predictions = [], [], []

    # TODO(jpgard): implement additional metrics as tf.keras.Metric subclasses; see
    #  https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/metrics/Metric

    metrics = {model_name: [tf.keras.metrics.SparseCategoricalAccuracy(name=ACC),
                            tf.keras.metrics.SparseCategoricalCrossentropy(name=CE),
                            ]
               for model_name in models_to_eval.keys()
               }
    for batch in test_ds_adv:
        perturbed_batch = reference_model.perturb_on_batch(batch)
        # Clipping makes perturbed examples have the same range as regular ones.
        perturbed_batch[IMAGE_INPUT_NAME] = tf.clip_by_value(
            perturbed_batch[IMAGE_INPUT_NAME], 0.0, 1.0)
        y_true = tf.argmax(perturbed_batch.pop(LABEL_INPUT_NAME), axis=-1)
        perturbed_images.append(perturbed_batch[IMAGE_INPUT_NAME].numpy())
        labels.append(y_true.numpy())
        predictions.append({})
        for model_name, model in models_to_eval.items():
            y_pred = model(perturbed_batch)
            predictions[-1][model_name] = tf.argmax(y_pred, axis=-1).numpy()
            for i in range(len(metrics[model_name])):
                metrics[model_name][i](y_true, y_pred)
    print("[INFO] perturbation evaluation complete.")
    metrics = metrics_to_dict(metrics)
    return perturbed_images, labels, predictions, metrics


def make_compiled_reference_model(model_base, adv_config, model_compile_args):
    reference_model = nsl.keras.AdversarialRegularization(
        model_base,
        label_keys=[LABEL_INPUT_NAME],
        adv_config=adv_config)
    reference_model.compile(**model_compile_args)
    return reference_model