import matplotlib.pyplot as plt
from math import sqrt
from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np
from dro.keys import ADV_MODEL, BASE_MODEL


def plot_faces(img_ary, nplot=7, figsize=(30, 10)):
    """Generate a plot of a sample of the first nplot faces in img_ary."""
    fig = plt.figure(figsize=figsize)
    for count in range(1, nplot):
        ax = fig.add_subplot(1, nplot, count)
        ax.imshow(img_ary[count])
    plt.show()
    return


def show_batch(image_batch, label_batch=None, fp=None):
    plt.figure(figsize=(12, 12))
    batch_size = len(image_batch)
    n_rowcol = int(sqrt(batch_size)) + 1
    for n in range(batch_size):
        ax = plt.subplot(n_rowcol, n_rowcol, n + 1)
        plt.imshow(image_batch[n])
        if label_batch is not None:
            plt.title(str(label_batch[n]))
        plt.axis('off')
    if fp:
        print("[INFO] saving batch to {}".format(fp))
        plt.savefig(fp)
    else:
        plt.show()


def show_adversarial_resuts(n_batches: int, perturbed_images, labels, predictions,
                            fp_basename, batch_size):
    """
    Generate a plot of adversarial results.

    :param n_batches: number of batches to plot.
    :param perturbed_images: numpy array of shape [n_batches, width, height, n_channnels]
    :param labels: array of labels.
    :param predictions: list of dicts, where each dict has format {MODEL_NAME:
    model_preds} and model_preds is a binary vector of predictions.
    :param fp_basename: basename of file to write to.
    :param batch_size: batch size.
    """
    n_col = 4
    n_row = (batch_size + n_col - 1) / n_col
    n_batches = min(n_batches, len(perturbed_images))
    for batch_index in range(n_batches):
        batch_image = perturbed_images[batch_index]
        batch_label = labels[batch_index]
        batch_pred = predictions[batch_index]
        # Main title, which gives the batchwise accuracy results for each model
        acc_summary = ''.join(
            ['%s model: %d / %d' % (name, np.sum(batch_label == pred), batch_size)
             for name, pred in batch_pred.items()]
        )
        # Create the figure and save it.
        plt.figure(figsize=(15, 15))
        plt.suptitle(acc_summary)
        for i, (image, y) in enumerate(zip(batch_image, batch_label)):
            y_base = batch_pred[BASE_MODEL][i]
            y_adv = batch_pred[ADV_MODEL][i]
            plt.subplot(n_row, n_col, i + 1)
            plt.title('true: %d, base: %d, adv: %d' % (y, y_base, y_adv))
            plt.imshow(array_to_img(image))
            plt.axis('off')
        adv_image_fp = fp_basename + "batch" + str(batch_index) + ".png"
        print("[INFO] writing adversarial examples to {}".format(adv_image_fp))
        plt.savefig(adv_image_fp)
        plt.close('all')
