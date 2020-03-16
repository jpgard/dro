import matplotlib.pyplot as plt
from math import sqrt
from tensorflow.keras.preprocessing.image import array_to_img


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
    for n in range(batch_size):
        ax = plt.subplot(int(sqrt(batch_size)) + 1, int(sqrt(batch_size)) + 1, n + 1)
        plt.imshow(image_batch[n])
        if label_batch is not None:
            plt.title(str(label_batch[n]))
        plt.axis('off')
    if fp:
        print("[INFO] saving batch to {}".format(fp))
        plt.savefig(fp)
    else:
        plt.show()

def show_adversarial_resuts(batch_image, batch_label, batch_pred, adv_image_fp,
                            n_row, n_col):
    plt.figure(figsize=(15, 15))
    for i, (image, y) in enumerate(zip(batch_image, batch_label)):
        y_base = batch_pred['base'][i]
        y_adv = batch_pred['adv-regularized'][i]
        plt.subplot(n_row, n_col, i + 1)
        plt.title('true: %d, base: %d, adv: %d' % (y, y_base, y_adv))
        plt.imshow(array_to_img(image))
        plt.axis('off')
    print("[INFO] writing adversarial examples to {}".format(adv_image_fp))
    plt.savefig(adv_image_fp)
