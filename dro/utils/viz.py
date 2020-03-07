import matplotlib.pyplot as plt

def plot_faces(img_ary, nplot=7, figsize=(30, 10)):
    """Generate a plot of a sample of the first nplot faces in img_ary."""
    fig = plt.figure(figsize=figsize)
    for count in range(1, nplot):
        ax = fig.add_subplot(1, nplot, count)
        ax.imshow(img_ary[count])
    plt.show()
    return

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(6, 14))
    for n in range(16):
        ax = plt.subplot(8, 2, n + 1)
        plt.imshow(image_batch[n])
        plt.title(str(label_batch[n]))
        plt.axis('off')
    plt.show()
