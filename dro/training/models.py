import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense

class BaselineModel(tf.keras.Model):
    """A base model defining the default architecture described in
    https://machine-learning-and-security.github.io/papers/mlsec17_paper_30.pdf ."""
    def __init__(self):
        super(BaselineModel, self).__init__()
        # TODO(jpgard): find number of filters from Cleverhans tutorials.
        self.conv_a = Conv2D(32, 8, activation='relu')
        self.conv_b = Conv2D(32, 6, activation='relu')
        self.conv_c = Conv2D(32, 5, activation='relu')
        self.flatten = Flatten()
        self.out = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv_a(x)
        x = self.conv_b(x)
        x = self.conv_c(x)
        x = self.flatten(x)
        return self.out(x)
