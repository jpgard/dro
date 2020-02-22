"""
Usage:
    python3 train_mnist.py
"""

import numpy as np
from absl import flags, app

import tensorflow as tf
from tensorflow.keras.datasets import mnist

from dro.training.models import BaselineModel
from dro.training.optimizer import optimizer_from_flags

flags.DEFINE_string("optimizer", default="sgd", help="Name of optimizer to use.")

FLAGS = flags.FLAGS


def main(argv):
    batch_size = 32

    baseline_model = BaselineModel()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Initialize epsilon
    epsilon_init = 0.1  # value to initialize epsilon; see Section 4 of paper
    p = 2
    q = np.inf
    n_train = x_train.shape[0]
    epsilon_train = tf.fill([n_train, ], epsilon_init)

    # train_indices are used to index into epsilon during training.
    train_indices = tf.convert_to_tensor(np.arange(n_train))

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train, train_indices)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = optimizer_from_flags(FLAGS)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = baseline_model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, baseline_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, baseline_model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = baseline_model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels, s in train_ds:
            X_s_T = tf.reshape(images, [batch_size, -1])
            X_s = tf.transpose(X_s_T)  # [n_features, batch_size]
            # Below is an example of using s to index into a tensor, here epsilon
            # tf.gather(epsilon_train, tf.reshape(s, [-1]))

            # TODO(jpgard): double-check the norm logic here is correct; tensorflow
            #  apparently cannot compute the p,q norm directly itself.

            # compute p-norm along rows; resulting tensor has shape [batch_size,]
            X_s_norm_p = tf.norm(X_s, ord=p, axis=0)
            # compute q-norm along cols, resulting tensor is a constant
            X_s_norm_pq = tf.norm(X_s_norm_p, ord=q)
            epsilon_s = X_s_norm_pq
            # TODO: continue with Algorithm 1 by solving OPT problem here using
            #  procedure from Appendix B.

            # Currently we are doing this for one obs, x_i.
            gamma = epsilon_s  # this is the constraint on the infty-norm of the result.
            i = 0
            X_i = X_s[:,i]  # X_i is the ith column (observation) of X_s.
            c_i = X_i
            x_i_star = gamma * tf.sign(c_i)
            v_i = x_i_star
            a_i = tf.tensordot(c_i, v_i)  # this is a scalar
            import ipdb;
            ipdb.set_trace()





            # TODO(jpgard): probably need to modify train_step.
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))


if __name__ == "__main__":
    app.run(main)
