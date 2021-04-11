import tensorflow as tf
import numpy as np
from tensorflow import keras

class CustomMetricsModel(tf.keras.Model):
    def __init__(self, reverse_token_map, **kwargs):
        super(CustomMetricsModel, self).__init__(**kwargs)
        # self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
        #     reduction=tf.keras.losses.Reduction.NONE
        # )
        # self.accr_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.reverse_token_map = reverse_token_map

    def train_step(self, inputs):
        if len(inputs) == 3:
            x, y, sample_weight = inputs
        else:
            x, y = inputs
            sample_weight = None


        if "<PAD>" in self.reverse_token_map:
            sample_weight = y != self.reverse_token_map["<PAD>"]

        with tf.GradientTape() as tape:

            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        if len(inputs) == 3:
            x, y, sample_weight = inputs
        else:
            x, y = inputs
            sample_weight = None

        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

