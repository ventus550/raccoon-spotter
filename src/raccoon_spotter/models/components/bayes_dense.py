import keras
from keras import initializers, ops
from keras.layers import *


def kullback_leibler_divergence(prior_mu, prior_sigma, weight_mu, weight_sigma):
    kl = (
        weight_sigma
        - prior_sigma
        + (ops.exp(prior_sigma) ** 2 + (prior_mu - weight_mu) ** 2)
        / (2 * ops.exp(weight_sigma) ** 2)
        - 0.5
    )
    return ops.sum(kl)


@keras.saving.register_keras_serializable()
class BayesDense(keras.layers.Layer):
    def __init__(  # noqa: PLR0913
        self,
        units,
        prior_mu=0,
        prior_sigma=0.1,
        alpha=1,
        bias=True,
        activation="linear",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.alpha = alpha
        self.bias = bool(bias)

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = ops.log(prior_sigma)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        input_shape = input_shape[-1]
        stdv = 1.0 / ops.sqrt(input_shape)

        self.weight_mu = self.add_weight(
            shape=(input_shape, self.units),
            initializer=initializers.RandomUniform(-stdv, stdv),
            trainable=True,
        )

        self.weight_log_sigma = self.add_weight(
            shape=(input_shape, self.units),
            initializer=initializers.Constant(self.prior_log_sigma),
            trainable=True,
        )

        if self.bias:
            self.bias_mu = self.add_weight(
                shape=(self.units,),
                initializer=initializers.RandomUniform(-stdv, stdv),
                trainable=True,
            )

            self.bias_log_sigma = self.add_weight(
                shape=(self.units,),
                initializer=initializers.Constant(self.prior_log_sigma),
                trainable=True,
            )

    def call(self, x):
        loss = self.alpha * kullback_leibler_divergence(
            self.prior_mu, self.prior_sigma, self.weight_mu, self.weight_log_sigma
        )
        loss += self.alpha * kullback_leibler_divergence(
            self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_log_sigma
        )
        self.add_loss(
            loss
            / ops.cast(
                ops.size(self.weight_mu) + ops.size(self.bias_mu), dtype="float32"
            )
        )
        weight = self.weight_mu + ops.exp(self.weight_log_sigma) * keras.random.normal(
            self.weight_log_sigma.shape
        )
        bias = self.bias_mu + ops.exp(self.bias_log_sigma) * keras.random.normal(
            self.bias_log_sigma.shape
        )
        return self.activation(x @ weight + bias)
