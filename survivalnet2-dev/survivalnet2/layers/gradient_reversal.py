import tensorflow as tf


@tf.custom_gradient
def gradient_reversal(x):
    """The gradient revsersal layer implements an identity transformation on
    the forward pass, and reverses the upstream gradient on the backwards pass.
    This is used in adversarial networks like domain-adversarial training to
    penalize ability to predict a target variable.

    Parameters
    ----------
    x : tensor
        The incoming gradient from the downstream layer.

    Returns
    -------
    y : tensor
        Identity transformation of x.
    grad: tensor
        Negation of upstream gradient.

    Notes
    -----
    Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks."
    The journal of machine learning research 17.1 (2016): 2096-2030.
    """

    # forward pass
    y = tf.identity(x)

    # gradient reversal
    def grad(dy):
        return -dy

    return y, grad


class GradientReversal(tf.keras.layers.Layer):
    """This class implements a simple gradient reversal layer for use in keras
    functional or sequential models. The input is unmodified in the forward
    pass and the gradient is negated on the backwards pass. This is used in
    adversarial networks like domain-adversarial training to penalize ability
    to predict a target variable.

    Attributes
    ----------
    None
    """

    def __init__(self):
        super().__init__()

    def call(self, input):
        return gradient_reversal(input)
