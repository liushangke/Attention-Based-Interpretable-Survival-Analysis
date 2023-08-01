import tensorflow as tf


# Checking if variable inputs is a ragged tensor
def isragged(inputs):
    return isinstance(inputs, tf.RaggedTensor)
