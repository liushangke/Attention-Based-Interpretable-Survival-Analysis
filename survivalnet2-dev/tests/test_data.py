import numpy as np
import pytest
import tensorflow as tf
from survivalnet2.data.labels import stack_labels, unstack_labels, mask


def test_labels_stack_labels():
    # unequal sizes
    with pytest.raises(tf.errors.InvalidArgumentError):
        stack_labels(tf.constant([0.0, 1.0, 2.0]), tf.constant([0, 1]))

    # combining rank 1, rank 2 degenerate tensors
    stacked = stack_labels(tf.constant([0.0, 1.0, 2.0]), tf.constant([0, 1, 1]))
    degenerate = stack_labels(tf.constant([0.0, 1.0, 2.0]), tf.constant([[0, 1, 1]]))
    assert tf.reduce_all(degenerate == stacked)
    degenerate = stack_labels(tf.constant([[0.0, 1.0, 2.0]]), tf.constant([0, 1, 1]))
    assert tf.reduce_all(degenerate == stacked)


def test_labels_unstack_labels():
    # degenerate rank 0 tensor
    with pytest.raises(ValueError):
        unstack_labels(tf.zeros((3)))

    # dimension 1 > 2
    with pytest.raises(ValueError):
        unstack_labels(tf.zeros((3, 3)))

    # empty input
    times, events = unstack_labels(tf.zeros((0, 2)))
    assert tf.size(times) == 0
    assert tf.size(events) == 0


def test_labels_mask():
    # NaN in first, second columns
    labels = tf.stack([[1.0, 0.0], [np.NaN, 0.0]], axis=1)
    masked, _ = mask(labels)
    assert tf.reduce_all(masked == tf.zeros((1, 2)))
    labels = tf.stack([[np.NaN, 0.0], [1, 0.0]], axis=1)
    masked, _ = mask(labels)
    assert tf.reduce_all(masked == tf.zeros((1, 2)))

    # check alternative test function argument
    func = lambda x: tf.math.greater(x, 0)
    labels = tf.stack([[0.0, 1.0], [0.0, 1.0]], axis=1)
    masked, _ = mask(labels, func)
    assert tf.reduce_all(masked == tf.zeros((1, 2)))
