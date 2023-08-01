import tensorflow as tf


def stack_labels(times, events):
    """Combines times and events into an N x 2 tensor for passing as a
    single argument to keras metrics, losses, and model functions.

    Parameters
    ----------
    times : float
        A 1D vector of event of last followup times for N subjects.
    events : bool or float
        A 1D vector of event statuses for N subjects. If bool, a value of
        True indicates that event was observed. If float, any nonzero value
        will be considered as an event.

    Returns
    -------
    labels : float
        An N x 2 float32 tensor where the times are in the first column and the
        labels are in the second column.

    See Also
    --------
    unstack_labels : Inverts the stacking to separate times, events.
    """

    # get type of times tensor
    time_type = times.dtype

    # cast and combine into a single tensor
    return tf.stack([tf.squeeze(times), tf.squeeze(tf.cast(events, time_type))], axis=1)


def unstack_labels(labels):
    """Separates times and events from an N x 2 tensor into separate
    float32 times and bool events. Used by functions receiving stacked
    times/events.

    Parameters
    ----------
    labels : float
        An N x 2 float32 tensor where the times are in the first column and the
        labels are in the second column.

    Returns
    -------
    times : float
        A 1D vector of event of last followup times for N subjects.
        Mask is applied if the associated label is not available in labels tensor
    events : bool
        A 1D vector of event statuses for N subjects. True indicates
        that event was observed.
        Mask is applied if the associated label is not available in labels tensor

    See Also
    --------
    stack_labels : Combines times, events into an N x 2 tensor.
    """

    # unstack
    times, events = tf.unstack(labels, axis=1)

    # cast events to bool
    events = tf.cast(events, tf.bool)

    return times, events


def mask(x, condition=tf.math.is_nan):
    """Mask tensor to remove rows where some condition is met. This is used
    broadly by losses and metrics to remove samples containing an nan value in
    their label.

    Parameters
    ----------
    x : tensor
        An N x M tensor where samples are in rows. This can be a tensor of
        labels, predictions, or features.
    condition : function
        A tensorflow function that tests for the criteria for masking a sample.
        Accepts a two-dimensional tensor and returns a bool tensor of same shape.

    Returns
    -------
    masked : float
        A K x M float32 tensor where K <= N and any row/sample containing
        an NaN value is removed.
    keep : bool
        An N-length bool tensor where True indicates which samples were retained.
    """

    # generate N-length boolean mask by reducing all across label columns
    keep = tf.reduce_all(tf.logical_not(condition(x)), axis=1)

    # apply mask
    masked = tf.boolean_mask(x, keep, axis=0)

    return masked, keep
