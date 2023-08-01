import tensorflow as tf
from survivalnet2.data.labels import mask, unstack_labels


def _selectragged(x, selected):
    """Generate a ragged tensor from selected elements of x.

    Parameters
    ----------
    x : tensor
        A two-dimensional tensor.
    selected : tensor
        A two-dimensional tensor with the same size of x, where non-zero
        elements indicate which entries of x to select.

    Returns
    -------
    ragged : RaggedTensor
        A RaggedTensor containing the selected elements of x.
    """

    # gather indices and values
    indices = tf.where(selected)
    values = tf.gather_nd(x, indices)

    return tf.RaggedTensor.from_value_rowids(values, indices[:, 0])


def _at_risk(event_times, times, y_pred):
    """Generates a tensor of predictions for the at risk sets at each event
    time.

    Parameters
    ----------
    event_times : tensor
        An M length tensor containing the times where events occurr.
    times : tensor
        An N length tensor containing all times (N >= M).
    y_pred : tensor
        An N length tensor containing predictions for all times.

    Returns
    -------
    atrisk : tensor
        An M x N tensor where row i contains the predictions of the at-risk set
        for event_times[i]. Values for the other samples are set to zero.
    indicator : tensor
        An M x N tensor where row i nonzero values indicate the at-risk set
        for event_times[i]. This tensor is necessary since predictions for
        at-risk set samples may be zero, and so the at-risk set cannot be
        identified from atrisk alone.
    """

    # create times[events] x N "at risk" indicator matrix
    indicator = tf.less_equal(
        tf.reshape(event_times, (tf.size(event_times), 1)),
        tf.reshape(times, (1, tf.size(times))),
    )

    # create matrix of inner sum terms
    atrisk = tf.reshape(y_pred, (1, tf.size(times))) * tf.cast(indicator, tf.float32)

    return atrisk, indicator


def cox(y_true, y_pred):
    """Calculates Cox negative partial log likelihood.

    This function returns a NaN value when evaluated on a dataset containing no
    events.

    Parameters
    ----------
    y_true : tensor (float32)
        An N x 2 float32 tensor where event or last followup times are in the
        first column and event indicators are in the second column.
    y_pred : tensor (float32)
        Predicted risks for input samples.
    """

    def defined(times, events, y_pred):
        # create times[events] x N "at risk" set tensor
        atrisk, atrisk_indicator = _at_risk(times[events], times, y_pred)

        # build a ragged tensor containing only nonzero values from each row
        atrisk = _selectragged(atrisk, atrisk_indicator)

        # generate maxima along each event time
        maxima = tf.reduce_max(atrisk, axis=1)

        # shift atrisk by maximum risk
        shifted = atrisk - tf.expand_dims(maxima, axis=1)

        # log(sum(exp(atrisk-maximum)) + maximum
        logsum = tf.math.log(tf.reduce_sum(tf.exp(shifted), axis=1)) + maxima

        # final sum over uncensored risks
        log_likelihood = tf.reduce_sum(tf.reshape(y_pred[events], [-1]) - logsum)

        return -log_likelihood

    # mask and unpack the labels
    masked, keep = mask(y_true)
    times, events = unstack_labels(masked)

    # if events exist, loss is defined, otherwise return NaN
    loss = tf.cond(
        tf.reduce_sum(tf.cast(events, tf.int32)) > 0,
        lambda: defined(times, events, y_pred),
        lambda: tf.constant(float("NaN"), tf.float32),
    )

    return loss


def _tied_atrisk(atrisk, equal):
    """Calculates logsumexp terms in Efron's approximation for one event time.

    Parameters
    ----------
    atrisk : tensor
        A tensor of predictions for one at risk set.
    equal : tensor
        A tensor of predictions for samples with events at the given time.

    Returns
    -------
    logsum : tensor
        A scalar tensor of the logsumexp for the Efron partial likelihood.
    """

    # get maxima of exponent terms in both equal, atrisk
    maxima = tf.reduce_max(tf.concat([atrisk, equal], axis=-1), axis=-1)

    # weighting terms [0,...,events-1]/|events| for current time
    m = tf.size(equal)
    l = tf.cast(tf.range(0, m, dtype=tf.int32), tf.float32) / tf.cast(m, tf.float32)
    l = tf.reshape(l, [m, 1])

    # repeat weightings to form |events| x
    equal = tf.repeat(tf.reshape(equal, [1, m]), m, axis=0)

    # tile atrisk to match size of equal - if equal empty then pass through
    atrisk = tf.repeat(
        tf.reshape(atrisk, [1, tf.size(atrisk)]), tf.maximum(1, m), axis=0
    )

    # reshape maxima for broadcast subtraction
    maxima = tf.reshape(tf.repeat(maxima, m), [m, 1])

    # max-shifted exponential terms with tied weighting
    shifted = tf.concat([tf.exp(atrisk - maxima), -l * tf.exp(equal - maxima)], axis=1)

    # calculate logsumexp over these values
    logsum = tf.reshape(tf.math.log(tf.reduce_sum(shifted, axis=1)), [m, 1]) + maxima

    # return total sum
    return tf.reduce_sum(logsum)


def efron(y_true, y_pred):
    """Calculates the partial log likelihood for Efron's approximation for tied
    times.

    This function returns a NaN value when evaluated on a dataset containing no
    events.

    Parameters
    ----------
    y_true : tensor (float32)
        An N x 2 float32 tensor where event or last followup times are in the
        first column and event indicators are in the second column.
    y_pred : tensor (float32)
        Predicted risks for input samples.
    """

    def defined(times, events, y_pred):
        # calculate unique event times
        unique, _ = tf.unique(times[events])

        # create |unique| x N "at risk" set tensor
        atrisk, atrisk_indicator = _at_risk(unique, times, y_pred)

        # convert to ragged tensor containing only nonzero values from each row
        atrisk = _selectragged(atrisk, atrisk_indicator)

        # create times[events] x N tensor of times[events] == times
        equal_indicator = tf.equal(
            tf.reshape(unique, (tf.size(unique), 1)),
            tf.reshape(times[events], (1, tf.size(times[events]))),
        )
        equal = tf.reshape(y_pred[events], (1, tf.size(times[events]))) * tf.cast(
            equal_indicator, tf.float32
        )

        # convert to ragged tensor containing only nonzero values from each row
        equal = _selectragged(equal, equal_indicator)

        # map loop function onto rows of ragged tensors
        logsum = tf.map_fn(
            lambda x: _tied_atrisk(x[0], x[1]),
            (atrisk, equal),
            fn_output_signature=tf.float32,
        )

        # calculate first term - y_true for set {j | t_j = t_i}, event times {t_i}
        log_likelihood = tf.reduce_sum(tf.reduce_sum(equal, axis=1) - logsum)

        return -log_likelihood

    # mask and unpack the labels
    masked, keep = mask(y_true)
    times, events = unstack_labels(masked)

    # if events exist, loss is defined, otherwise return NaN
    loss = tf.cond(
        tf.reduce_sum(tf.cast(events, tf.int32)) > 0,
        lambda: defined(times, events, y_pred),
        lambda: tf.constant(float("NaN"), tf.float32),
    )

    return loss
