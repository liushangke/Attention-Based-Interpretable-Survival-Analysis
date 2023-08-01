import numpy as np
import tensorflow as tf
from survivalnet2.data.labels import mask, unstack_labels


def km(labels):
    """Tensorflow Kaplan Meier estimator of the survival function.

    Parameters
    ----------
    labels : float
        An N x 2 float32 tensor where the times are in the first column and the
        labels are in the second column. Any nonzero event value -> event was
        observed.

    Returns
    -------
    t_i : float
        Tensor of times where survival function values in 's_t' are defined.
    s_t : float
        Tensor of estimated survival function values for times in 't_i'.

    See Also
    --------
    km_eval : evaluates Kaplan Meier estimator at specific time.
    km_np : Numpy Kaplan Meier estimator.
    """
    # pylint: disable=no-value-for-parameter
    # unstack times, events
    masked, keep = mask(labels)
    times, events = unstack_labels(masked)

    # unique event times, event count
    order = tf.argsort(times)
    s_times = tf.gather(times, order)
    s_events = tf.gather(events, order)
    t_i, _, d_i = tf.unique_with_counts(tf.boolean_mask(s_times, s_events))

    # calculate events, at-risk
    n_i = tf.TensorArray(dtype=tf.float32, size=tf.size(t_i))
    at_risk = lambda x, y: tf.cast(tf.greater_equal(x, y), tf.float32)
    cond = lambda i, _: tf.less(i, tf.size(t_i))
    body = lambda i, n_i: (i + 1, n_i.write(i, tf.reduce_sum(at_risk(times, t_i[i]))))
    tf.while_loop(cond, body, (0, n_i))
    n_i = n_i.stack()

    # estimate survival function
    s_t = tf.math.cumprod(1 - tf.math.divide(tf.cast(d_i, tf.float32), n_i))

    # add origin
    t_i = tf.concat([tf.zeros(1, dtype=t_i.dtype), t_i], 0)
    s_t = tf.concat([tf.ones(1, dtype=s_t.dtype), s_t], 0)

    return t_i, s_t


def km_eval(times, t_i, s_t):
    """Tensorflow evaluation of Kaplan Meier estimated survival function.

    Parameters
    ----------
    times : float
        A tensor of event of last followup times for each subject.
    t_i : float
         Tensor of times where survival function values in 's_t' are defined.
    s_t : float
        Tensor of estimated survival function values for times in 't_i'.

    Returns
    -------
    s_i : float
        Survival function values at times 'times' from km estimator in 't_i'
        and 's_t'.

    See Also
    --------
    km : TensorFlow Kaplan Meier estimator.
    """
    # pylint: disable=no-value-for-parameter
    # pylint: disable=unexpected-keyword-arg

    # calculate differences between km times and event/followup times
    delta = tf.reshape(times, (tf.size(times), 1)) - tf.reshape(t_i, (1, tf.size(t_i)))

    # add column of -1 to end - if time exceeds max(t_i) index should = end
    delta = tf.concat([delta, -tf.ones((tf.size(times), 1), tf.float32)], axis=1)

    # manipulate differences to find smallest negative - t_i are sorted by km
    delta = tf.sign(tf.multiply(delta, tf.cast(delta < 0.0, tf.float32)))

    # calculate index of first negative and subtract 1
    index = tf.argmin(delta, axis=1) - 1

    # gather s_t values at indices
    s_i = tf.gather(s_t, index)

    return s_i


def km_np(labels):
    """Kaplan Meier estimator of the survival function. Estimates confidence
    intervals using log-log method with Greenwood's formula.

    Parameters
    ----------
    labels : float
        An N x 2 float32 tensor where the times are in the first column and the
        labels are in the second column. Any nonzero event value -> event was
        observed.

    Returns
    -------
    t_i : float
        Array of times where survival function values in 's_t' are defined.
    s_i : float
        Array of estimated survival function values for times in 't_i'.
    med_t_i : float
        Estimated median survival time. 'None' if not defined due to censoring.
    upper : float
        Array of upper 95% confidence interval of 's_i' at times 't_i'.
    lower : float
        Array of lower 95% confidence interval of 's_i' at times 't_i'.
    n_i : float
        Array of number of at-risk samples at times 't_i'.
    c_i : float
        Array of times associated with censored samples (duplicates removed).
    s_c : float
        Array of survival function values associated with times in 'c_i'.

    See Also
    --------
    km_plot : Plotting of Kaplan Meier estimates of survival function.
    """

    # unstack times, events
    masked, _ = mask(labels)
    times, events = unstack_labels(masked)

    # unique event times
    t_i = np.sort(np.unique(times[events]))

    # initialize d_i, n_i
    d_i = np.zeros(t_i.shape, dtype=np.float32)
    n_i = np.zeros(t_i.shape, dtype=np.float32)

    # calculate d_i, n_i at each unique event time
    for i, t in enumerate(t_i):
        # all events at time t_i
        d_i[i] = np.sum(np.logical_and(times == t, events))

        # all at risk at t_i
        n_i[i] = np.sum(times >= t)

    # estimate survival function
    s_t = np.cumprod(1 - d_i / n_i)

    # calculate confidence interval
    if d_i[-1] == n_i[-1]:
        a = 1.96 * np.sqrt(
            1
            / np.log(s_t[0:-1]) ** 2
            * np.cumsum(d_i[0:-1] / (n_i[0:-1] * (n_i[0:-1] - d_i[0:-1])))
        )
        lower = s_t[0:-1] ** np.exp(a)
        upper = s_t[0:-1] ** np.exp(-a)
        lower = np.append(lower, lower[-1])
        upper = np.append(upper, upper[-1])
    else:
        a = 1.96 * np.sqrt(1 / np.log(s_t) ** 2 * np.cumsum(d_i / (n_i * (n_i - d_i))))
        lower = s_t ** np.exp(a)
        upper = s_t ** np.exp(-a)

    # add origin
    t_i = np.insert(t_i, 0, 0)
    s_t = np.insert(s_t, 0, 1)
    lower = np.insert(lower, 0, 1)
    upper = np.insert(upper, 0, 1)

    # handle end condition where last sample is censored
    if ~events[np.argmax(times)]:
        t_i = np.append(t_i, np.max(times))
        s_t = np.append(s_t, s_t[-1])
        lower = np.append(lower, lower[-1])
        upper = np.append(upper, upper[-1])

    # calculate median survival time
    if np.min(s_t) <= 0.5:
        med_t_i = np.min(t_i[s_t <= 0.5])
    else:
        med_t_i = None

    # get location of censored points
    c_i = np.unique(times[~events])
    s_c = np.ones(np.unique(times[~events]).size)
    for i, t in enumerate(c_i):
        index = np.argwhere(t > t_i)
        if index.size > 0:
            s_c[i] = s_t[np.max(index)]

    return t_i, s_t, med_t_i, upper, lower, n_i, c_i, s_c
