from survivalnet2.estimators import km, km_eval
from survivalnet2.data.labels import mask, stack_labels, unstack_labels
import tensorflow as tf


def _brier(y_true, probs, t, km_t, km_s):
    # calculates brier score at time t without normalizing by number of samples

    # mask and unpack the labels
    masked, keep = mask(y_true)
    times, events = unstack_labels(masked)

    # reshape y_pred from [N, 1] to [N]
    probs = tf.reshape(probs, [-1])

    # Mask the survival probabilities
    probs = tf.boolean_mask(probs, keep, axis=0)

    # generate IPCW weights for times and t
    g_i = km_eval(times, km_t, km_s)
    g_t = km_eval(tf.repeat(t, tf.shape(times)[0]), km_t, km_s)

    # inverse weighted survival functions
    p1 = tf.boolean_mask(tf.math.square(probs) / g_i, events & (times <= t))
    p2 = tf.boolean_mask(tf.math.square(1 - probs) / g_t, times > t)

    # return unnormalized sum
    return tf.reduce_sum(p1) + tf.reduce_sum(p2)


class Brier(tf.keras.metrics.Metric):
    """Calculates Brier score at a single time using inverse-probability of
    censoring with the Kaplan Meier estimator. The constructor of this class
    takes a single time and a set of samples for performing the km estimation.
    Performing km estimation at construction prevents these estimates from
    fluxuating with each batch. When creating a class instance, you can pass in
    the times and events of your training and validation data to create these
    estimates.

    Parameters
    ----------
    labels : bool
        An N x 2 float32 tensor where event or last followup times are in
        the first column and event indicators are in the second column. Used
        for km estimation at construction.
    survival : function
        Survival function that generates survival probabilities given model
        predictions and time values (e.g. losses.Exponential().survival).
        This function accepts y_pred (N x K tensor) and times (N x D) and
        returns an N x D tensor of survival probabilities.
    time : float32
        The scalar time used in calculating the Brier score.
    name : string
        The name of the class instance. Default value 'brier'.

    Attributes
    ----------
    score : float32
        The current scalar brier score unnormalized by the number of samples.
    time : float32
        The scalar time to calculate the brier score at.
    total : int
        The total number of samples seen (both censored and uncensored).
    km_t : tensor (float32)
        A tensor of time points from the km estimator in chronological order.
    km_s : tensor (float32)
        A tensor of km estimated survival function values at times km_t.
    survival : function
        The survival function passed during initialization.

    See Also
    --------
    km : generates Kaplan Meier estimate.
    km_eval : evaluates Kaplan Meier estimator at specific times.
    """

    def __init__(self, labels, survival, time, name="brier", **kwargs):
        super().__init__(name=name, **kwargs)

        # km estimation with negated labels
        masked, _ = mask(labels)
        t, events = unstack_labels(masked)
        km_t, km_s = km(stack_labels(t, tf.math.logical_not(events)))

        # initialize variables
        self.score = self.add_weight(
            name="score", initializer="zeros", dtype=tf.float32
        )
        self.time = self.add_weight(name="time", initializer="zeros", dtype=tf.float32)
        self.time.assign(time)
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.int32)
        self.km_t = self.add_weight(
            name="km_t", initializer="zeros", dtype=tf.float32, shape=tf.size(km_s)
        )
        self.km_t.assign(km_t)
        self.km_s = self.add_weight(
            name="km_s", initializer="zeros", dtype=tf.float32, shape=tf.size(km_s)
        )
        self.km_s.assign(km_s)
        self.survival = survival

    def reset_state(self):
        self.score.assign(0.0)
        self.total.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):  # pragma: no cover
        """Calculates the unnormalized brier score for a set of samples and
        updates the internal state of scores and total samples.

        Parameters
        ----------
        y_true : tensor (float32)
            An N x 2 float32 tensor where event or last followup times are in
            the first column and event indicators are in the second column.
        y_pred : tensor (float32)
            An N x K tensor predicted by the network and used by self.survival
            to generate probabilities for individualized survival functions.
        """

        # generate survival probabilities at time self.time
        prob = self.survival(y_pred, self.time * tf.ones(tf.shape(y_pred)[0]))

        # calculate brier thresholded by t
        update = _brier(y_true, prob, self.time, self.km_t, self.km_s)

        # update state
        self.score.assign_add(update)
        self.total.assign_add(tf.shape(y_pred)[0])

    def result(self):  # pragma: no cover
        return self.score / tf.cast(self.total, tf.float32)


class IntegratedBrier(tf.keras.metrics.Metric):
    """Calculates the integrated Brier score over a range of times using
    inverse-probability of censoring with the Kaplan Meier estimator and
    trapezoidal integration. The constructor of this class takes a set of
    samples for performing the km estimation and a desired time range for the
    integration. If the time range is not provided it will be calculated from
    the samples used for km estimation. Performing km estimation at
    construction prevents these estimates from fluxuating with each batch. When
    creating a class instance, you can pass in the times and events of your
    training and validation data to create these estimates and time ranges.

    Parameters
    ----------
    labels : tensor (float32)
        An N x 2 float32 tensor where event or last followup times are in
        the first column and event indicators are in the second column. Used
        for km estimation at construction.
    times : tensor (float32)
        The scalar time used in calculating the Brier score. Default value
        'None' will cause this range to be estimated from the labels input.
    survival : function
        Survival function that generates survival probabilities given model
        predictions and time values (e.g. losses.Exponential().survival).
        This function accepts y_pred (N x K tensor) and times (N x D) and
        returns an N x D tensor of survival probabilities.
    name : string
        The name of the class instance. Default value 'brier'.

    Returns
    -------
    score : tensor (float32)
        Integrated Brier Score calculated over self.times. Score is in interval
        [0, inf] where 0 is a perfect score.

    Attributes
    ----------
    scores : variable (float32)
        The current scalar brier scores unnormalized by the number of samples
        and calculated at times.
    times : variable (float32)
        The scalar time to calculate the brier score at.
    total : variable (int32)
        The total number of samples seen (both censored and uncensored).
    km_t : variable (float32)
        The time points from the km estimator in chronological order.
    km_s : variable (float32)
        The km estimated survival function values at times km_t.
    survival : function
        The survival function passed during initialization.

    See Also
    --------
    km : generates Kaplan Meier estimate.
    km_eval : evaluates Kaplan Meier estimator at specific time.
    """

    def __init__(self, labels, survival, times=None, name="integratedbrier", **kwargs):
        super().__init__(name=name, **kwargs)

        # km estimation with negated labels
        masked, keep = mask(labels)
        t, events = unstack_labels(masked)
        km_t, km_s = km(stack_labels(t, tf.math.logical_not(events)))

        # calculate times if not provided
        if times is None:
            times = tf.sort(tf.unique(t)[0])

        # initialize variables
        self.scores = self.add_weight(
            name="score", initializer="zeros", shape=(tf.size(times)), dtype=tf.float32
        )
        self.times = self.add_weight(
            name="times", initializer="zeros", shape=(tf.size(times)), dtype=tf.float32
        )
        self.times.assign(times)
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.int32)
        self.km_t = self.add_weight(
            name="km_t", initializer="zeros", dtype=tf.float32, shape=tf.size(km_s)
        )
        self.km_t.assign(km_t)
        self.km_s = self.add_weight(
            name="km_s", initializer="zeros", dtype=tf.float32, shape=tf.size(km_s)
        )
        self.km_s.assign(km_s)
        self.survival = survival

    def reset_state(self):
        self.scores.assign(tf.zeros(tf.size(self.scores), tf.float32))
        self.total.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):  # pragma: no cover
        """Calculates the unnormalized integrated brier score for a set of
        samples and updates the internal state of scores and total samples.

        Parameters
        ----------
        y_true : float
            An N x 2 float32 tensor where event or last followup times are in
            the first column and event indicators are in the second column.
        y_pred : float32
            An N x K tensor predicted by the network and used by self.survival
            to generate probabilities for individualized survival functions.
        """

        # get number of samples, number of times
        N = tf.shape(y_pred)[0]
        D = tf.size(self.times)

        # generate survival probabilities at self.times for each sample
        probs = self.survival(y_pred, tf.tile(tf.reshape(self.times, [1, D]), [N, 1]))

        # calculate brier thresholded by t
        update = tf.map_fn(
            lambda x: _brier(y_true, x[0], x[1], self.km_t, self.km_s),
            (tf.transpose(probs), self.times),
            fn_output_signature=tf.float32,
        )

        # update state
        self.scores.assign_add(update)
        self.total.assign_add(N)

    def result(self):  # pragma: no cover
        # calculate time differences and score trapezoid midpoints
        delta_t = self.times[1:] - self.times[:-1]
        midpoints = (self.scores[1:] + self.scores[:-1]) / 2.0
        weight = self.times[-1] - self.times[0]

        return (
            tf.reduce_sum(tf.multiply(delta_t, midpoints))
            / tf.cast(self.total, tf.float32)
            / weight
        )
