from survivalnet2.data.labels import mask, unstack_labels
import tensorflow as tf


class Dcal(tf.keras.metrics.Metric):
    """Calculates the d-calibration metric for right-censored data. This
    compares the distribution of values from individualized survival functions
    at the event or censoring times which for a well calibrated model should
    be close to uniform. Returns a chi-square statistic between the histogram
    of binned survival function values and an ideal uniform distribution.

    Parameters
    ----------
    survival : function
        Survival function that generates survival probabilities given model
        predictions and time values (e.g. losses.Exponential().survival).
        This function accepts y_pred (N x K tensor) and times (N x D) and
        returns an N x D tensor of survival probabilities.
    name : string
        The name of the class instance.
    bins : int
        The number of bins to use for histogram analysis. Default value 10.

    Attributes
    ----------
    bins : int
        The number of bins to use for the survival function value histogram.
    n_k : float32
        The cumulative un-normalized histogram of survival function values.
    survival : function
        The survival function passed during initialization.

    Notes
    -----
    Haider, Humza, et al. "Effective Ways to Build and Evaluate Individual
    Survival Distributions." J. Mach. Learn. Res. 21.85 (2020): 1-63.
    """

    def __init__(self, survival, name="dcalibration", bins=10, **kwargs):
        super().__init__(name=name, **kwargs)
        self.bins = self.add_weight(name="bins", initializer="zeros", dtype=tf.int32)
        self.n_k = self.add_weight(
            name="n_k", initializer="zeros", shape=(bins), dtype=tf.float32
        )
        self.bins.assign(bins)
        self.survival = survival

    def reset_state(self):
        self.n_k.assign(tf.zeros((self.bins), dtype=tf.float32))

    def update_state(self, y_true, y_pred, sample_weight=None):  # pragma: no cover
        """Updating the internal state increments the histogram for a batch of
        samples.

        Parameters
        ----------
        y_true : float
            An N x 2 float32 tensor where event or last followup times are in
            the first column and event indicators are in the second column.
        y_pred : float
            An N x K tensor predicted by the network and used by self.survival
            to generate probabilities for individualized survival functions.
        """

        # mask and unpack the labels
        masked, keep = mask(y_true)
        times, events = unstack_labels(masked)

        # generate survival probabilities
        prob = self.survival(y_pred, times[:, tf.newaxis])

        # reshape prob from [N, 1] to [N]
        prob = tf.reshape(prob, [-1])

        # Mask the predictions
        prob = tf.boolean_mask(prob, keep, axis=0)

        # shortcuts for bin width and number of bins
        width = 1.0 / tf.cast(self.bins, tf.float32)
        b = self.bins

        # calculate bin assignments using individualized survival functions
        k_i = tf.minimum(tf.cast(tf.math.floor(prob / width), tf.int32), b - 1)

        # histogram for uncensored samples
        unique, _, counts = tf.unique_with_counts(tf.boolean_mask(k_i, events))
        n_event = tf.scatter_nd(
            tf.expand_dims(unique, axis=1), tf.cast(counts, tf.float32), shape=[b]
        )

        # calculate left bin edes - needed for censored sample blurring
        b_k = tf.cast(tf.range(0, b), tf.float32) / tf.cast(b, tf.float32)

        # generate blur component for censored samples for first bin
        k_i = tf.boolean_mask(k_i, ~events)
        prob = tf.boolean_mask(prob, ~events)
        blur_first = 1.0 - tf.gather(b_k, k_i) / prob
        n_censored = tf.tensor_scatter_nd_add(
            tf.fill([b], 0.0), tf.expand_dims(k_i, axis=1), blur_first
        )

        # combine with blur component for censored samples for remaining bins
        blur_rest = width / prob
        update = tf.ones([tf.size(prob), 1], tf.int32) * tf.range(
            0, 10, 1, tf.int32
        ) < tf.expand_dims(k_i, axis=1)
        update = tf.cast(update, tf.float32) * tf.expand_dims(blur_rest, axis=1)
        n_censored = n_censored + tf.reduce_sum(update, axis=0)

        # update state
        self.n_k.assign_add(n_event + n_censored)

    def result(self):  # pragma: no cover
        # normalize histogram
        normalized = self.n_k / tf.reduce_sum(self.n_k)

        # bin width
        width = 1.0 / tf.cast(self.bins, tf.float32)

        # calculate chi-square with uniform distribution
        chisq = tf.reduce_sum((normalized - width) * (normalized - width) / width)

        return chisq
