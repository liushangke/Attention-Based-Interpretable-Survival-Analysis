import tensorflow as tf
from survivalnet2.data.labels import mask, unstack_labels


def _concordance_event(delta_t, delta_r, events):  # pragma: no cover
    # calculates concordance statistics for a sample with event=True

    prod = tf.multiply(delta_r, delta_t)

    # comparison to right-censored samples
    orderable_rc = tf.logical_not(events) & (delta_t >= 0.0)
    concordant_rc = orderable_rc & (delta_r < 0.0)
    discordant_rc = orderable_rc & (delta_r > 0.0)
    tiedx_rc = orderable_rc & (delta_r == 0.0)

    # comparison to non-censored samples
    orderable = events & (delta_t != 0.0)
    concordant = events & (prod < 0.0)
    discordant = events & (prod > 0.0)
    tiedx = orderable & (delta_r == 0.0)
    tiedy = events & ((delta_t == 0.0) & (delta_r != 0.0))
    tiedxy = events & ((delta_t == 0.0) & (delta_r == 0.0))

    # combine
    orderable = orderable_rc | orderable
    concordant = concordant_rc | concordant
    discordant = discordant_rc | discordant
    tiedx = tiedx_rc | tiedx

    return concordant, discordant, tiedx, tiedy, tiedxy


def _concordance_nonevent(delta_t, delta_r, events):  # pragma: no cover
    # calculates concordance statistics for a sample with event=False

    # comparison to non-censored samples
    orderable = events & (delta_t <= 0.0)
    concordant = orderable & (delta_r > 0.0)
    discordant = orderable & (delta_r < 0.0)
    tiedx = orderable & (delta_r == 0.0)
    tiedy = tf.zeros(tf.shape(orderable), tf.bool)
    tiedxy = tf.zeros(tf.shape(orderable), tf.bool)

    return concordant, discordant, tiedx, tiedy, tiedxy


def _concordance_loop(i, inputs, statistics):  # pragma: no cover
    # calculates concordance statistics for one sample in tf.while_loop

    # unpack inputs
    risks = inputs[0]
    times = inputs[1]
    events = inputs[2]
    c = statistics[0]
    d = statistics[1]
    tx = statistics[2]
    ty = statistics[3]
    txy = statistics[4]

    # compare sample i to subsequent samples (risks may have singleton dimension)
    delta_r = tf.squeeze(risks[i + 1 :] - risks[i])
    delta_t = times[i + 1 :] - times[i]

    # compute concordance statistics
    concordant, discordant, tiedx, tiedy, tiedxy = tf.cond(
        events[i],
        lambda: _concordance_event(delta_t, delta_r, events[i + 1 :]),
        lambda: _concordance_nonevent(delta_t, delta_r, events[i + 1 :]),
    )

    # reduce
    c = c + tf.reduce_sum(tf.cast(concordant, tf.int32))
    d = d + tf.reduce_sum(tf.cast(discordant, tf.int32))
    tx = tx + tf.reduce_sum(tf.cast(tiedx, tf.int32))
    ty = ty + tf.reduce_sum(tf.cast(tiedy, tf.int32))
    txy = txy + tf.reduce_sum(tf.cast(tiedxy, tf.int32))

    return i + 1, (risks, times, events), (c, d, tx, ty, txy)


@tf.function
def _concordance(risks, times, events, parallel=16):  # pragma: no cover
    # calculates concordance statistics used in ConcordanceMetric

    i = tf.constant(0, tf.int32)
    zero = tf.constant(0, tf.int32)
    state = [i, (risks, times, events), (zero, zero, zero, zero, zero)]
    condition = lambda x, y, z: x < tf.size(risks) - 1
    body = lambda x, y, z: _concordance_loop(x, y, z)
    i, inputs, statistics = tf.while_loop(condition, body, state)

    return statistics


class ConcordanceMetric(tf.keras.metrics.Metric):
    """This class implements a generic concordance metric that can be
    subclassed by the HarrellsC, SommersD, GoodmanKruskalGamma, and KendallTauA/B
    metrics. These metrics all consume the same concordance statistics. This
    class calculates concordance statistics given inputs ytrue=[times, events]
    and ypred including the number of concordant pairs, discordant pairs, and
    pairs with tied times, tied risks, and tied times and risks. Each subclass
    provides their own result function which calculates a unique statistic
    based on these values. This is a stateful metric that accumulates these
    quantities over multiple calls. It is vectorized and parallelizes across
    samples to reduce calculation time.

    ***Note: Calling this metric several times on groups of samples does not
    produce identical results to calling it on the combined groups. Samples
    are not compared across calls, rather, the concordance statistics are
    accumulated and combined across calls.

    Parameters
    ----------
    name : string
        The name of the class instance.
    parallel_iterations : int
        The number of parallel iterations to use in calculation. Default value
        is 16.

    Attributes
    ----------
    concordant : int
        The number of sample pairs with concordant predictions and times/events.
    discordant : int
        The number of sample pairs with discordant predictions and times/events.
    tiedrisks : int
        The number of pairs with tied predictions.
    tiedtimes : int
        The number of pairs with tied times (with events).
    tiedriskstimes : int
        The number of pairs with both tied risks and times (with events).
    parallel : int
        The number of parallel iterations to use in update_state.
    """

    def __init__(self, name, parallel_iterations=16, **kwargs):
        super().__init__(name=name, **kwargs)
        self.concordant = self.add_weight(
            name="concordant", initializer="zeros", dtype=tf.int32
        )
        self.discordant = self.add_weight(
            name="discordant", initializer="zeros", dtype=tf.int32
        )
        self.tiedrisks = self.add_weight(
            name="tiedrisks", initializer="zeros", dtype=tf.int32
        )
        self.tiedtimes = self.add_weight(
            name="tiedtimes", initializer="zeros", dtype=tf.int32
        )
        self.tiedriskstimes = self.add_weight(
            name="tiedriskstimes", initializer="zeros", dtype=tf.int32
        )
        self.parallel = self.add_weight(
            name="parallel", initializer="zeros", dtype=tf.int32
        )
        self.parallel.assign_add(parallel_iterations)

    def reset_state(self):
        self.concordant.assign(0)
        self.discordant.assign(0)
        self.tiedrisks.assign(0)
        self.tiedtimes.assign(0)
        self.tiedriskstimes.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):  # pragma: no cover
        # mask and unpack the labels
        masked, keep = mask(y_true)
        times, events = unstack_labels(masked)

        # Mask the prediction scores
        y_pred = tf.boolean_mask(y_pred, keep, axis=0)

        # calculate concordance statistics
        c, d, tx, ty, txy = _concordance(y_pred, times, events, parallel=self.parallel)

        # update state variables
        self.concordant.assign_add(c)
        self.discordant.assign_add(d)
        self.tiedrisks.assign_add(tx)
        self.tiedtimes.assign_add(ty)
        self.tiedriskstimes.assign_add(txy)


class HarrellsC(ConcordanceMetric):
    """Calculates Harrel's concordance index for right-censored data. Pairs of
    uncensored samples with tied times are ignored, and other pairs of
    samples with tied predictions are are counted as 1/2 correct.

    Returns
    -------
    cindex : float
      Proportion of concordant pairs among all orderable pairs. Range [0, 1].

    Notes
    -----
    Harrell FE Jr, Califf RM, Pryor DB, Lee KL, Rosati RA. Evaluating the yield of
    medical tests. JAMA. 1982;247(18):2543-2546.
    Reference R package:
    https://cran.r-project.org/web/packages/survival/vignettes/concordance.pdf
    """

    def __init__(self, name="harrellsc", parallel_iterations=16, **kwargs):
        super().__init__(name=name, parallel_iterations=parallel_iterations, **kwargs)

    def result(self):  # pragma: no cover
        return (
            (
                (self.concordant - self.discordant)
                / (self.concordant + self.discordant + self.tiedrisks)
            )
            + 1
        ) / 2


class SomersD(ConcordanceMetric):
    """Calculates Somer's d statistic for right-censored data. A scaled version
    of Harrell's concordance index with range [-1, 1]. Pairs of uncensored samples
    with tied times are ignored, and other pairs of samples with tied predictions
    are are counted as 1/2 correct.

    Returns
    -------
    d : float
      Proportion of concordant pairs among all orderable pairs scaled and
      centered at zero. Range [-1, 1].

    Notes
    -----
    Reference R package:
    https://cran.r-project.org/web/packages/survival/vignettes/concordance.pdf
    """

    def __init__(self, name="sommersd", parallel_iterations=16, **kwargs):
        super().__init__(name=name, parallel_iterations=parallel_iterations, **kwargs)

    def result(self):  # pragma: no cover
        return (self.concordant - self.discordant) / (
            self.concordant + self.discordant + self.tiedrisks
        )


class GoodmanKruskalGamma(ConcordanceMetric):
    """Calculates Goodman and Kruskal's gamma statistic for right-censored data.
    Tied times or risks are ignored.

    Returns
    -------
    gamma : float
      Proportion of concordant pairs among all orderable pairs scaled and
      centered at zero. Range [-1, 1].

    Notes
    -----
    Reference R package:
    https://cran.r-project.org/web/packages/survival/vignettes/concordance.pdf
    """

    def __init__(self, name="goodmankruskalgamma", parallel_iterations=16, **kwargs):
        super().__init__(name=name, parallel_iterations=parallel_iterations, **kwargs)

    def result(self):  # pragma: no cover
        return (self.concordant - self.discordant) / (self.concordant + self.discordant)


class KendallTauA(ConcordanceMetric):
    """Calculates Kendall's tau (a) for right-censored data. Pairs with tied
    times, tied risks, or both are counted as failures.

    Returns
    -------
    tau : float
      Proportion of concordant pairs among all orderable pairs scaled and
      centered at zero with tied risks and tied times pairs counting as
      errors. Range [-1, 1].

    See Also
    --------
    vectorized_concordance : Generates concordance data
    kendall_tau_b : b-version of Kendall's tau rank correlation.
    Notes
    -----
    Reference R package:
    https://cran.r-project.org/web/packages/survival/vignettes/concordance.pdf
    """

    def __init__(self, name="kendalltaua", parallel_iterations=16, **kwargs):
        super().__init__(name=name, parallel_iterations=parallel_iterations, **kwargs)

    def result(self):  # pragma: no cover
        return (self.concordant - self.discordant) / (
            self.concordant + self.discordant + self.tiedrisks + self.tiedtimes
        )


class KendallTauB(ConcordanceMetric):
    """Calculates Kendall's tau (b) for right-censored data. Pairs with tied
    times or tied risks are counted as failures (ties are symmetrized).

    Returns
    -------
    tau : float
      Proportion of concordant pairs among all orderable pairs scaled and centered
      at zero. Range [-1, 1].

    Notes
    -----
    Reference R package:
    https://cran.r-project.org/web/packages/survival/vignettes/concordance.pdf
    """

    def __init__(self, name="kendalltaub", parallel_iterations=16, **kwargs):
        super().__init__(name=name, parallel_iterations=parallel_iterations, **kwargs)

    def result(self):  # pragma: no cover
        return tf.cast(self.concordant - self.discordant, tf.float32) / tf.pow(
            tf.cast(
                (self.concordant + self.discordant + self.tiedrisks)
                * (self.concordant + self.discordant + self.tiedtimes),
                tf.float32,
            ),
            0.5,
        )
