import tensorflow as tf


class ThresholdMetric(tf.keras.metrics.Metric):
    """A generic binary classification metric class.

    This is subclassed by BalancedAccracy, F1, Mcc, Sensitivity, and Specificity
    metrics. It maintains a state of TP/FP/TN/FN that are used to calculate
    the aforementioned metrics. This is initialized with a single scalar threshold
    in the range [0,1] to transform predictions into binary labels. Sample weights
    are not currently supported.

    Parameters
    ----------
    name : string
        The name of the class instance.
    threshold : float
        A single float threshold in [0,1] used to determine the truth value
        of the predictions.

    Attributes
    ----------
    tp : int
        Running count of the true positives.
    fp : int
        Running count of the false positives.
    tn : int
        Running count of the true negatives.
    fn : int
        Running count of the false negatives.
    """

    def __init__(self, threshold, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype=tf.float32)
        self.tn = self.add_weight(name="tn", initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype=tf.float32)
        self.threshold = self.add_weight(
            name="threshold", initializer="zeros", dtype=tf.float32
        )
        self.threshold.assign(threshold)

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.tn.assign(0)
        self.fn.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates metric state containing comprised of TP/FP/TN/FN.

        Accepts as input labels `y_true` and predictions `y_pred`. Sample weights are
        not currently supported.

        Parameters
        ----------
        y_pred : float
            A two-dimensional float tensor containing predictions. If one column,
            `y_pred` is assumed to contain positive class predictions. If two
            columns, `y_pred` is assumed to contain one-hot encoded predictions.
        y_true : float
            A two-dimensional float tensor containing ground truth labels. If one
            column, `y_pred` is assumed to contain positive class predictions. If
            two columns, `y_pred` is assumed to contain one-hot encoded predictions.
        """

        # quantize to avoid problem with setting 'threshold' from tensor
        q1 = tf.cast(tf.cast(y_true[:, -1], tf.float32) > self.threshold, tf.float32)
        q2 = tf.cast(tf.cast(y_pred[:, -1], tf.float32) > self.threshold, tf.float32)

        # update tp, fp, tn, fn counts
        tp = tf.reduce_sum(tf.reshape(q1 * q2, [-1]))
        fp = tf.reduce_sum(tf.reshape((1.0 - q1) * q2, [-1]))
        tn = tf.reduce_sum(tf.reshape((1.0 - q1) * (1.0 - q2), [-1]))
        fn = tf.reduce_sum(tf.reshape(q1 * (1.0 - q2), [-1]))

        # update state variables
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.tn.assign_add(tn)
        self.fn.assign_add(fn)


class Balanced(ThresholdMetric):
    r"""A stateful metric class for binary balanced accuracy.

    This metric can be used as a stateless callable `Balanced()(y_true, y_pred)`,
    or as a stateful metric that aggregates results over multiple calls to
    `update_state(y_true, y_pred)` The current state can be acquired using `result()`,
    and the state can be reset using `reset_state()`. The balanced accuracy is
    the weighted average of sensitivity and specificity:

    .. math:: Balanced = \frac{1}{2} \left( \frac{TP}{TP+FN} + \frac{TN}{TN+FP} \right)

    Parameters
    ----------
    threshold : float
        A single float threshold in [0,1] used to determine the truth value
        of the predictions.
    name : string
        The name of the class instance.

    Returns
    -------
    balanced : float
        The balanced accuracy. Range [0, 1].
    """

    def __init__(self, threshold=0.5, name="balanced", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):  # pragma: no cover
        return 0.5 * (self.tp / (self.tp + self.fn) + self.tn / (self.tn + self.fp))


class F1(ThresholdMetric):
    r"""A stateful metric class for F1 score.

    This metric can be used as a stateless callable `F1()(y_true, y_pred)`,
    or as a stateful metric that aggregates results over multiple calls to
    `update_state(y_true, y_pred)`. The current state can be acquired using `result()`,
    and the state can be reset using reset_state(). The F1 score is defined as:

    .. math:: F1 = \frac{TP}{TP + \frac{1}{2} \left( FP + FN \right)}

    Parameters
    ----------
    threshold : float
        A single float threshold in [0,1] used to determine the truth value
        of the predictions.
    name : string
        The name of the class instance.

    Returns
    -------
    f1 : float
        The F1 score. Range [0, 1].
    """

    def __init__(self, threshold=0.5, name="f1", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):  # pragma: no cover
        return self.tp / (self.tp + 0.5 * (self.fp + self.fn))


class Mcc(ThresholdMetric):
    r"""A stateful metric class for binary Matthews' correlation coefficient.

    This metric can be used as a stateless callable `Mcc()(y_true, y_pred)`,
    or as a stateful metric that aggregates results over multiple calls to
    `update_state(y_true, y_pred)`. The current state can be acquired using `result()`,
    and the state can be reset using `reset_state()`. Matthew's correlation
    coefficient is defined as:

    .. math:: MCC = \frac{(TP \cdot TN) - (FP \cdot FN)}{\sqrt{(TP + FP)(TP+FN)(TN+FP)(TN+FN)}}

    Parameters
    ----------
    threshold : float
        A single float threshold in [0,1] used to determine the truth value
        of the predictions.
    name : string
        The name of the class instance.

    Returns
    -------
    mcc : float
        The Matthew's correlation coefficient. Range [0, 1].
    """

    def __init__(self, threshold=0.5, name="mcc", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):  # pragma: no cover
        numerator = self.tp * self.tn - self.fp * self.fn
        denominator = (
            (self.tp + self.fp)
            * (self.tp + self.fn)
            * (self.tn + self.fp)
            * (self.tn + self.fn)
        )
        return numerator / tf.math.pow(denominator, 0.5)


class Sensitivity(ThresholdMetric):
    r"""A stateful metric class for sensitivity.

    This metric can be used as a stateless callable `Sensitivity()(y_true, y_pred)`,
    or as a stateful metric that aggregates results over multiple calls to
    `update_state(y_true, y_pred)`. The current state can be acquired using `result()`,
    and the state can be reset using `reset_state()`. Sensitivity is defined as:

    .. math:: Sensitivity = \frac{TP}{TP+FN}

    Parameters
    ----------
    threshold : float
        A single float threshold in [0,1] used to determine the truth value
        of the predictions.
    name : string
        The name of the class instance.

    Returns
    -------
    sensitivity : float
        The sensitivity. Range [0, 1].
    """

    def __init__(self, threshold=0.5, name="sensitivity", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):  # pragma: no cover
        return self.tp / (self.tp + self.fn)


class Specificity(ThresholdMetric):
    r"""A stateful metric class for specificity.

    This metric can be used as a stateless callable `Specificity()(y_true, y_pred)`,
    or as a stateful metric that aggregates results over multiple calls to
    `update_state(y_true, y_pred)`. The current state can be acquired using `result()`,
    and the state can be reset using `reset_state()`. Specificity is defined as:

    .. math:: Specificity = \frac{TN}{TN+FP}

    Parameters
    ----------
    threshold : float
        A single float threshold in [0,1] used to determine the truth value
        of the predictions.
    name : string
        The name of the class instance.

    Returns
    -------
    specificity : float
        The specificity. Range [0, 1].
    """

    def __init__(self, threshold=0.5, name="specificity", **kwargs):
        super().__init__(threshold=threshold, name=name, **kwargs)

    def result(self):  # pragma: no cover
        return self.tn / (self.tn + self.fp)
