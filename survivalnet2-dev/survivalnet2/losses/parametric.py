import tensorflow as tf
from tensorflow import math
from survivalnet2.data.labels import unstack_labels


class Parametric:
    """This class implements a generic framework for all models based on parametric
    distributions. It is subclassed by specific parametric models like Exponential, Weibull,
    and Gompertz. This class includes functions to calculate the hazard, survival probability,
    and a loss function (negative log likelihood) that is common to all parametric models.
    """

    def expected(self, params):
        """Calculates the expected event time given distribution parameters.

        Parameters
        ----------
        params : float
            A N x K tensor output from a network where N is the sample dimension
            and K is the parameter dimension. Conditioning paramaters to appropriate
            ranges is the  responsibility of the network.

        Returns
        -------
        times : float
            A 1D vector containing the expected event times for the N samples.
        """

        raise NotImplementedError("Expected function is not implemented.")

    def median(self, params):
        """Calculates the median event time given distribution parameters.

        Parameters
        ----------
        params : float
            A N x K tensor output from a network where N is the sample dimension
            and K is the parameter dimension. Conditioning paramaters to appropriate
            ranges is the  responsibility of the network.

        Returns
        -------
        times : float
            A 1D vector containing the median event times for the N samples.
        """

        raise NotImplementedError("Expected function is not implemented.")

    def hazard(self, params, times):
        """Calculates the hazard given parameters for a distribution and times.
        Must be implemented in subclasses.

        Parameters
        ----------
        params : float
            A N x K tensor output from a network where N is the sample dimension
            and K is the parameter dimension. Conditioning paramaters to appropriate
            ranges is the  responsibility of the network.
        times : float
            A 1-D vector containing times to evalaute the hazard function at.

        Returns
        -------
        hazards : float
            A 1D vector containing the hazards calculated at
        """

        log_hazard = self._log_hazard(params, times)
        haz = math.exp(log_hazard)

        return haz

    def survival(self, params, times):
        """Calculates the survival probabilities given parameters for a distribution and
        times. Must be implemented in subclasses.

        Parameters
        ----------
        params : float
            A N x K tensor output from a network where N is the sample dimension
            and K is the parameter dimension. Conditioning paramaters to appropriate
            ranges is the  responsibility of the network.
        times : float
            A 1-D vector containing times to evalaute the hazard function at.

        Returns
        -------
        lmbda : float
            A 1D vector containing the calculated survival probabilities.
        """

        log_probs = self._log_survival(params, times)
        probs = math.exp(log_probs)

        return probs

    def _log_hazard(self, params, times):
        """Function to avoid numeric instability that arises from taking the log of
        exponents. Overwrite in subclass to provide a simplified/stable expression for
        log hazard. Used in loss calculation.
        """

        hazard = self.hazard(params, times)
        log_hazard = math.log(hazard)

        return log_hazard

    def _log_survival(self, params, times):
        """Function to avoid numeric instability that arises from taking the log of
        exponents. Overwrite in subclass to provide a simplified/stable expression for
        log survival. Used in loss calculation.
        """

        probs = self.survival(params, times)
        log_probs = math.log(probs)

        return log_probs

    def _ntimes(self, times):
        """Infer the number of times contained in the 'times' input to hazard /
        survival functions. In order to vectorize across times, the number of times
        must be inferred from the shape.

        Parameters
        ----------
        times : float
            An N x D, N x 1, or N tensor containing times to evaluate the hazard
            or survival function at. N in this case is the number of parameter
            values to evaluate these functions at, and is determined by the size
            of 'params'. 'times' can include a single time point (N or N x 1), or
            D time points (N x D).

        Returns
        -------
        D : int
            A scalar indicating the number of times inferred from the shape of
            'times'.
        """

        # get number of times
        ntimes = tf.cond(tf.rank(times) > 1, lambda: tf.shape(times)[-1], lambda: 1)

        return ntimes

    def negative_log_likelihood(self, y_true, y_pred):
        """The universal negative log likelihood loss function used while optimizing
        all parametric models.

        Parameters
        ----------
        y_true : float
            N x 2 tensor containing OSR labels with two columns (times, events).
        y_pred : float
            N x K tensor of predicted distribution parameters from model
            (e.g. lambda in an exponential distribution). The second dimension
            depends on how many parameters the chosen distribution requires.

        Returns
        -------
        -log_likelihood : float
            Negative log likelihood calculated based on inputs.
        """

        # Unstack labels
        times, events = unstack_labels(y_true)

        # reshape times, events for correct broadcasting
        times = tf.reshape(times, [-1, 1])
        events = tf.reshape(events, [-1, 1])

        # Sample input dimension
        N = tf.size(times)

        # Cast events from bool to float32
        events = tf.cast(events, "float32")

        # Calculate log hazard
        log_hazard = self._log_hazard(y_pred, times)

        # Calculate log survival probabilities
        log_probs = self._log_survival(y_pred, times)

        # Calculate log likelihood
        log_likelihood = math.add(
            tf.reduce_sum(events * log_hazard), tf.reduce_sum(log_probs)
        )

        return -log_likelihood / tf.cast(N, tf.float32)


class Exponential(Parametric):
    """This class implements the exponential distribution. It is the simplest
    of the parametric distributions, using only one parameter (lambda) and having
    a constant hazard function.
    """

    def expected(self, params):
        return 1 / params

    def median(self, params):
        return math.log(2.0) / params

    def hazard(self, params, times):
        # get number of times
        ntimes = self._ntimes(times)

        # reshape to enable vectorization along both parameters and times
        lmbda = tf.reshape(params, [-1, 1])
        lmbda = lmbda * tf.ones([1, ntimes], times.dtype)

        return lmbda

    def _log_survival(self, params, times):
        # get number of times
        ntimes = self._ntimes(times)

        # reshape to enable vectorization along both parameters and times
        lmbda = tf.reshape(params, [-1, 1])
        times = tf.reshape(times, [tf.size(lmbda), ntimes])

        # calculate log probabilities
        log_probs = -lmbda * times

        return log_probs


class Weibull(Parametric):
    """The Weibull model includes scale (lambda) and shape (p) parameters
    that are strictly positive.
    """

    def expected(self, params):
        # unstack scale, shape parameters
        lmbda, p = tf.unstack(params, axis=1)

        return 1 / lmbda * tf.math.exp(tf.math.lgamma(1 + 1 / p))

    def median(self, params):
        # unstack scale, shape parameters
        lmbda, p = tf.unstack(params, axis=1)

        return 1 / lmbda * math.pow(math.log(2.0), 1 / p)

    def hazard(self, params, times):
        # get number of times
        ntimes = self._ntimes(times)

        # unstack distribution parameters
        lmbda, p = tf.unstack(params, axis=1)

        # reshape to enable vectorization along both parameters and times
        lmbda = tf.reshape(lmbda, [-1, 1])
        p = tf.reshape(p, [-1, 1])
        times = tf.reshape(times, [tf.size(lmbda), ntimes])

        # calculate hazards
        hazard = math.pow(lmbda, p) * p * math.pow(times, p - 1)

        return hazard

    def _log_survival(self, params, times):
        # get number of times
        ntimes = self._ntimes(times)

        # unstack distribution parameters
        lmbda, p = tf.unstack(params, axis=-1)

        # reshape to enable vectorization along both parameters and times
        lmbda = tf.reshape(lmbda, [-1, 1])
        p = tf.reshape(p, [-1, 1])
        times = tf.reshape(times, [tf.size(lmbda), ntimes])

        # calculate log probabilities
        log_probs = -math.pow(lmbda * times, p)

        return log_probs


class Gompertz(Parametric):
    """The Gompertz model features an exponentially increasing rate. It
    has two parameters, one for shape (beta) and one for rate (alpha).
    Notes: This is not the Gompertz-Makeham model which adds a constant
    third parameter to the hazard function. The exponential integral
    function used to calculate expected event time is not defined correctly
    in tensorflow (returns 'nan' for negative values) so we use the Swamee
    Ohija approximation instead.
    """

    # approximation to exponential integral E1
    def _swamee_ohija(self, x):
        A = math.log((0.56146 / x + 0.65) * (1 + x))
        B = math.pow(x, 4.0) * math.exp(7.7 * x) * math.pow(2.0 + x, 3.7)
        return math.pow(math.pow(A, -7.7) + B, -0.13)

    def expected(self, params):
        # unstack scale, shape parameters
        alpha, beta = tf.unstack(params, axis=1)

        return self._swamee_ohija(alpha / beta) * math.exp(alpha / beta) / beta

    def median(self, params):
        # unstack scale, shape parameters
        alpha, beta = tf.unstack(params, axis=1)

        return math.log(1.0 + beta / alpha * math.log(2.0)) / beta

    def _log_hazard(self, params, times):
        # get number of times
        ntimes = self._ntimes(times)

        # unstack distribution parameters
        alpha, beta = tf.unstack(params, axis=1)

        # reshape to enable vectorization along both parameters and times
        alpha = tf.reshape(alpha, [-1, 1])
        beta = tf.reshape(beta, [-1, 1])
        times = tf.reshape(times, [tf.size(alpha), ntimes])

        # calculate log hazard
        log_hazard = math.log(alpha) + beta * times

        return log_hazard

    def _log_survival(self, params, times):
        # get number of times
        ntimes = self._ntimes(times)

        # unstack distribution parameters
        alpha, beta = tf.unstack(params, axis=-1)

        # reshape to enable vectorization along both parameters and times
        alpha = tf.reshape(alpha, [-1, 1])
        beta = tf.reshape(beta, [-1, 1])
        times = tf.reshape(times, [tf.size(alpha), ntimes])

        # calculate log probabilities
        log_probs = -alpha / beta * (math.exp(beta * times) - 1)

        return log_probs
