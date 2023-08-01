import numpy as np
import os
import pandas as pd
import pytest
import tensorflow as tf
from survivalnet2.losses.cox import cox, efron
from survivalnet2.losses.parametric import Exponential, Gompertz, Weibull

# execute eagerly to get coverage inside @tf.function
tf.config.run_functions_eagerly(True)

time_low = 1e-10
time_high = 10.0
param_low = 1e-10


def times(shape, low=time_low, high=time_high):
    return np.random.uniform(time_low, time_high, shape).astype(np.float32)


def generate_parameters(n, k, low=param_low):
    return tuple(
        np.random.uniform(low=low, size=(n)).astype(np.float32)
        for value in np.arange(0, k)
    )


def generate_exponential_data(n):
    # set seed
    np.random.seed(9999)

    # generate individualized survival function parameters in range [0, 1]
    lambda_i = generate_parameters(500, 1)

    # calculate survival times and survival function values
    t_i = np.random.exponential(1.0 / lambda_i)

    # random censoring with uniform censoring times
    fraction = 0.2
    events = np.random.uniform(size=(n)) >= fraction
    t_i[~events] = t_i[~events] * np.random.uniform(size=(np.sum(~events)))

    y_pred = tf.reshape(tf.constant(lambda_i, tf.float32), [-1, 1])
    times = tf.constant(t_i, tf.float32)
    events = tf.constant(events, tf.bool)
    y_true = tf.stack([times, tf.cast(events, tf.float32)], axis=1)

    return y_true, y_pred


def generate_weibull_data(n):
    # set seed
    np.random.seed(9999)

    # generate individualized survival function parameters in range [0, 1]
    lambda_i, p = generate_parameters(500, 2)

    # calculate survival times and survival function values
    t_i = lambda_i * np.random.weibull(p)

    # random censoring with uniform censoring times
    fraction = 0.2
    events = np.random.uniform(size=(n)) >= fraction
    t_i[~events] = t_i[~events] * np.random.uniform(size=(np.sum(~events)))

    y_pred = tf.stack(
        [tf.constant(lambda_i, tf.float32), tf.constant(p, tf.float32)], axis=1
    )
    times = tf.constant(t_i, tf.float32)
    events = tf.constant(events, tf.bool)
    y_true = tf.stack([times, tf.cast(events, tf.float32)], axis=1)

    return y_true, y_pred


def evaluate_parametric(params, parametric, reference, atol=1e-9):
    # this function generalizes testing of parametric loss functions
    # for hazard, survival given generated parameters, a parametric
    # function, and a reference function (for comparison)

    # define sequences of time variables for testing correctness and broadcasting
    timeseq = [times(500), times((500, 1)), times((500, 10))]

    # test each time case for correctness and broadcasting
    for time in timeseq:
        tf.debugging.assert_near(
            parametric(params, time), reference(params, time), atol=atol
        )

    # define sequences of time variables for triggering errors
    # mismatches between parameter, time first dimension should raise error
    timeseq = [times(500 - 1), times((500 - 1, 1)), times((500 - 1, 10))]

    # calls which should invoke broadcasting errors
    with pytest.raises(tf.errors.InvalidArgumentError):
        for time in timeseq:
            parametric(params, time)


def test_cox_is_nan_if_all_nan():
    labels = np.full((3, 2), np.nan)
    preds = np.full((3, 1), np.nan)

    labels = tf.convert_to_tensor(labels, dtype="float")
    preds = tf.convert_to_tensor(preds, dtype="float")

    assert np.isnan(cox(labels, preds))


def test_cox_is_nan_if_all_censored():
    N = 24
    times = np.random.rand(N).reshape((N, 1))
    events = np.zeros((N, 1))

    labels = np.concatenate((times, events), axis=1)
    preds = np.random.rand(N).reshape((N, 1))

    labels = tf.convert_to_tensor(labels, dtype="float")
    preds = tf.convert_to_tensor(preds, dtype="float")

    assert np.isnan(cox(labels, preds))


def test_cox_value_if_all_ones():
    N = 24
    labels = np.full((N, 2), 1.0)
    preds = np.full((N, 1), 1.0)

    labels = tf.convert_to_tensor(labels, dtype="float")
    preds = tf.convert_to_tensor(preds, dtype="float")

    assert cox(labels, preds) == N * np.log(N)


def test_cox_on_real_data():
    """
    Data taken from : https://www.statsdirect.com/help/survival_analysis/cox_regression.htm
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_dir = os.path.join(root_dir, "tests", "test_data", "test_data_cox_loss.csv")

    values = pd.read_csv(file_dir, header=None).to_numpy()

    labels = values[:, 1:]
    preds = np.expand_dims(values[:, 0], -1)
    preds_all_ones = np.full((labels.shape[0], 1), 1.0)

    labels = tf.convert_to_tensor(labels, dtype="float")
    preds = tf.convert_to_tensor(preds, dtype="float")
    preds_all_ones = tf.convert_to_tensor(preds_all_ones, dtype="float")

    np.testing.assert_almost_equal(cox(labels, preds).numpy(), 203.737609, decimal=2)
    np.testing.assert_almost_equal(
        cox(labels, preds_all_ones).numpy(), 207.554801, decimal=2
    )


def test_efron_is_nan_if_all_nan():
    labels = np.full((3, 2), np.nan)
    preds = np.full((3, 1), np.nan)

    labels = tf.convert_to_tensor(labels, dtype="float")
    preds = tf.convert_to_tensor(preds, dtype="float")

    assert np.isnan(efron(labels, preds))


def test_efron_is_nan_if_all_censored():
    N = 24
    times = np.random.rand(N).reshape((N, 1))
    events = np.zeros((N, 1))

    labels = np.concatenate((times, events), axis=1)
    preds = np.random.rand(N).reshape((N, 1))

    labels = tf.convert_to_tensor(labels, dtype="float")
    preds = tf.convert_to_tensor(preds, dtype="float")

    assert np.isnan(efron(labels, preds))


def test_efron_cox_no_ties():
    N = 24
    times = np.arange(N).reshape((N, 1)) + 1
    events = np.ones((N, 1))

    labels = np.concatenate((times, events), axis=1)
    preds = np.random.rand(N).reshape((N, 1))

    labels = tf.convert_to_tensor(labels, dtype="float")
    preds = tf.convert_to_tensor(preds, dtype="float")

    tf.debugging.assert_near(efron(labels, preds), cox(labels, preds), atol=1e-9)


def test_efron_on_real_data():
    """
    Data taken from : https://www.statsdirect.com/help/survival_analysis/cox_regression.htm
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_dir = os.path.join(root_dir, "tests", "test_data", "test_data_cox_loss.csv")

    values = pd.read_csv(file_dir, header=None).to_numpy()

    labels = values[:, 1:]
    preds = np.expand_dims(values[:, 0], -1)
    preds_all_ones = np.full((labels.shape[0], 1), 1.0)

    labels = tf.convert_to_tensor(labels, dtype="float")
    preds = tf.convert_to_tensor(preds, dtype="float")
    preds_all_ones = tf.convert_to_tensor(preds_all_ones, dtype="float")

    np.testing.assert_almost_equal(efron(labels, preds).numpy(), 203.62216, decimal=2)
    np.testing.assert_almost_equal(
        efron(labels, preds_all_ones).numpy(), 207.43744, decimal=2
    )


def test_exponential_hazard():
    # evaluates exponential hazard functions for correctness, and correct broadcasting
    # for various argument/output sizing

    # note: Since the exponential hazard function does not depend on time values, and
    # only on the size of times (if rank=2), its broadcasting behavior differs slightly.
    # For this reason, we test cases individually and do not use 'test_parametric_function'

    # generate random exponential parameters
    y_pred = np.stack(generate_parameters(500, 1), axis=1)

    # create exponential loss
    exponential = Exponential()

    # the outputs of exponential.hazard are only conditional on the number of times provided

    # measure exponential hazard with correct sizing - scalar
    hazard = exponential.hazard(y_pred, times((1)))
    tf.debugging.assert_near(y_pred, hazard, atol=1e-9)

    # measure exponential hazard with correct sizing - 1D times
    hazard = exponential.hazard(y_pred, times((500)))
    tf.debugging.assert_near(y_pred, hazard, atol=1e-9)

    # measure exponential hazard with correct sizing - 2D singleton times
    hazard = exponential.hazard(y_pred, times((500, 1)))
    tf.debugging.assert_near(y_pred, hazard, atol=1e-9)

    # measure exponential hazard with correct sizing - 2D non-singleton times
    hazard = exponential.hazard(y_pred, times((500, 10)))
    tf.debugging.assert_near(y_pred * tf.ones([1, 10]), hazard, atol=1e-9)

    # mismatch between parameter, time sizes in first dimension should not raise error
    # for exponential, hazard is not conditional on time, so first dimension is inferred from
    # 'params' input to hazard, and second dimension of 'times' input
    hazard = exponential.hazard(y_pred, times((499)))
    tf.debugging.assert_near(y_pred, hazard, atol=1e-9)
    hazard = exponential.hazard(y_pred, times((499, 1)))
    tf.debugging.assert_near(y_pred, hazard, atol=1e-9)
    hazard = exponential.hazard(y_pred, times((499, 10)))
    tf.debugging.assert_near(y_pred * tf.ones([1, 10]), hazard, atol=1e-9)

    # coverage of log_hazard
    log_hazard = exponential._log_hazard(y_pred, times(500, 10))
    tf.debugging.assert_near(
        tf.math.log(y_pred) * tf.ones([1, 10]), log_hazard, atol=1e-9
    )


def test_exponential_survival():
    # evaluates exponential survival functions for correctness, and
    # argument/output sizing

    # generate random exponential parameters
    y_pred = np.stack(generate_parameters(500, 1), axis=1)

    # create exponential loss
    exponential = Exponential()

    # define exponential survival function
    def np_survival(y_pred, time):
        lambda_i = np.reshape(y_pred, [np.size(y_pred), -1])
        time = np.reshape(time, [np.size(lambda_i), -1])
        hazard = np.exp(-lambda_i * time)
        return hazard

    evaluate_parametric(y_pred, exponential.survival, np_survival)


def test_exponential_stats():
    # evaluates exponential distribution statistics for correctness, and
    # argument/output sizing

    # generate random exponential parameters
    y_pred = np.stack(generate_parameters(500, 1), axis=1)

    # create exponential loss
    exponential = Exponential()

    # evalaute expected values - 1 / lambda
    expected = exponential.expected(y_pred)
    tf.debugging.assert_near(expected, 1.0 / y_pred, atol=1e-9)

    # evaluate median values - log(2) / lambda
    median = exponential.median(y_pred)
    tf.debugging.assert_near(median, np.log(2.0) / y_pred, atol=1e-9)


def test_weibull_hazard():
    # evaluates Weibull hazard functions for correctness, and correct broadcasting
    # for various argument/output sizing

    # generate random weibull parameters
    y_pred = np.stack(generate_parameters(500, 2), axis=1)

    # create weibull loss
    weibull = Weibull()

    # define numpy weibull hazard for reference
    def np_hazard(y_pred, time):
        lambda_i = y_pred[:, 0]
        p = y_pred[:, 1]
        lambda_i = np.reshape(lambda_i, [np.size(lambda_i), -1])
        p = np.reshape(p, [np.size(p), -1])
        time = np.reshape(time, [np.size(p), -1])
        hazard = np.power(lambda_i, p) * p * np.power(time, p - 1)
        return hazard

    evaluate_parametric(y_pred, weibull.hazard, np_hazard)


def test_weibull_survival():
    # evaluates Weibull survival functions for correctness, and correct broadcasting
    # for various argument/output sizing

    # generate random weibull parameters
    y_pred = np.stack(generate_parameters(500, 2), axis=1)

    # create weibull loss
    weibull = Weibull()

    # define numpy weibull hazard for reference
    def np_survival(y_pred, time):
        lambda_i = y_pred[:, 0]
        p = y_pred[:, 1]
        lambda_i = np.reshape(lambda_i, [np.size(lambda_i), -1])
        p = np.reshape(p, [np.size(p), -1])
        time = np.reshape(time, [np.size(p), -1])
        log_survival = -np.power(lambda_i * time, p)
        return np.exp(log_survival)

    evaluate_parametric(y_pred, weibull.survival, np_survival)


def test_weibull_stats():
    # evaluates weibull distribution statistics for correctness, and
    # argument/output sizing

    # generate random weibull parameters - avoid overflowing gamma func.
    y_pred = np.stack(generate_parameters(500, 2, low=3e-2), axis=1)

    # create weibull loss
    weibull = Weibull()

    # evalaute expected values - numpy lacks a gamma function alternative
    expected = weibull.expected(y_pred)
    reference = 1 / y_pred[:, 0] * np.exp(tf.math.lgamma(1.0 + 1.0 / y_pred[:, 1]))
    ignore = tf.math.logical_or(tf.math.is_inf(expected), tf.math.is_inf(reference))
    tf.debugging.assert_near(expected[~ignore], reference[~ignore], atol=1e-5)

    # evaluate median
    median = weibull.median(y_pred)
    reference = 1 / y_pred[:, 0] * np.power(np.log(2.0), 1 / y_pred[:, 1])
    tf.debugging.assert_near(median, reference, atol=1e-9)


def test_gompertz_hazard():
    # evaluates Gompertz hazard functions for correctness, and correct broadcasting
    # for various argument/output sizing

    # generate random gompertz parameters
    y_pred = np.stack(generate_parameters(500, 2), axis=1)

    # create gompertz loss
    gompertz = Gompertz()

    # define numpy gompertz hazard for reference
    def np_hazard(y_pred, time):
        alpha = y_pred[:, 0]
        beta = y_pred[:, 1]
        alpha = np.reshape(alpha, [np.size(alpha), -1])
        beta = np.reshape(beta, [np.size(beta), -1])
        time = np.reshape(time, [np.size(beta), -1])
        hazard = np.exp(np.log(alpha) + beta * time)
        return hazard

    evaluate_parametric(y_pred, gompertz.hazard, np_hazard)


def test_gompertz_survival():
    # evaluates gompertz survival functions for correctness, and correct broadcasting
    # for various argument/output sizing

    # generate random gompertz parameters
    y_pred = np.stack(generate_parameters(500, 2, low=3e-2), axis=1)

    # create gompertz loss
    gompertz = Gompertz()

    # define numpy gompertz survival for reference
    def np_survival(y_pred, time):
        alpha = y_pred[:, 0]
        beta = y_pred[:, 1]
        alpha = np.reshape(alpha, [np.size(alpha), -1])
        beta = np.reshape(beta, [np.size(beta), -1])
        time = np.reshape(time, [np.size(beta), -1])
        log_survival = -alpha / beta * (np.exp(beta * time) - 1.0)
        return np.exp(log_survival)

    evaluate_parametric(y_pred, gompertz.survival, np_survival, 1e-5)


def test_gompertz_stats():
    # evaluates gompertz distribution statistics for correctness, and
    # argument/output sizing

    # generate random gompertz parameters - avoid overflowing gamma func.
    y_pred = np.stack(generate_parameters(500, 2, low=3e-2), axis=1)

    # create gompertz loss
    gompertz = Gompertz()

    # evalaute expected values - relies on an approximation of Ei(x)
    def so_approx(x):
        A = np.log((0.56146 / x + 0.65) * (1 + x))
        B = np.power(x, 4.0) * np.exp(7.7 * x) * np.power(2.0 + x, 3.7)
        return np.power(np.power(A, -7.7) + B, -0.13)

    expected = gompertz.expected(y_pred)
    reference = (
        so_approx(y_pred[:, 0] / y_pred[:, 1])
        * np.exp(y_pred[:, 0] / y_pred[:, 1])
        / y_pred[:, 1]
    )
    tf.debugging.assert_near(expected, reference, atol=1e-5)

    # evaluate median
    median = gompertz.median(y_pred)
    reference = np.log(1.0 + y_pred[:, 1] / y_pred[:, 0] * np.log(2.0)) / y_pred[:, 1]
    tf.debugging.assert_near(median, reference, atol=1e-9)
