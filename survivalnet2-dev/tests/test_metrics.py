import numpy as np
import tensorflow as tf
from survivalnet2.metrics.concordance import (
    HarrellsC,
    SomersD,
    GoodmanKruskalGamma,
    KendallTauA,
    KendallTauB,
)
from survivalnet2.estimators import km, km_eval
from survivalnet2.metrics.logrank import Logrank
from survivalnet2.metrics.brier import _brier, Brier, IntegratedBrier
from survivalnet2.metrics.dcal import Dcal
from survivalnet2.losses.parametric import Exponential


def generate_bland_altman_data():
    # data from Bland, Altman paper
    # Bland JM, Altma DG. The logrank test. BMJ. 2004 May 1;328(7447):1073.
    # doi: 10.1136/bmj.328.7447.1073. PMID: 15117797; PMCID: PMC403858.

    times = np.array(
        [
            6,
            13,
            21,
            30,
            31,
            37,
            38,
            47,
            49,
            50,
            63,
            79,
            80,
            82,
            82,
            86,
            98,
            149,
            202,
            219,
            10,
            10,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            20,
            24,
            24,
            25,
            28,
            30,
            33,
            34,
            35,
            37,
            40,
            40,
            40,
            46,
            48,
            70,
            76,
            81,
            82,
            91,
            112,
            181,
        ],
        np.float32,
    )
    events = np.array(
        [
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        np.float32,
    )
    labels = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
    )

    return tf.stack([times, events], axis=1), labels


def generate_concordance_data():
    # this example produces 2 concordant pairs, 3 discordant pairs, 16 pairs
    # with tied predictions, 3 pairs with tied times (w/events), and 3 pairs
    # with both tied predictions and tied times (w/events)

    # example problem to generate all types of concordances
    times = tf.constant([2, 2, 1, 2, 2, 2, 2, 1, 2, 1], tf.float32)
    events = tf.constant([0, 1, 0, 1, 1, 0, 0, 0, 1, 1], tf.float32)
    risks = tf.constant([1, 0, 1, 1, 1, 1, 1, 0, 1, 1], tf.float32)

    return times, events, risks


def generate_exponential_data():
    # set seed
    np.random.seed(9999)

    # generate individualized survival function parameters in range [0, 1]
    n = 500
    lambda_i = np.random.uniform(size=(n))

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


def test_harrellsc():
    # example problem to generate all types of concordances
    times, events, risks = generate_concordance_data()

    metric = HarrellsC()
    result = metric(tf.stack([times, events], axis=1), risks)
    tf.debugging.assert_near(result, (20 / 21) / 2, atol=1e-9)

    metric.reset_state()
    metric.update_state(tf.stack([times, events], axis=1), risks)
    tf.debugging.assert_near(metric.result(), (20 / 21) / 2, atol=1e-9)


def test_somersd():
    # example problem to generate all types of concordances
    times, events, risks = generate_concordance_data()

    metric = SomersD()
    result = metric(tf.stack([times, events], axis=1), risks)
    tf.debugging.assert_near(result, (2 - 3) / (2 + 3 + 16), atol=1e-9)

    metric.reset_state()
    metric.update_state(tf.stack([times, events], axis=1), risks)
    tf.debugging.assert_near(metric.result(), (2 - 3) / (2 + 3 + 16), atol=1e-9)


def test_goodmankruskalgamma():
    # example problem to generate all types of concordances
    times, events, risks = generate_concordance_data()

    metric = GoodmanKruskalGamma()
    result = metric(tf.stack([times, events], axis=1), risks)
    tf.debugging.assert_near(result, (2 - 3) / (2 + 3), atol=1e-9)

    metric.reset_state()
    metric.update_state(tf.stack([times, events], axis=1), risks)
    tf.debugging.assert_near(metric.result(), (2 - 3) / (2 + 3), atol=1e-9)


def test_kendalltaua():
    # example problem to generate all types of concordances
    times, events, risks = generate_concordance_data()

    metric = KendallTauA()
    result = metric(tf.stack([times, events], axis=1), risks)
    tf.debugging.assert_near(result, (2 - 3) / (2 + 3 + 16 + 3), atol=1e-9)

    metric.reset_state()
    metric.update_state(tf.stack([times, events], axis=1), risks)
    tf.debugging.assert_near(metric.result(), (2 - 3) / (2 + 3 + 16 + 3), atol=1e-9)


def test_kendalltaub():
    # example problem to generate all types of concordances
    times, events, risks = generate_concordance_data()

    metric = KendallTauB()
    result = metric(tf.stack([times, events], axis=1), risks)
    tf.debugging.assert_near(
        result,
        (2 - 3) / (tf.pow((2 + 3 + 16) * (2 + 3 + 3), tf.constant(0.5))),
        atol=1e-9,
    )

    metric.reset_state()
    metric.update_state(tf.stack([times, events], axis=1), risks)
    tf.debugging.assert_near(
        metric.result(),
        (2 - 3) / (tf.pow((2 + 3 + 16) * (2 + 3 + 3), tf.constant(0.5))),
        atol=1e-9,
    )


def test_logrank():
    # generate data with hard cluster assignments
    y_true, labels = generate_bland_altman_data()
    labels = tf.one_hot(labels - 1, depth=2)

    # predict with all data - tests .update_state and .result
    metric = Logrank(k=2)
    result = metric(y_true, labels)
    tf.debugging.assert_near(result, 6.884727581210048, atol=1e-9)

    # reset state
    metric.reset_state()
    tf.debugging.assert_near(metric.observed, tf.zeros(2, tf.float32), atol=1e-9)
    tf.debugging.assert_near(metric.expected, tf.zeros(2, tf.float32), atol=1e-9)

    # test .update_state
    metric.update_state(y_true, labels)
    tf.debugging.assert_near(metric.result(), 6.884727581210048, atol=1e-9)


def test__brier_helper():
    # evaluates the _brier helper function using exponential data
    # this function is not traced by coverage if called from
    # Brier.update_state, so we evaluate it separately

    # create a perfect dataset
    y_true, _ = generate_bland_altman_data()
    time = np.median(y_true[:, 0])
    y_pred = np.ones_like(y_true[:, 0]).astype(np.float32)
    y_pred = np.logical_not(y_true[:, 1]).astype(np.float32)
    y_pred[y_true[:, 0] > time] = 1.0

    # generate km estimates
    km_t, km_s = km(y_true)

    # evalute _brier on perfect data
    result = _brier(y_true, y_pred, time, km_t, km_s)

    tf.print(result)

    tf.debugging.assert_near(result, 0.0, atol=1e-9)


def test_brier():
    # tests if the Brier score calculated on exponential data has
    # changed since the initial implementation

    # generate random exponential data
    y_true, y_pred = generate_exponential_data()

    # create exponential loss
    exponential = Exponential()

    # initialize metric
    metric = Brier(y_true, exponential.survival, 3.0)

    # check metric value from direct call
    result = metric(y_true, y_pred)
    tf.debugging.assert_near(result, 0.1573471, atol=1e-5)

    # check reset state
    metric.reset_state()
    tf.debugging.Assert(tf.equal(metric.score, 0.0), metric.score)
    tf.debugging.Assert(tf.equal(metric.total, 0), metric.total)

    # check update state
    metric.update_state(y_true, y_pred)
    tf.debugging.assert_near(metric.result(), 0.1573471, atol=1e-5)


def test_integrated_brier():
    # tests if the Brier score calculated on exponential data has
    # changed since the initial implementation

    # generate random exponential data
    y_true, y_pred = generate_exponential_data()

    # create exponential loss
    exponential = Exponential()

    # initialize metric - default times
    metric = IntegratedBrier(y_true, exponential.survival)

    # check metric value from direct call
    result = metric(y_true, y_pred)
    print(result)
    tf.debugging.assert_near(result, 0.012709122, atol=1e-5)

    # check reset state
    metric.reset_state()
    tf.debugging.Assert(all(tf.equal(metric.scores, 0.0)), metric.scores)
    tf.debugging.Assert(tf.equal(metric.total, 0), metric.total)

    # check update state
    metric.update_state(y_true, y_pred)
    tf.debugging.assert_near(metric.result(), 0.012709122, atol=1e-5)


def test_dcal():
    # generate random exponential data
    y_true, y_pred = generate_exponential_data()

    # create exponential loss
    exponential = Exponential()

    # initialize metric
    metric = Dcal(exponential.survival)

    # check metric value from direct call
    result = metric(y_true, y_pred)
    tf.debugging.assert_near(result, 0.033155613, atol=1e-9)

    # check reset state
    metric.reset_state()
    tf.debugging.assert_near(metric.n_k, tf.zeros((metric.bins), tf.float32), atol=1e-9)

    # check update state
    metric.update_state(y_true, y_pred)
    tf.debugging.assert_near(metric.result(), 0.033155613, atol=1e-9)
