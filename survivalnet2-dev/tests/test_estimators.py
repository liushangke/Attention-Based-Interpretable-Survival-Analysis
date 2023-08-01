import numpy as np
import tensorflow as tf
from survivalnet2.data.labels import stack_labels
from survivalnet2.estimators.km import km, km_eval, km_np


def km_test_data():
    # testing data
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
        bool,
    )
    return times, events


def test_km_np():
    # testing data
    times, events = km_test_data()

    # define expected answers
    t_i_exp = np.array(
        [
            0,
            6,
            10,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            20,
            21,
            24,
            25,
            28,
            30,
            33,
            35,
            37,
            38,
            40,
            46,
            48,
            49,
            50,
            63,
            76,
            79,
            81,
            82,
            86,
            91,
            98,
            112,
            181,
            202,
            219,
        ]
    )
    s_t_exp = np.array(
        [
            1.0,
            0.98039216,
            0.9411765,
            0.92156863,
            0.88235295,
            0.8627451,
            0.84313726,
            0.8235294,
            0.8039216,
            0.78431374,
            0.7647059,
            0.74509805,
            0.7058824,
            0.6862745,
            0.6666667,
            0.627451,
            0.6072107,
            0.58627236,
            0.54439574,
            0.5234574,
            0.48158082,
            0.45969078,
            0.43670624,
            0.4137217,
            0.39073715,
            0.3677526,
            0.34323576,
            0.3187189,
            0.29215902,
            0.2655991,
            0.23239921,
            0.19919932,
            0.16599943,
            0.13279955,
            0.08853303,
            0.04426651,
            0.0,
        ]
    )
    med_t_i_exp = 40
    upper_exp = np.array(
        [
            1.0,
            0.9972146,
            0.9806409,
            0.96981853,
            0.9453494,
            0.93208176,
            0.9182766,
            0.90400773,
            0.8893303,
            0.87428695,
            0.8589113,
            0.84323025,
            0.8110359,
            0.7945555,
            0.777837,
            0.7437252,
            0.72591984,
            0.7074189,
            0.66968966,
            0.6504698,
            0.6113318,
            0.5907431,
            0.5691288,
            0.54719776,
            0.52494633,
            0.5023687,
            0.47839195,
            0.4539856,
            0.42776746,
            0.4009322,
            0.36959818,
            0.33672166,
            0.30220118,
            0.26584727,
            0.22365227,
            0.17496257,
            0.17496257,
        ]
    )
    lower_exp = np.array(
        [
            1.0,
            0.8688456,
            0.82860863,
            0.8043773,
            0.7567308,
            0.7335241,
            0.7107207,
            0.68829423,
            0.6662177,
            0.64446646,
            0.6230188,
            0.6018561,
            0.5603246,
            0.53993076,
            0.5197712,
            0.48012432,
            0.4597899,
            0.43878523,
            0.39763123,
            0.37746623,
            0.33793625,
            0.3174127,
            0.29588455,
            0.27476168,
            0.25404197,
            0.23372705,
            0.21206686,
            0.1909635,
            0.16814396,
            0.1461505,
            0.11771072,
            0.09148609,
            0.06755435,
            0.04611463,
            0.0197515,
            0.00378989,
            0.00378989,
        ]
    )
    n_i_exp = np.array(
        [
            51.0,
            50.0,
            48.0,
            47.0,
            45.0,
            44.0,
            43.0,
            42.0,
            41.0,
            40.0,
            39.0,
            38.0,
            36.0,
            35.0,
            34.0,
            31.0,
            29.0,
            28.0,
            26.0,
            25.0,
            22.0,
            20.0,
            19.0,
            18.0,
            17.0,
            15.0,
            14.0,
            12.0,
            11.0,
            8.0,
            7.0,
            6.0,
            5.0,
            3.0,
            2.0,
            1.0,
        ]
    )
    c_i_exp = np.array([31, 34, 40, 47, 70, 80, 82, 149])
    s_c_exp = np.array(
        [
            0.627451,
            0.6072107,
            0.52345741,
            0.45969078,
            0.36775261,
            0.31871891,
            0.29215902,
            0.13279955,
        ]
    )

    # stack times
    labels = stack_labels(times, events)

    # calculate KM
    t_i, s_t, med_t_i, upper, lower, n_i, c_i, s_c = km_np(labels)

    # compare to expected values
    np.testing.assert_allclose(t_i, t_i_exp, atol=1e-7)
    np.testing.assert_allclose(s_t, s_t_exp, atol=1e-7)
    assert med_t_i == med_t_i_exp
    np.testing.assert_allclose(upper, upper_exp, atol=1e-7)
    np.testing.assert_allclose(lower, lower_exp, atol=1e-7)
    np.testing.assert_allclose(n_i, n_i_exp, atol=1e-7)
    np.testing.assert_allclose(c_i, c_i_exp, atol=1e-7)
    np.testing.assert_allclose(s_c, s_c_exp, atol=1e-7)


def test_km():
    # testing data
    times, events = km_test_data()

    # define expected answers
    t_i_exp = tf.constant(
        [
            0,
            6,
            10,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            20,
            21,
            24,
            25,
            28,
            30,
            33,
            35,
            37,
            38,
            40,
            46,
            48,
            49,
            50,
            63,
            76,
            79,
            81,
            82,
            86,
            91,
            98,
            112,
            181,
            202,
            219,
        ],
        tf.float32,
    )
    s_t_exp = tf.constant(
        [
            1.0,
            0.98039216,
            0.9411765,
            0.92156863,
            0.88235295,
            0.8627451,
            0.84313726,
            0.8235294,
            0.8039216,
            0.78431374,
            0.7647059,
            0.74509805,
            0.7058824,
            0.6862745,
            0.6666667,
            0.627451,
            0.6072107,
            0.58627236,
            0.54439574,
            0.5234574,
            0.48158082,
            0.45969078,
            0.43670624,
            0.4137217,
            0.39073715,
            0.3677526,
            0.34323576,
            0.3187189,
            0.29215902,
            0.2655991,
            0.23239921,
            0.19919932,
            0.16599943,
            0.13279955,
            0.08853303,
            0.04426651,
            0.0,
        ],
        tf.float32,
    )

    # stack times
    labels = stack_labels(times, events)

    # calculate KM
    t_i, s_t = km(labels)

    # compare to expected values
    np.testing.assert_allclose(t_i, t_i_exp, atol=1e-7)
    np.testing.assert_allclose(s_t, s_t_exp, atol=1e-7)


def test_km_eval():
    # testing data
    times, events = km_test_data()

    # define expected answers
    s_i_exp = tf.constant(
        [
            0.98039216,
            0.88235295,
            0.74509805,
            0.627451,
            0.627451,
            0.54439574,
            0.5234574,
            0.45969078,
            0.4137217,
            0.39073715,
            0.3677526,
            0.3187189,
            0.3187189,
            0.2655991,
            0.2655991,
            0.23239921,
            0.16599943,
            0.13279955,
            0.04426651,
            0.0,
            0.9411765,
            0.9411765,
            0.92156863,
            0.88235295,
            0.8627451,
            0.84313726,
            0.8235294,
            0.8039216,
            0.78431374,
            0.7647059,
            0.7058824,
            0.7058824,
            0.6862745,
            0.6666667,
            0.627451,
            0.6072107,
            0.6072107,
            0.58627236,
            0.54439574,
            0.48158082,
            0.48158082,
            0.48158082,
            0.45969078,
            0.43670624,
            0.3677526,
            0.34323576,
            0.29215902,
            0.2655991,
            0.19919932,
            0.13279955,
            0.08853303,
        ]
    )

    # stack times
    labels = stack_labels(times, events)

    # calculate KM
    t_i, s_t = km(labels)

    # evaluate km function at times
    s_i = km_eval(times, t_i, s_t)

    # compare to expected values
    tf.debugging.assert_near(s_i, s_i_exp, atol=1e-9)
