import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from survivalnet2.layers.advmtl import (
    advmtl_model,
    advmtl_space,
    task_space,
)
from survivalnet2.layers.advmtl import activations
from survivalnet2.layers.gradient_reversal import GradientReversal
from survivalnet2.metrics.concordance import HarrellsC
from tensorflow.keras.losses import CategoricalHinge
from tensorflow.keras.metrics import CategoricalAccuracy


# helper function to generate an advmtl layer prototype
def generate_proto(activation, dropout, units):
    proto = {"activation": activation, "dropout": dropout, "units": units}
    return proto


# helper function to generate an advmtl task configuration
def generate_task(adversarial, layers, proto, loss, loss_weight, metrics):
    task = {
        "adversarial": adversarial,
        "layers": layers,
        "layer_proto": proto,
        "loss": loss,
        "loss_weight": loss_weight,
        "metrics": metrics,
    }
    return task


# helper function to sample an advmtl layer prototype
def proto_sample():
    return {
        "activation": np.random.choice(list(activations)),
        "dropout": np.random.uniform(0, 0.5),
        "units": np.random.choice([64, 48, 32, 16]),
    }


# helper function to sample an advmtl task configuration
def task_sample():
    return {
        "adversarial": np.random.uniform() > 0.5,
        "layers": np.random.randint(1, 4),
        "layer_proto": proto_sample(),
        "loss": np.random.choice(["efron", "cox"]),
        "loss_weight": 1.0,
        "metrics": {"c-index": "harrellsc"},
    }


# helper function to sample an advmtl model configuration
def config_sample(dimension=10):
    return {
        "dimension": dimension,
        "input": {
            "layer_proto": proto_sample(),
            "layers": np.random.randint(1, 4),
        },
        "shared": {
            "layer_proto": proto_sample(),
            "layers": np.random.randint(1, 4),
        },
        "tasks": {"task1": task_sample(), "task2": task_sample()},
    }


def test_task_space():
    # test raise on non-bool adversarial
    with pytest.raises(Exception) as exc:
        exc = task_space(adversarial=1)
    assert str(exc.value) == "adversarial must be bool, received int"

    # test raise on non int layers
    with pytest.raises(Exception) as exc:
        exc = task_space(layers=1.0)
    assert (
        str(exc.value) == "layers must be int, list[int], or set[int], received float"
    )

    # test raise on non int layers
    with pytest.raises(Exception) as exc:
        exc = task_space(layers=[1, 2.0])
    assert str(exc.value) == "layers list expects elements of type int, received float"

    # test raise on non int layers
    with pytest.raises(Exception) as exc:
        exc = task_space(layers={1, 2.0})
    assert str(exc.value) == "layers set expects elements of type int, received float"

    # verify that incomplete proto raises ValueError
    proto = {"invalid": 0.0}
    with pytest.raises(Exception) as exc:
        exc = task_space(layer_proto=proto)
    assert (
        str(exc.value)
        == "Layer prototype must contain keys for activation, dropout, and units."
    )

    # check that extra input doesn't raise error
    proto = {"activation": "relu", "units": 10, "dropout": 0.1, "extra": 1}
    task_space(layer_proto=proto)

    # test raise on non callable loss
    with pytest.raises(Exception) as exc:
        exc = task_space(loss=1.0)
    assert str(exc.value) == "loss must be str or set[str], received float"

    # test raise on non float loss weight
    with pytest.raises(Exception) as exc:
        exc = task_space(loss_weight=1)
    assert (
        str(exc.value)
        == "loss_weight must be float, list[float], or set[float], received int"
    )

    # test raise on invalid set of list element for loss weight
    with pytest.raises(Exception) as exc:
        exc = task_space(loss_weight=[1.0, 1])
    assert (
        str(exc.value)
        == "loss_weight list expects elements of type float, received int"
    )

    # test raise on invalid metrics
    with pytest.raises(Exception) as exc:
        exc = task_space(metrics=tf.keras.metrics.CategoricalAccuracy())
    assert (
        str(exc.value) == "metrics must be a dict of str, received CategoricalAccuracy"
    )

    # test raise on invalid metrics element
    with pytest.raises(Exception) as exc:
        exc = task_space(metrics={"named": lambda x: x + 1})
    assert str(exc.value) == "metric elements must be str, received function"


def test_advmtl_network():
    # sample 100 model configurations and verify that the models compile
    for _ in range(100):
        # create a network with default configuration
        config = config_sample()
        model, losses, loss_weights, metrics = advmtl_model(config)

        # try to compile network
        model.compile(loss=losses, loss_weights=loss_weights, metrics=metrics)


def test_advmtl_space():
    # helper function to compare layer prototypes
    def compare_layer_proto(generated, input_proto):
        if isinstance(input_proto["activation"], str):
            assert generated["activation"] == input_proto["activation"]
        elif isinstance(input_proto["activation"], set):
            assert set(generated["activation"].categories) == set(
                input_proto["activation"]
            )
        assert generated["dropout"].lower == input_proto["dropout"][0]
        assert generated["dropout"].upper == input_proto["dropout"][1]
        assert generated["dropout"].sampler.q == input_proto["dropout"][2]
        assert set(generated["units"].categories) == set(input_proto["units"])

    # check default search space - single task model
    space = advmtl_space(1)
    assert list(space["tasks"].keys()) == ["task1"]
    assert space["tasks"]["task1"]["loss"] == "efron"
    assert space["tasks"]["task1"]["loss_weight"] == 1.0
    assert space["tasks"]["task1"]["adversarial"] == False

    # build a multi-task search space
    proto1 = generate_proto("relu", 0.1, 16)
    proto2 = generate_proto("relu", 0.2, 32)
    tasks = {
        "task1": generate_task(False, 2, proto1, "efron", 1.0, {"cindex": HarrellsC()}),
        "task2": generate_task(
            True,
            3,
            proto2,
            CategoricalHinge,
            0.5,
            {"accuracy": CategoricalAccuracy()},
        ),
    }
    space = advmtl_space(dimension=10, tasks=tasks)
    assert list(space["tasks"].keys()) == ["task1", "task2"]
    assert space["tasks"]["task1"]["loss"] == "efron"
    assert space["tasks"]["task1"]["loss_weight"] == 1.0
    assert space["tasks"]["task1"]["adversarial"] == False
    # assert compare_layer_proto(space["tasks"]["task1"]["layer_proto"], proto1)
    assert space["tasks"]["task2"]["loss"] == CategoricalHinge
    assert space["tasks"]["task2"]["loss_weight"] == 0.5
    assert space["tasks"]["task2"]["adversarial"] == True
    # assert compare_layer_proto(space["tasks"]["task2"]["layer_proto"], proto2)

    # test adjustment of shared_layers
    arg = [2, 5]
    shared = {"layer_proto": generate_proto("relu", 0.1, units=16), "layers": arg}
    space = advmtl_space(dimension=10, shared=shared)
    assert space["shared"]["layers"].lower == arg[0]
    assert space["shared"]["layers"].upper == arg[1]
    assert space["shared"]["layers"].sampler.q == 1


def test_gradient_reversal():
    model = Sequential([Dense(10)])
    model_grl = Sequential([model, GradientReversal()])

    model.build(input_shape=(1, 10))
    model_grl.build(input_shape=(1, 10))

    g1 = tf.random.Generator.from_seed(1)
    inp = g1.normal(shape=[1, 10])

    with tf.GradientTape() as tape0, tf.GradientTape() as tape1:
        tape0.watch(inp)
        tape1.watch(inp)

        y0 = model(inp)
        y1 = model_grl(inp)
        g1 = tape0.gradient(y0, inp)
        g2 = tape1.gradient(y1, inp)
        assert tf.math.reduce_all((g1 == -1 * g2), axis=None, keepdims=False, name=None)
