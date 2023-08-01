from .gradient_reversal import GradientReversal
from survivalnet2.search.utils import (
    check_tunable,
    keras_losses,
    keras_metrics,
    set_hyperparameter,
)
import tensorflow as tf


activations = {"elu", "gelu", "linear", "relu", "selu", "sigmoid", "softplus"}


def optimization_space(
    batch={32, 64, 128},
    method={"rms", "sgd", "adadelta", "adagrad", "adam"},
    learning_rate=[1e-5, 1e-2, 1e-5],
    rho=[0.5, 1.0, 1e-2],
    momentum=[0.0, 1e-1, 1e-2],
    beta_1=[0.5, 1.0, 1e-2],
    beta_2=[0.5, 1.0, 1e-2],
):
    """Generate a search space for a training and optimization parameters.

    This function is used to generate a search space for batching and gradient optimization
    parameters. This includes batch size, gradient optimization method, and method
    parameters like learning rate. The search space for each parameter
    is defined as: 1. A set defining specific values to search 2. A list specifying a search
    range [min, max, increment(optional)] 3. A specific value to assign to hyperparameter
    (no search).

    Parameters
    ----------
    batch : set[int]
        A set of batch sizes (int) to explore. Default value is {32, 64, 128}.
    method : set[string]
        A list of strings encoding the gradient optimization method. The strings in `method`
        are converted to tf.keras.optimizer objects during training by
        search.utils.keras_optimizer. Default value is {"rms", "sgd", "adadelta", "adagrad",
        "adam"}.
    learning_rate : list[float]
        The range of learning rates for the gradient optimizer. Default value
        is [1e-5, 1e-2, 1e-5].
    rho : list[float]
        The range of learning rate decay values for the optimizers. Default value
        is [0.5, 1.0, 1e-2].
    momentum : list[float]
        The range of momentum values for the momentum optimizers. Default value
        is [0.0, 1e-1, 1e-2].
    beta_1 : list[float]
        The range of beta_1 values for the adam optimizer. Default value is
        [0.5, 1.0, 1e-2].
    beta_2 : list[float]
        The range of beta_2 values for the adam optimizer. Default value is
        [0.5, 1.0, 1e-2].

    Returns
    -------
    task : dict
        A ray optimization search space.
    """

    # verify that batch is int, set of int, or list of int
    check_tunable(batch, int, "batch")

    # verify that method is str or set of str
    check_tunable(method, str, "method")

    # verify float parameters
    check_tunable(learning_rate, float, "learning_rate")
    check_tunable(rho, float, "rho")
    check_tunable(momentum, float, "momentum")
    check_tunable(beta_1, float, "beta_1")
    check_tunable(beta_2, float, "beta_2")

    # convert to ray search space
    space = set_hyperparameter(
        {
            "batch": batch,
            "method": method,
            "learning_rate": learning_rate,
            "rho": rho,
            "momentum": momentum,
            "beta_1": beta_1,
            "beta_2": beta_2,
        }
    )

    return space


def task_space(
    adversarial=False,
    layers=[1, 3],
    layer_proto={
        "activation": activations,
        "dropout": [0.0, 0.5, 0.05],
        "units": {64, 48, 32, 16},
    },
    loss="efron",
    loss_weight=1.0,
    metrics={"harrellsc": "harrellsc"},
):
    """Generate a search space for a task in a multitask adversarial network.

    This function is used to generate task search spaces for `advmtl_search`. The task search
    space includes the number of task layers, layer configuration, task loss, task loss weight,
    and task metrics. The search space for each parameter is defined as: 1. A set defining
    specific values to search 2. A list specifying a search range [min, max, increment(optional)]
    3. A specific value to assign to hyperparameter (no search).

    Parameters
    ----------
    adversarial : bool
        If true, the gradient of the task will be negated during training to provide
        negative feedback to the model.
    layers : int, list[int], set[int]
        The total number of layers (int), range of layers (list), or discrete layer
        choices (set) for the task, including the output layer.
    layer_proto : dict
        A dict defining the activations (string), number of units (int), and dropout
        rates (float) for task layers 1:layers-1.
    loss : callable, set[callable]
        A loss function or set of possible loss functions.
    loss_weight : float, list[float], set[float]
        The loss weight (float), range of loss weights (list), or discrete loss weight
        choices (set) for the task.
    metrics : dict
        A dictionary of metric name strings for measuring task performance.

    Returns
    -------
    task : dict
        A ray.tune task search space.
    """

    # verify that adversarial is scalar bool
    if not isinstance(adversarial, bool):
        raise ValueError(
            f"adversarial must be bool, received {type(adversarial).__name__}"
        )

    # verify that layers is int, set of int, or list of int
    check_tunable(layers, int, "layers")

    # verify that layer_proto has minimum required keys
    if not {"activation", "units", "dropout"}.issubset(set(layer_proto.keys())):
        raise ValueError(
            "Layer prototype must contain keys for activation, dropout, and units."
        )

    # verify that loss is str or set of str
    check_tunable(loss, str, "loss")

    # verify that loss_weight is float, set of float, or list of float
    check_tunable(loss_weight, float, "loss_weight")

    # verify that metrics is a dict of strings
    if not isinstance(metrics, dict):
        raise ValueError(
            f"metrics must be a dict of str, received {type(metrics).__name__}"
        )
    for metric in metrics:
        if not isinstance(metrics[metric], str):
            raise ValueError(
                f"metric elements must be str, received {type(metrics[metric]).__name__}"
            )

    # set initial values
    task = {
        "adversarial": False,
        "layers": layers,
        "layer_proto": layer_proto,
        "loss": loss,
        "loss_weight": loss_weight,
        "metrics": metrics,
    }

    # convert to ray search space
    space = set_hyperparameter(task)

    return space


def advmtl_space(
    dimension,
    optimization=optimization_space(),
    tasks={"task1": task_space()},
    inputs={
        "layers": 1,
        "layer_proto": {
            "activation": activations,
            "dropout": [0.0, 0.5, 0.05],
            "units": {256, 128, 64},
        },
    },
    shared={
        "layers": [1, 3],
        "layer_proto": {
            "activation": activations,
            "dropout": [0.0, 0.5, 0.05],
            "units": {256, 128, 64, 32},
        },
    },
):
    """Generates a search space for multi-task domain adversarial models. A
    simple single-task model with efron loss is generated by default.
    The search space defines the range of model hyperparameters that ray.tune can
    evaluate. During tuning, this space will be sampled to create, train, and
    evaluate a model using `advmtl_model` and `Tuner.train`.
    The search space for each hyperparameter can be defined by 1. A set defining
    specific values to search or 2. A list containing a search range [min, max,
    increment(optional)]. Any other value is interpreted as a specific hyperparameter
    choice (no search).
    An adversarial multi-task network has an input layer followed by a sequence of
    shared layers, followed by task branches/heads containing task-specific layers.
    Input and shared layers are defined by layer prototypes setting the range
    of activations, dropout rate, and units to explore. The number of layers is
    tunable for the input and shared layers, and independently for each task.
    Flagging a task as adversarial provides negative feedback to the model during
    training by negating the gradient where the task and shared networks meet.
    For example, when mixing datasets from different sources the adversarial penalty
    will punish the network for correctly predicting which dataset a sample
    originates from.
    Parameters
    ----------
    dimension : int
        The dimension of model input features.
    optimization : dict
        A dictionary defining the optimization search space. This defines the batch
        size, gradient optimization method, and method parameters. See `optimization_space`.
    tasks : dict
        A dictionary of task ray search spaces. This defines the task layer prototype,
        the number of task outputs, output activation, the task loss, loss weight,
        task metrics, and whether the task is an adversarial task. See `task_space`.
    inputs : dict
        A dict defining the number of input layers, and the input layer search space.
        The search space defines the activations (string), number of units (int), and
        dropout rates (float) for the input layers.
    shared : dict
        A dict defining the number of shared layers and the shared layer search space.
        The search space defines the activations (string), number of units (int), and
        dropout rates (float) for all shared layers.
    Returns
    -------
    space : dict
        A search space definition for use with ray.tune.
    """

    # set feature dimension
    space = {"dimension": dimension}

    # set optimization search space
    space["optimization"] = optimization

    # set input layer search space
    space["input"] = set_hyperparameter(inputs)

    # set shared layers search space
    space["shared"] = set_hyperparameter(shared)

    # set task layers search space
    space["tasks"] = tasks

    return space


def advmtl_model(config):
    """Creates a tf.keras.Model from a hyperparameter configuration.
    This function produces a model from a hyperparameter configuration
    drawn from a search space.
    Parameters
    ----------
    config : dict
        A model configuration describing input dimension and hyperparameter
        values sampled from the search space.
    Returns
    -------
    model : tf.keras.Model
        A tf.keras.Model object build to specificiations in config. Outputs
        will be named according to the `task_names` and `adversarial_names`
        arguments.
    losses : dict
        A dict of model output name : tf.keras.losses.Loss key value pairs
        for use with tf.keras.Model.compile.
    loss_weights : dict
        A dict of model output name : float key value pairs for use with
        tf.keras.Model.compile.
    metrics : dict
        A dict of model output name : tf.keras.losses.Loss key value pairs
        for use with tf.keras.Model.compile.
    """

    def _build_layer(x, units, activation, dropout, name):
        # dense layer
        x = tf.keras.layers.Dense(units, activation=activation, name=name)(x)

        # add dropout if necessary
        if dropout > 0.0:
            x = tf.keras.layers.Dropout(dropout)(x)

        return x

    # create input layers
    x = tf.keras.Input([config["dimension"]], name="input")

    # create lists to store inputs / outputs of input + shared layers
    outputs = [x]

    # build input layers
    for i, (u, a, d) in enumerate(
        zip(
            config["input"]["layers"] * [config["input"]["layer_proto"]["units"]],
            config["input"]["layers"] * [config["input"]["layer_proto"]["activation"]],
            config["input"]["layers"] * [config["input"]["layer_proto"]["dropout"]],
        )
    ):
        x = _build_layer(x, u, a, d, f"input_{i}")
        outputs.append(x)

    # build shared layers
    for i, (u, a, d) in enumerate(
        zip(
            config["shared"]["layers"] * [config["shared"]["layer_proto"]["units"]],
            config["shared"]["layers"]
            * [config["shared"]["layer_proto"]["activation"]],
            config["shared"]["layers"] * [config["shared"]["layer_proto"]["dropout"]],
        )
    ):
        x = _build_layer(x, u, a, d, f"shared_{i}")
        outputs.append(x)

    # build the task heads
    tasks = []
    for name in config["tasks"]:
        # if task is adversarial, apply gradient reversal
        if config["tasks"][name]["adversarial"]:
            x = GradientReversal()(outputs[-1])
        else:
            x = outputs[-1]

        # iterate over first n-1 task layers
        for j, (u, a, d) in enumerate(
            zip(
                (config["tasks"][name]["layers"] - 1)
                * [config["tasks"][name]["layer_proto"]["units"]],
                (config["tasks"][name]["layers"] - 1)
                * [config["tasks"][name]["layer_proto"]["activation"]],
                (config["tasks"][name]["layers"] - 1)
                * [config["tasks"][name]["layer_proto"]["dropout"]],
            )
        ):
            x = _build_layer(x, u, a, d, f"{name}_{j}")

        # add task output layer
        x = _build_layer(x, 1, "linear", 0.0, f"{name}")

        # capture task subnetwork outputs
        tasks.append(x)

    # build named output dict
    named = {f"{name}": task for (name, task) in zip(config["tasks"].keys(), tasks)}

    # create model
    model = tf.keras.Model(inputs=outputs[0], outputs=named)

    # create a loss dictionary
    losses, loss_weights = keras_losses(config)

    # create a metric dictionary from the config
    metrics = keras_metrics(config)

    # return output
    return model, losses, loss_weights, metrics
