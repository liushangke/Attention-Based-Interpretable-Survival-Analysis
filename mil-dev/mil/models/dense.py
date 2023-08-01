import tensorflow as tf
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
from mil.io.dataset import mil_datagen, threshold


def build_layer(x, units, activation, dropout, name):
    # single or split dense layer
    if isinstance(activation, (tuple, list)):
        if len(activation) == 2:
            top = tf.keras.layers.Dense(
                units, activation=activation[0], name=name + "_top"
            )(x)
            bottom = tf.keras.layers.Dense(
                units, activation=activation[1], name=name + "_bottom"
            )(x)
            x = tf.multiply(top, bottom)
        else:
            raise ValueError("Split layers requires two activations.")
    else:
        x = tf.keras.layers.Dense(units, activation=activation, name=name)(x)

    # add dropout if necessary
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout)(x)

    return x


def attention_flat(
    D,
    config={
        "backbone": {
            "units": [512, 256],  # backbone configuration
            "activations": ["relu", ("tanh", "sigmoid")],
            "dropout": [0.0, 0.0],
        },
        "attention": 2
        * [
            {
                "units": [1],  # attention configuration
                "activations": ["relu"],
                "dropout": [0.0],
                "transform": tf.exp,
            }
        ],
        "task": {
            "units": [1],  # attention configuration
            "activations": ["linear"],
            "dropout": [0.0],
        },
        "tap": 2,  # output of this layer feeds attention network (one-index)
        "pool": 1,  # attention weights used to pool this layer (one-index)
    },
    ragged=False,  # a flag used to make ragged model
):
    # create input layer
    x = tf.cond(
        ragged,
        lambda: tf.keras.Input([None, D], ragged=True),
        lambda: tf.keras.Input([None, D]),
    )

    # create lists to store inputs / outputs of backbone layers
    layers = [x]

    # squeeze singleton batch dimension
    x = tf.squeeze(x, axis=0)

    # build the backbone up to attention tap layer
    for i, (u, a, d) in enumerate(
        zip(
            config["backbone"]["units"][: config["tap"]],
            config["backbone"]["activations"][: config["tap"]],
            config["backbone"]["dropout"][: config["tap"]],
        )
    ):
        x = build_layer(x, u, a, d, f"dense{i}")
        layers.append(x)

    # build the attention networks
    weights = []
    for i, att in enumerate(config["attention"]):
        for j, (u, a, d) in enumerate(
            zip(att["units"], att["activations"], att["dropout"])
        ):
            if j == 0:
                x = build_layer(layers[-1], u, a, d, f"a{j}_{i}")
            else:
                x = build_layer(x, u, a, d, f"a{j}_{i}")

        # capture attention network outputs
        if "transform" in att.keys():
            weights.append(att["transform"](x))
        else:
            weights.append(att["transform"](x))

    # normalize weights
    normalized = []
    total = tf.reduce_sum(tf.concat(weights, axis=1), axis=0)
    for k, (w, t) in enumerate(zip(weights, total)):
        normalized.append(tf.math.divide_no_nan(w, t, name=f"attention_branch{k}"))

    # weighted average pooling
    pooled = []
    for n in normalized:
        pooled.append(
            tf.tensordot(tf.reshape(n, [1, -1]), layers[config["pool"]], axes=1)
            / tf.reduce_sum(tf.reshape(n, [-1]))
        )

    # build remaining layers for prediction - multiclass branches
    predictions = []
    for i, p in enumerate(pooled):
        for j, (u, a, d) in enumerate(
            zip(
                config["task"]["units"],
                config["task"]["activations"],
                config["task"]["dropout"],
            )
        ):
            if j == 0:
                x = build_layer(p, u, a, d, f"pooled{j}_{i}")
            else:
                x = build_layer(x, u, a, d, f"pooled{j}_{i}")

        # capture outputs
        predictions.append(x)

    # combine outputs
    scores = tf.keras.layers.Concatenate(axis=1)(predictions)
    softmax = tf.keras.layers.Activation("softmax", name="softmax")(scores)

    # return output
    return tf.keras.Model(
        inputs=layers[0],
        outputs=[
            softmax,
            tf.keras.layers.Concatenate(-1, name="attention_weights", trainable=False)(
                normalized
            ),
        ],
    )


def attention_flat_train(config):
    model = attention_flat(
        config["training_params"]["D"], config["trainable_flat_model_hyperparam"]
    )

    model.compile(
        loss=config["training_params"]["loss"],
        optimizer=config["training_params"]["optimizer"],
        metrics=config["training_params"]["metric"],
    )

    datagen = mil_datagen(
        config["dataset_params"]["files"],
        config["dataset_params"]["variables"],
        config["training_params"]["batch_size"],
        config["dataset_params"]["structured"],
    )

    model.fit(
        datagen,
        batch_size=config["training_params"]["batch_size"],
        epochs=config["training_params"]["epochs"],
        verbose=config["training_params"]["verbose"],
        callbacks=[
            TuneReportCallback(
                {"mean_metric": "softmax_" + config["training_params"]["metric"].name}
            )
        ],
    )


class attention_flat_tune:
    """A class for tuning attention_flat model's hyperparameters.

    It is highly customizable so that you can choose any hyperparameters for tuning and modify training and tuning parameters as well.

    Parameters
    ----------
    trial_num : number
        Number of times to sample from the hyperparameter space

    resources_per_trial : number
        Machine resources to allocate per trial, e.g. {"cpu": 64, "gpu": 8}.

        # If you have 4 GPUs on your machine, this will run 2 concurrent trials at a time.
            tune.run(trainable, num_samples=10, resources_per_trial={"gpu": 2})

    Returns
    -------
    variables : dict
        Best found hyperparameters as a dict format. You can pass the index .
    """

    def __init__(self, trial_num=10, resources_per_trial=1):
        self.trial_num = trial_num
        self.resources_per_trial = resources_per_trial
        self.config = {
            "dataset_params": {"files": None, "variables": None, "structured": False},
            "training_params": {
                "epochs": 12,
                "batch_size": 1,
                "D": 1536,
                "verbose": 1,
                "threads": 2,
                "loss": tf.keras.losses.BinaryCrossentropy(),
                "optimizer": tf.keras.optimizers.Adam(learning_rate=1e-4),
                "metric": tf.keras.metrics.BinaryAccuracy(),
                "scheduler": None,  # for scheduler can use ray.tune.schedulers.AsyncHyperBandScheduler
            },
            "tune_run_params": {
                "name": "model_tune",
                "mode": "max",
                "max_metric_to_stop": 0.99,
                "max_iteration_to_stop": 500,
            },
            "trainable_flat_model_hyperparam": {
                "backbone": {
                    "units": [
                        tune.randint(100, 512),
                        tune.randint(100, 256),
                    ],  # backbone configuration
                    "activations": ["relu", ("tanh", "sigmoid")],
                    "dropout": [
                        tune.uniform(0.0, 0.5),
                        tune.uniform(0.0, 0.5),
                    ],
                },
                "attention": 2
                * [
                    {
                        "units": [1],  # attention configuration
                        "activations": ["relu"],
                        "dropout": [tune.uniform(0.0, 0.5)],
                        "transform": tf.exp,
                    }
                ],
                "task": {
                    "units": [1],  # attention configuration
                    "activations": ["linear"],
                    "dropout": [tune.uniform(0.0, 0.5)],
                },
                "tap": 2,  # output of this layer feeds attention network (one-index)
                "pool": 1,  # attention weights used to pool this layer (one-index)
                "ragged": False,
            },
        }

    def tune(self):
        analysis = tune.run(
            attention_flat_train,
            name=self.config["tune_run_params"]["name"],
            scheduler=self.config["training_params"]["scheduler"],
            metric="mean_metric",
            mode=self.config["tune_run_params"]["mode"],
            stop={
                "mean_metric": self.config["tune_run_params"]["max_metric_to_stop"],
                "training_iteration": self.config["tune_run_params"][
                    "max_iteration_to_stop"
                ],
            },
            num_samples=self.trial_num,
            resources_per_trial={"gpu": self.resources_per_trial},
            config=self.config,
        )
        print(
            "Best hyperparameters found were: ",
            analysis.best_config["trainable_flat_model_hyperparam"],
        )
        return analysis.best_config["trainable_flat_model_hyperparam"]

    def get_config(self):
        return self.config

    def set_config(self, config):
        self.config = config
