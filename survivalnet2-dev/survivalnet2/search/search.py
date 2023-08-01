import datetime
from ray import tune
from ray.air import CheckpointConfig
from ray.air.config import FailureConfig, RunConfig
from ray.tune import CLIReporter
from ray.tune.integration.keras import TuneReportCheckpointCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.tune_config import TuneConfig
from survivalnet2.search.utils import keras_optimizer


class Search(object):
    """A general-purpose class for automated hyperparameter tuning.

    This class is model-agnostic, and users can apply it to any model type by
    implementing a search space and model builder function. Initialization sets
    a default tuning options including possible batch sizes, epochs,
    optimization algorithms and hyperparameters, as well as the number of
    tuning trials and trial concurrency. This class instance can be re-used
    for different models to make head to head performance comparisons.
    """

    def __init__(
        self,
        num_samples=100,
        max_concurrent_trials=8,
        tune_config_kwargs={},
        run_config_kwargs={},
        fit_kwargs={},
    ):
        """Constructor.

        Parameters
        ----------
        num_samples : int
            The number of trials to run when tuning a model. Default value 100.
        max_concurrent_trials : int
            The maximum number of trials to run concurrently. Default value 8.
        tune_config_kwargs : dict
            Keyword arguments for ray.air.config.RunConfig. These allow
            additional customization of ray.tune.Tuner.fit runtime options.
            Default value is {}.
        run_config_kwargs : dict
            Keyword arguments for ray.tune.tune_config.TuneConfig. These alllow
            additional customization of ray.tune.Tuner.fit tuning behavior.
            Default value is {}.
        fit_kwargs : dict
            Keyword arguments for tf.keras.model.fit. These allow customization
            of keras model training options. Devault value is {}.
        """

        # kwargs for ray.tune.tune_config.TuneConfig
        self.tune_config_kwargs = {
            "mode": "max",
            "num_samples": num_samples,
            "max_concurrent_trials": max_concurrent_trials,
        }

        # override tune config arguments with user preferences
        for key in tune_config_kwargs:
            self.tune_config_kwargs[key] = tune_config_kwargs[key]

        # kwargs for ray.air.config.RunConfig
        self.run_config_kwargs = {
            "failure_config": FailureConfig(max_failures=5),
            "log_to_file": True,
        }

        # override run config arguments with user preferences
        for key in run_config_kwargs:
            self.run_config_kwargs[key] = run_config_kwargs[key]

        # kwargs to tf.keras.model.fit
        self.fit_kwargs = fit_kwargs

    def trial(self, config):
        """Trains a model from a hyperparameter configuration.

        This function compiles a model from the config and trains using general
        parameters found in self.options. Communication of results is performed
        using the TuneReportCheckpointCallback from ray.tune.integration.keras.

        Parameters
        ----------
        config : dict
            A configuration describing optimization and model hyperparameters
            as well as functions for data loading and model building.
        """

        # create the model from the config
        model, losses, loss_weights, metrics = config["builder"](config)

        # build optimizer
        optimizer = keras_optimizer(config["optimization"])

        # compile the model with the optimizer, losses, and metrics
        model.compile(
            optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics
        )

        # load example data, generate random train/test split
        train_dataset, validation_dataset = config["loader"](
            config["optimization"]["batch"]
        )

        # epoch reporting of performance metrics - link keras metric names
        # to names displayed by ray in reporting
        report = {
            f"{t}_{m}": f"val_{t}_{m}" if len(config["tasks"]) > 1 else f"val_{m}"
            for t in config["tasks"]
            for m in config["tasks"][t]["metrics"]
        }

        callback = TuneReportCheckpointCallback(report)

        # train the model for the desired epochs using the call back
        model.fit(
            train_dataset,
            epochs=config["epochs"],
            validation_data=validation_dataset,
            callbacks=[callback],
            verbose=0,
            **config["fit_kwargs"],
        )

    def experiment(
        self,
        metric,
        space,
        builder,
        loader,
        local_dir,
        name=None,
        scheduler=AsyncHyperBandScheduler(
            time_attr="training_iteration",
            max_t=100,
            grace_period=10,
            stop_last_trials=False,
        ),
        search_alg=None,
    ):
        """Run hyperparameter tuning experiment trials.

        This can be called multiple times with different search spaces
        definitions, model building functions, schedulers, and search
        algorithms to perform experiments under different conditions.

        Parameters
        ----------
        metric : str
            The name of the metric to optimize. This is in the form "task_name"
            where "task" is the task name and "name" is the key value of the
            metric to optimize.
        space : dict
            A model search space that defines the range of possible model
            characteristics like architecture, activations, dropout, losses,
            and loss weights.
        builder : callable
            A function that returns a tf.keras.model given a configuration
            sampled from the search space.
        loader : callable
            A function that returns batched training and validation datasets
            of type tf.data.Dataset. This function should accept a single
            argument defining the batch size.
        local_dir : str
            The path to store experiment results including logs and model
            checkpoints.
        name : str
            The name for the experiment. Used by developers to organize
            experimental results. Default value of None uses datetime
            year_month_day_hour_minute_second.
        scheduler : object
            A scheduling algorithm from ray.tune.schedulers that can be used
            to terminate poorly performing trials, to pause trials, to clone
            trials, and to alter hyperparameters of a running trial. Some
            search algorithms do not require a scheduler. Default value is the
            AsyncHyperBandScheduler.
        search_alg : object
            A search algorithm from ray.tune.search for adaptive hyperparameter
            selection. Default value of None results in a random search with
            the AsyncHyperBandScheduler.

        Returns
        -------
        analysis : ray.tune.ResultGrid
            A dictionary describing the tuning experiment outcomes. See the
            documentation of ray.tune.Tuner.fit() and ray.tune.ResultGrid for
            more details.
        """

        # set config name
        if name is None:
            name = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        # capture builder and loader functions in config
        space["builder"] = builder
        space["loader"] = loader

        # capture fit kwargs
        space["fit_kwargs"] = self.fit_kwargs

        # setup reporter
        reporter = CLIReporter(
            metric_columns=[
                f"{t}_{m}" for t in space["tasks"] for m in space["tasks"][t]["metrics"]
            ],
            parameter_columns={
                "optimization/batch": "batch",
                "optimization/method": "method",
                "optimization/learning_rate": "learning rate",
            },
            sort_by_metric=True,
        )

        # create tune config
        self.tune_config_kwargs["metric"] = metric
        self.tune_config_kwargs["scheduler"] = scheduler
        if search_alg is not None:
            self.tune_config_kwargs["search_alg"] = search_alg
        tune_config = TuneConfig(**self.tune_config_kwargs)

        # create checkpoint config for run config
        checkpoint_config = CheckpointConfig(
            checkpoint_score_attribute=metric,
            checkpoint_score_order=self.tune_config_kwargs["mode"],
            num_to_keep=1,  # saved checkpoints per trial
        )

        # create run config
        self.run_config_kwargs["local_dir"] = local_dir
        self.run_config_kwargs["name"] = name
        self.run_config_kwargs["progress_reporter"] = reporter
        self.run_config_kwargs["sync_config"] = tune.SyncConfig(syncer=None)
        self.run_config_kwargs["checkpoint_config"] = checkpoint_config
        run_config = RunConfig(**self.run_config_kwargs)

        # run experiment
        tuner = tune.Tuner(
            self.trial,
            param_space=space,
            tune_config=tune_config,
            run_config=run_config,
        )
        analysis = tuner.fit()

        return analysis
