import numpy as np
import tensorflow as tf
from survivalnet2.estimators.km import km_np
from matplotlib import pyplot as plt


def km_plot(
    labels, groups=None, xlabel="Time", ylabel="Survival probability", legend=None
):
    """Generates Kaplan-Meier plot of possibly multiple groups on a single
    plot. Displays censored points, confidence limits, and median survival
    times.

    Parameters
    ----------
    labels : float
        An N x 2 float32 tensor where the times are in the first column and the
        labels are in the second column. Any nonzero event value -> event was
        observed.
    groups : int or float
        A vector of group labels for each sample. Default value 'None'.

    Returns
    -------
    figure : object
        Handle to figure containing axes with Kaplan Meier plot.
    axes : object
        Handle to axes containing Kaplan Meier plot.

    See Also
    -----
    km : Kaplan-Meier estimator.
    """

    # generate figure, axes handles for output
    figure, axes = plt.subplots()

    # get unique group labels
    if groups is None:
        groups = np.ones(labels.shape[0])
    unique = np.unique(groups)

    # generate traces for each group
    for i, group in enumerate(unique):
        # select samples in group
        members = np.argwhere(groups == group)[:, 0]

        # km estimator of survival function, 95% confidence limits, median times
        t_i, s_t, med_t_i, upper, lower, n_i, c_i, s_c = km_np(labels[members, :])

        # survival function plot and confidence limits
        axes.step(t_i, s_t, where="post")
        axes.fill_between(t_i, lower, upper, step="post", alpha=0.3, label="_nolegend_")

        # censored points
        axes.scatter(c_i, s_c, marker="+", color="k", label="_nolegend_")

        # median survival
        if med_t_i is not None:
            axes.plot(
                [0, med_t_i, med_t_i],
                [0.5, 0.5, 0],
                linestyle="--",
                color="gray",
                label="_nolegend_",
            )

    # set limits, label axes, add legend
    plt.ylim((0, 1.02))
    plt.xlim(left=0)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if legend is None:
        legend = [str(group) for group in unique]
    plt.legend(legend)
