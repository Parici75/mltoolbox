""""Computational statistics utilities."""
import logging
from numpy.typing import ArrayLike
import numpy as np
from scipy.cluster.hierarchy import dendrogram


logger = logging.getLogger(__name__)

def bootstrap_mean_ci(data: ArrayLike, n_boot: int = 1000, ci: float = 0.95
                      ) -> np.ndarray | None:
    """Estimates confidence interval of the mean by bootstrap resampling."""
    low_end = (1 - ci) / 2
    high_end = 1 - low_end

    # To numpy array
    data_array = np.asarray(data)
    if data_array.ndim == 0:
        raise ValueError(f"Can not calculate CI for a scalar input")

    # Bootstrap mean
    boot_sample = np.random.choice(
        data_array[~np.isnan(data_array)], (data_array.size, n_boot)
    )
    boot_sample_mean = boot_sample.mean(axis=0)

    return np.quantile(boot_sample_mean, [low_end, high_end])


def bootstrap_corr_pval(
    data_array1: ArrayLike,
    data_array2: ArrayLike,
    n_boot: int = 1000,
) -> float:
    """Estimates p-value of a correlation between two data series by random permutations of the second series."""

    # To numpy arrays
    x1 = np.asarray(data_array1)
    x2 = np.asarray(data_array2)
    # Calculate Pearson correlation coefficient
    pcc = np.corrcoef(x1, x2)[0, 1]

    # Bootstrap samples of permuted arrays
    boot_sample = np.array(
        [np.random.permutation(x2) for _ in np.arange(n_boot)]
    ).transpose()
    # Cross-correlation between array and bootstraps
    cross_corr = np.corrcoef(
        np.concatenate((x1.reshape((-1, 1)), boot_sample), axis=1).transpose()
    )
    # 1st column of the matrix gives us the bootstrap correlation values
    boot_sample_corr = cross_corr[0, 1:]

    if pcc > 0:
        return np.sum(boot_sample_corr > pcc) / n_boot

    return np.sum(boot_sample_corr < pcc) / n_boot


def linalg_norm(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Computes pairwise euclidian distance between a vector and each row of a matrix."""

    return np.linalg.norm(vector - matrix, axis=1)


def fancy_dendrogram(*args, **kwargs):
    """Plots a pretty dendrogram.

    From https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering
    -and-dendrogram-tutorial/
    """

    max_d = kwargs.pop("max_d", None)  # Plot a cut-off line
    if max_d and "color_threshold" not in kwargs:
        kwargs["color_threshold"] = max_d
    annotate_above = kwargs.pop("annotate_above", 0)

    xlabel = kwargs.pop("xlabel", None)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get("no_plot", False):
        if kwargs.get("truncate_mode", False):
            plt.title("Hierarchical Clustering Dendrogram (truncated)")
        else:
            plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel(xlabel)
        plt.ylabel("Distance")
        for i, d, c in zip(
            ddata["icoord"], ddata["dcoord"], ddata["color_list"]
        ):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, "o", c=c)
                plt.annotate(
                    "%.3g" % y,
                    (x, y),
                    xytext=(0, -5),
                    textcoords="offset points",
                    va="top",
                    ha="center",
                )
        if max_d:
            plt.axhline(y=max_d, c="k")

        plt.xticks(horizontalalignment="right")  # Rotate the xticklabels

    return ddata