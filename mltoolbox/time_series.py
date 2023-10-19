"""Tools for common time-series analyses."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pymodules.numpy_utils import window_slide


def ccf(
    x_series: ArrayLike,
    y_series: ArrayLike,
    lag: int | None = None,
    ci: float | None = None,
    n_boot: int = 1000,
    block_width: int = 5,
) -> tuple[NDArray[np.float_], NDArray[Any] | None]:
    """Calculates the sample cross correlation function (CCF) across time lags between a "predictor"
    x_series and a "predicted" y_series.

    We calculate sample CCF as the set of sample correlations between and xt and yt+h
    (that is, x "lags" y).
    see https://stackoverflow.com/a/20463466/4696032
    """

    def time_pearson_corr(h: int) -> float:
        """Calculates time-dependent cross-correlations between two time series using
        Pearson Correlation Coefficient.
        see https://en.wikipedia.org/wiki/Autocorrelation#Estimation/

        """
        partial_x = x_data[: n - h]
        partial_y = y_data[h:]

        # Product of the standard deviation of the series
        c0 = float(n - h) * np.std(partial_x) * np.std(partial_y)
        # PCC formula
        ccf_lag = np.sum((partial_x - np.mean(partial_x)) * (partial_y - np.mean(partial_y))) / c0

        return np.round(ccf_lag, 3)

    def moving_block_bootstrap_ci(h: int) -> list[float]:
        """Kunsch procedure to resample blocks of the array.
        This avoids losing time structure in the data.
        """

        x_kunsch_blocks = window_slide(x_data[: n - h], 1, block_width)
        y_kunsch_blocks = window_slide(y_data[h:], 1, block_width)
        if len(x_kunsch_blocks) == 0 | len(y_kunsch_blocks) == 0:
            x_kunsch_blocks = np.array([x_data[: n - h]])
            y_kunsch_blocks = np.array([y_data[h:]])

        # If array is not divisible by window size, ceil blocks number
        # and adjust the bootstrap array
        arr_size = len(y_data[h:])
        block_number = int(np.ceil(arr_size / block_width))

        bootstrap_corr = np.zeros(n_boot)
        for i in range(n_boot):
            # Random sampling of blocks indices with replacement
            random_block_idx = random.choices(  # noqa: S311
                np.arange(len(x_kunsch_blocks)), k=block_number
            )

            # Picks random blocks in both time series while preserving the time structure
            x_random_blocks = x_kunsch_blocks[random_block_idx]
            y_random_blocks = y_kunsch_blocks[random_block_idx]

            # Concatenating the array blocks
            resampled_xarray = np.hstack(x_random_blocks)
            resampled_yarray = np.hstack(y_random_blocks)

            # adjusting the boostrap array
            resampled_xarray = resampled_xarray[:arr_size]
            resampled_yarray = resampled_yarray[:arr_size]

            # Estimation of PCC
            c0 = float(n - h) * np.std(resampled_xarray) * np.std(resampled_yarray)
            bootstrap_corr[i] = (
                np.sum(
                    (resampled_xarray - np.mean(resampled_xarray))
                    * (resampled_yarray - np.mean(resampled_yarray))
                )
                / c0
            )

        quantile = np.quantile(bootstrap_corr, [low_end, high_end])

        return [quantile[0], quantile[-1]]

    x_data = np.asarray(x_series)
    y_data = np.asarray(y_series)
    if x_data.shape != y_data.shape:
        raise ValueError("Data arrays must have the same length")

    n = len(x_data)
    if lag is None:
        lag = n - 1
    elif lag > n:
        raise ValueError(f"Invalid value for lag: {lag} for a signal of length: {n}")

    if ci is not None:
        low_end = (1 - ci) / 2
        high_end = 1 - low_end

    time_lags = np.arange(lag)
    ccf_coeffs = np.array([time_pearson_corr(x) for x in time_lags])  # Calculating correlations
    if ci is not None:
        bootstrap_ci = np.array([moving_block_bootstrap_ci(x) for x in time_lags])
    else:
        bootstrap_ci = None

    return ccf_coeffs, bootstrap_ci
