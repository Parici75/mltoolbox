"""Main classes of the anomaly_detection subpackage."""

import logging
from collections.abc import Callable
from functools import partial, wraps
from typing import Any, Self, TypeVar, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pydantic import field_validator
from sklearn.mixture import GaussianMixture

from mltoolbox.base_model import BaseModel
from mltoolbox.exceptions import ModelFittingFailure
from mltoolbox.latent_space import PCALatentSpace, SignalTuner

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
GAF = TypeVar("GAF", bound="GaussianAnomalyQuantifier")

COVARIANCE_TYPE = ["spherical", "tied", "diag", "full"]


class GaussianAnomalyQuantifier(BaseModel):
    """Wrapper class for Gaussian modelling of data distribution."""

    signal_tuner: SignalTuner | None = None
    distribution_model: GaussianMixture
    n_components: int

    @property
    def min_var_retained(self) -> float | None:
        if self.signal_tuner is not None:
            return self.signal_tuner.min_var_retained
        return None

    class Decorators:
        @staticmethod
        def _tune_signal(fit_model_function: F) -> F:
            """Optionally tune a signal with PCA."""

            @wraps(fit_model_function)
            def wrapper(self: type[GAF], df: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
                if (signal_tuner := self.signal_tuner) is not None:
                    df = signal_tuner.tune(df, orig_space=False)

                return fit_model_function(self, df, *args, **kwargs)

            return cast(F, wrapper)

    @classmethod
    def initialize(
        cls,
        pca_model: PCALatentSpace | None = None,
        min_var_retained: float | None = None,
        n_components: int = 1,
        covariance_type: str = "full",
        **kwargs: Any,
    ) -> Self:
        distribution_model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            **kwargs,
        )

        return cls(
            signal_tuner=(
                SignalTuner.initialize(pca_model=pca_model, min_var_retained=min_var_retained)
                if min_var_retained is not None
                else None
            ),
            distribution_model=distribution_model,
            n_components=n_components,
        )

    @Decorators._tune_signal
    def fit(
        self,
        data_matrix: pd.DataFrame,
        gaussian_means_specifier: Any | None = None,
    ) -> Self:
        """Fit the gaussian model.

        Args:
            data_matrix:
                A `pandas.DataFrame` to fit the model.

            gaussian_means_specifier:
                An index level identifier to calculate gaussian initialization means from the `data_matrix`.

        Returns:
            Self
        """

        if gaussian_means_specifier is not None:
            logger.info(
                f"Initializing {self.n_components} gaussian kernel means on {gaussian_means_specifier} centroids"
            )
            gaussian_means_init = data_matrix.groupby(level=gaussian_means_specifier).mean()
            if (n_means := len(gaussian_means_init)) != self.n_components:
                raise ValueError(
                    "The number of mixture initialization means must match the number of mixture components, "
                    f"got {n_means} means for {self.n_components} components"
                )
            self.distribution_model.means_init = gaussian_means_init

        self.distribution_model.fit(data_matrix)

        return self

    @Decorators._tune_signal
    def score_samples(self, data_matrix: pd.DataFrame) -> pd.Series:
        """Get the probabilities of data points under the distribution model."""

        log_probability = self.distribution_model.score_samples(data_matrix)

        return pd.Series(log_probability, index=data_matrix.index, name="log_probability")

    @Decorators._tune_signal
    def compute_bic(self, data_matrix: pd.DataFrame) -> float:
        """Returns Bayesian information criterion for the model on the data
        matrix."""

        return self.distribution_model.bic(data_matrix)

    @Decorators._tune_signal
    def score(self, data_matrix: pd.DataFrame) -> float:
        """Returns the average log-likelihood of the model given the data."""

        return self.distribution_model.score(data_matrix)

    @Decorators._tune_signal
    def predict_components_proba(self, data_matrix: pd.DataFrame) -> pd.Series:
        """Returns components densities of each sample."""
        components_densities = self.distribution_model.predict_proba(data_matrix)

        return pd.Series(
            list(components_densities),
            index=data_matrix.index,
            name="mixture_component",
        )

    def predict_component(self, data_matrix: pd.DataFrame) -> pd.Series:
        """Get the most likely gaussian the data belongs to."""
        components_densities = self.predict_components_proba(data_matrix)

        return components_densities.apply(np.argmax)

    def get_anomalies(self, data_matrix: pd.DataFrame, threshold: float) -> NDArray[Any]:
        """Returns anomalies data points with log-probability < to threshold."""

        log_probability = self.score_samples(data_matrix)

        return np.where(log_probability < np.log(threshold))[0]


class GMMHyperparameterTuner(BaseModel):
    """Wrapper class for hyperparameter tuning of GaussianAnomalyQuantifier model."""

    pca_model: PCALatentSpace | None = None
    min_var_retained: float = 0.8
    max_n_components: int = 10
    covariance_type: list[str] = COVARIANCE_TYPE
    gaussian_means_specifier: Any | None = None

    @field_validator("covariance_type")
    def validate_covariance_type(cls, value: list[str] | str) -> list[str]:
        if isinstance(value, str):
            return [value]
        return value

    def find_best_param(
        self,
        data_matrix: pd.DataFrame,
        n_components: NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[int, str]:
        """Performs selection of the best set of hyperparameters by
        minimizing the bic.

        Args:
            data_matrix: A matrix of shape (n_observation, n_variables) to fit the model.
            n_components: The number of mixture components. If None, tuning is done using a full grid of (1, max_n_components).
            **kwargs: A dictionary of {key: value} arguments accepted by :func:`GaussianAnomalyQuantifier.initialize()`

        Returns:
            A tuple of optimal (n_components, covariance_type).

        """
        if n_components is None:
            n_components_array = np.arange(1, min(self.max_n_components, data_matrix.shape[0]) + 1)
        elif np.asarray(n_components).size == 1:
            n_components_array = np.array([n_components])
        else:
            n_components_array = np.asarray(n_components)

        def _fit_model(
            model_init: Callable[..., Any],
            n: int,
            covariance: str,
        ) -> tuple[float, GaussianAnomalyQuantifier | None]:
            try:
                model = model_init(n_components=n, covariance_type=covariance).fit(
                    data_matrix, gaussian_means_specifier=self.gaussian_means_specifier
                )
                bic = model.compute_bic(data_matrix)
                return bic, model
            except ValueError as exc:
                # Fitting failed for some reason
                logger.exception(exc)
                return np.inf, None

        logger.info(
            f"Finding the best parameters in the {self.covariance_type} x"
            f" {n_components_array} components grid"
        )

        model_init = partial(
            GaussianAnomalyQuantifier.initialize,
            pca_model=self.pca_model,
            min_var_retained=self.min_var_retained,
            **kwargs,
        )
        # Parallelize model fitting
        results = Parallel(n_jobs=-1)(
            delayed(_fit_model)(model_init, n, covariance)
            for n in n_components_array
            for covariance in self.covariance_type
        )

        # Find the best model and its parameters
        best_bic = np.inf
        best_model = None
        for bic, model in results:
            if bic < best_bic:
                best_bic = bic
                best_model = model

        if best_model is None:
            raise ModelFittingFailure("Impossible to fit a model")

        return (
            best_model.distribution_model.n_components,
            best_model.distribution_model.covariance_type,
        )
