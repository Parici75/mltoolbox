"""Main classes of the anomaly_detection subpackage."""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pydantic import field_validator, model_validator
from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

from mltoolbox.base_model import BaseModel
from mltoolbox.exceptions import ModelFittingFailure
from mltoolbox.latent_space import PCALatentSpace

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
GAF = TypeVar("GAF", bound="GaussianAnomalyQuantifier")


class GaussianAnomalyQuantifier(BaseModel):
    """Wrapper class for Gaussian modelling of data distribution."""

    pca_model: PCALatentSpace | None = None
    distribution_model: GaussianMixture
    var_explained: float | None
    n_components: int
    var_retained_: float | None = None
    n_pcomponents_: int | None = None

    @model_validator(mode="before")
    def validate_model_consistency(cls, data: dict[str, Any]) -> dict[str, Any]:
        if (var_explained := data.get("var_explained")) is not None and data.get(
            "pca_model"
        ) is None:
            data["pca_model"] = PCALatentSpace.initialize()
            logger.debug(f"Setting a `PCALatentSpace` with `var-explained`={var_explained}")
        return data

    class Decorators:
        @staticmethod
        def _fit_pca(fit_model_function: F) -> F:
            """Optionally fit a PCA."""

            @wraps(fit_model_function)
            def wrapper(self: type[GAF], df: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
                if (pca_model := self.pca_model) is not None:
                    # Fit the PCA first
                    try:
                        check_is_fitted(pca_model.model)
                    except NotFittedError:
                        scores = pca_model.fit_and_project_data(df)
                    else:
                        scores = pca_model.project_data(df)

                    # Find the n_pcomponents which explain enough variance
                    cumulative_variance = pca_model.model.explained_variance_ratio_.cumsum() * 100
                    try:
                        component_idx = np.where(cumulative_variance >= self.var_explained)[0][0]
                    except IndexError:
                        logger.warning(
                            f"{self.var_explained} variance not reached with available components,"
                            f" consider setting PCA n_components > {pca_model.n_components}"
                        )
                        # We take all components
                        component_idx = len(cumulative_variance) - 1

                    self.var_retained_ = cumulative_variance[component_idx]
                    self.n_pcomponents_ = component_idx + 1

                    return fit_model_function(
                        self, scores.iloc[:, : self.n_pcomponents_], *args, **kwargs
                    )
                else:
                    return fit_model_function(self, df, *args, **kwargs)

            return cast(F, wrapper)

        @staticmethod
        def _apply_pca(project_model_function: F) -> F:
            """Optionally transforms in PC space."""

            @wraps(project_model_function)
            def wrapper(self: type[GAF], df: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
                if (pca_model := self.pca_model) is not None:
                    # Transform in PCs
                    scores = pca_model.project_data(df)

                    return project_model_function(
                        self, scores.iloc[:, : self.n_pcomponents_], *args, **kwargs
                    )
                else:
                    return project_model_function(self, df, *args, **kwargs)

            return cast(F, wrapper)

    @classmethod
    def initialize(
        cls,
        pca_model: PCALatentSpace | None = None,
        var_explained: float | None = None,
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
            pca_model=pca_model,
            distribution_model=distribution_model,
            var_explained=var_explained,
            n_components=n_components,
        )

    @Decorators._fit_pca
    def fit(self, data_matrix: pd.DataFrame) -> Self:
        """Fit the gaussian model."""

        self.distribution_model.fit(data_matrix)

        return self

    @Decorators._apply_pca
    def score_samples(self, data_matrix: pd.DataFrame) -> pd.Series:
        """Get the probabilities of data points under the distribution modeling."""

        log_probability = self.distribution_model.score_samples(data_matrix)

        return pd.Series(log_probability, index=data_matrix.index, name="log_probability")

    @Decorators._apply_pca
    def compute_bic(self, data_matrix: pd.DataFrame) -> float:
        """Returns Bayesian information criterion for the model on the data
        matrix."""

        return self.distribution_model.bic(data_matrix)

    @Decorators._apply_pca
    def score(self, data_matrix: pd.DataFrame) -> float:
        """Returns the average log-likelihood of the model given the data."""

        return self.distribution_model.score(data_matrix)

    @Decorators._apply_pca
    def predict_component(self, data_matrix: pd.DataFrame) -> pd.Series:
        """Get the most likely gaussian the data belongs to."""
        predicted_components = self.distribution_model.predict(data_matrix)

        return pd.Series(predicted_components, index=data_matrix.index, name="mixture_component")

    def get_anomalies(self, data_matrix: pd.DataFrame, threshold: float) -> NDArray[Any]:
        """Returns anomalies data points with log-probability < to threshold."""
        # Get the scores and probabilities
        log_probability = self.score_samples(data_matrix)

        return np.where(log_probability < np.log(threshold))[0]


class GMMHyperparameterTuner(BaseModel):
    """Wrapper class for hyperparameter tuning of GaussianAnomalyQuantifier model."""

    pca_model: PCALatentSpace | None = None
    var_explained: float = 80
    max_n_components: int = 10
    covariance_type: list[str] = ["spherical", "tied", "diag", "full"]

    @field_validator("covariance_type")
    def validate_covariance_type(cls, value: list[str] | str) -> list[str]:
        if isinstance(value, str):
            return [value]
        return value

    def find_best_param(
        self,
        data_matrix: pd.DataFrame,
        n_components: int | None = None,
        **kwargs: Any,
    ) -> tuple[int, str]:
        """Performs selection of the best set of hyperparameters by
        minimizing the bic.

        Args:
            data_matrix:
                A matrix of shape (n_observation, n_variables) t fit the model.
            n_components:
                The number of mixture components
                If None, tuning is done using a full grid of (1, max_n_components).
            **kwargs:
                Any (key: value) argument accepted by :func:`GaussianAnomalyQuantifier.initialize()`

        Returns:
            A tuple of optimal (n_components, covariance_type)

        """
        # Parse arguments
        if n_components is None:
            n_components_array = np.arange(1, min(self.max_n_components, data_matrix.shape[0]) + 1)
        else:
            n_components_array = np.array([n_components])

        def _fit_model(n: int, covariance: str) -> tuple[float, GaussianAnomalyQuantifier | None]:
            try:
                model = GaussianAnomalyQuantifier.initialize(
                    pca_model=self.pca_model,
                    var_explained=self.var_explained,
                    n_components=n,
                    covariance_type=covariance,
                    **kwargs,
                ).fit(data_matrix)
                bic = model.compute_bic(data_matrix)
                return bic, model
            except ValueError as exc:
                # Fitting failed for some reason
                logger.error(exc)
                return np.inf, None

        logger.info(
            f"Finding the best parameters in the {self.covariance_type} x"
            f" {n_components_array} components grid"
        )
        # Parallelize model fitting
        results = Parallel(n_jobs=-1)(
            delayed(_fit_model)(n, covariance)
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
