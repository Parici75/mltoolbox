"""Main classes of the anomaly_detection subpackage."""

import logging

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from mltoolbox.anomaly_detection.exceptions import ModelFittingFailure

logger = logging.getLogger(__name__)


class GaussianAnomalyQuantifier(object):
    """Wrapper class for Gaussian modelling of data distribution."""

    class Decorators(object):
        """Class to store the decorators."""

        @staticmethod
        def _fit_pca(fit_model_function):
            """Optionally fit a PCA."""

            def inner(self, df, *args, **kwargs):
                if self.pca_model:
                    # Fit the PCA first
                    try:
                        check_is_fitted(self.pca_model)
                    except NotFittedError:
                        scores = self.pca_model.fit_transform(df)
                    else:
                        scores = self.pca_model.project_data(df)

                    # Find the n_pcomponents which explain the variance
                    cum_variance = (
                        self.pca_model.model.explained_variance_ratio_.cumsum() * 100
                    )
                    try:
                        component_idx = np.where(cum_variance >= self.var_explained)[0][
                            0
                        ]
                    except IndexError:
                        logging.info(
                            "%d variance not reached with available "
                            "components, consider setting PCA n_components > %d"
                            % (self.var_explained, self.pca_model.n_components)
                        )
                        # We take all components
                        component_idx = len(cum_variance) - 1

                    self.var_retained_ = cum_variance[component_idx]
                    self.n_pcomponents_ = component_idx + 1

                    return fit_model_function(
                        self, scores.iloc[:, : self.n_pcomponents_], *args, **kwargs
                    )
                else:
                    return fit_model_function(self, df, *args, **kwargs)

            return inner

        @staticmethod
        def _apply_pca(project_model_function):
            """Optionally transforms in PC space."""

            def inner(self, df, *args, **kwargs):
                if self.pca_model:
                    # Transform in PCs
                    scores = self.pca_model.project_data(df)

                    return project_model_function(
                        self, scores.iloc[:, : self.n_pcomponents_], *args, **kwargs
                    )
                else:
                    return project_model_function(self, df, *args, **kwargs)

            return inner

    @property
    def n_components(self):
        return self.distribution_model.n_components

    def __init__(
        self,
        pca_model=None,
        var_explained=80,
        n_components=1,
        covariance_type="full",
        init_params="kmeans",
        n_init=1,
        **kwargs,
    ):
        """Initializes the anomaly detector object."""
        self.pca_model = pca_model
        if self.pca_model is not None:
            self.var_explained = var_explained
        else:
            self.var_explained = None

        # Initialize parameters
        self.var_retained_ = None
        self.n_pcomponents_ = None

        # Initialize mixture model
        self.distribution_model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            init_params=init_params,
            n_init=n_init,
            **kwargs,
        )

    @Decorators._fit_pca
    def fit(self, data_matrix):
        """Fit the gaussian model."""

        self.distribution_model.fit(data_matrix)

        return self

    @Decorators._apply_pca
    def score_samples(self, data_matrix):
        """Get the probability of the scores under the distribution
        computed."""

        log_probability = self.distribution_model.score_samples(data_matrix)

        return pd.Series(
            log_probability, index=data_matrix.index, name="log_probability"
        )

    @Decorators._apply_pca
    def compute_bic(self, data_matrix):
        """Returns Bayesian information criterion for the model on the data
        matrix."""

        return self.distribution_model.bic(data_matrix)

    @Decorators._apply_pca
    def score(self, data_matrix):
        """Returns the average log-likelihood of the model given the data."""

        return self.distribution_model.score(data_matrix)

    @Decorators._apply_pca
    def predict_component(self, data_matrix):
        """Get the most likely gaussian the data belongs to."""
        predicted_components = self.distribution_model.predict(data_matrix)

        return pd.Series(
            predicted_components, index=data_matrix.index, name="mixture_component"
        )

    def get_anomalies(self, data_matrix, threshold):
        """Returns anomalies"""
        # Get the scores and probabilities
        log_probability = self.score_samples(data_matrix)

        return np.where(log_probability < np.log(threshold))[0]


class HyperparameterTuner(object):
    """Wrapper class for hyperparameter tuning of GaussianAnomalyQuantifier
    model."""

    def __init__(
        self,
        pca_model=None,
        var_explained=80,
        max_n_components=10,
        cv_types=["spherical", "tied", "diag", "full"],
    ):
        """Initializes the tuner."""
        self.pca_model = pca_model
        self.var_explained = var_explained
        self.max_n_components = max_n_components
        self.cv_types = cv_types

    def find_best_param(self, data_matrix, n_components=None):
        """Performs selection of the best set of hyperparameters by
        minimizing the bic.

        Args:
            data_matrix:
                A matrix of shape (n_observation, n_variables) t fit the model
            n_components:
                The number of mixture components. If None, tuning is done with an array of [1:max(self.max_n_components,
                                              data_matrix.shape[0]))]

        Returns:
            A tuple of optimal (n_components, covariance_type)

        """
        lowest_bic = np.infty
        bic = []

        # Parse arguments
        if n_components is None:
            n_components_array = np.arange(
                1, min(self.max_n_components, data_matrix.shape[0]) + 1
            )
        else:
            n_components_array = [n_components]

        logger.info(
            f"Finding the best parameters in the {self.cv_types} x {n_components_array} components grid"
        )

        # Loop
        for cv_type in self.cv_types:
            # Loop through component configuration
            for n in n_components_array:
                # Fit a Gaussian mixture with EM
                logger.debug(
                    f"Fitting model with {n} components and {cv_type} covariance matrix"
                )
                try:
                    model = GaussianAnomalyQuantifier(
                        pca_model=self.pca_model,
                        var_explained=self.var_explained,
                        n_components=n,
                        covariance_type=cv_type,
                    ).fit(data_matrix)
                except ValueError as exc:
                    # Fitting failed for some reasons, so we continue to loop
                    logger.info(exc)
                    continue
                except Exception:
                    raise

                # Calculate bic and update best model
                bic.append(model.compute_bic(data_matrix))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_model = model

        # Check that we manage to fit at least one model
        if len(bic) == 0:
            raise ModelFittingFailure("Impossible to fit a model")

        return (
            best_model.distribution_model.n_components,
            best_model.distribution_model.covariance_type,
        )
