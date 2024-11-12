import numpy as np
import pytest
from sklearn.datasets import load_iris

from mltoolbox.anomaly_detection import (
    GaussianAnomalyQuantifier,
    GMMHyperparameterTuner,
)
from mltoolbox.exceptions import ModelFittingFailure
from mltoolbox.latent_space import PCALatentSpace, SignalTuner

MEANS_INIT_SPECIFIER = "target"
iris = (
    load_iris(as_frame=True)["data"]
    .join(load_iris(as_frame=True)[MEANS_INIT_SPECIFIER])
    .set_index(MEANS_INIT_SPECIFIER)
)


class TestGaussianAnomalyQuantifier:
    def test_valid_construction(self):
        model = GaussianAnomalyQuantifier.initialize(n_components=1, random_state=1)
        assert model.signal_tuner is None
        assert model.min_var_retained is None

    def test_model_specification(self, caplog):
        model = GaussianAnomalyQuantifier.initialize(min_var_retained=0.7)
        assert isinstance(model.signal_tuner.pca_model, PCALatentSpace)
        assert model.min_var_retained == 0.7

    def test_model_fit(self):
        model = GaussianAnomalyQuantifier.initialize(min_var_retained=0.95, random_state=1).fit(
            iris
        )
        assert model.signal_tuner.n_pcomponents == 2

    def test_not_enough_components(self, caplog):
        model = GaussianAnomalyQuantifier.initialize(
            pca_model=PCALatentSpace.initialize(n_components=1),
            min_var_retained=0.95,
            random_state=1,
        ).fit(iris)
        assert model.signal_tuner.n_pcomponents == 1
        assert (
            "95.0% variance not reached with available components, consider setting PCA"
            " n_components > 1" in caplog.text
        )

    def test_gaussian_means_specifier(self, caplog):
        model = GaussianAnomalyQuantifier.initialize(
            n_components=3,
            min_var_retained=0.95,
            random_state=1,
        ).fit(
            iris,
            gaussian_means_specifier=MEANS_INIT_SPECIFIER,
        )

        expected_init_means = (
            SignalTuner.initialize(min_var_retained=0.95)
            .tune(iris)
            .groupby(level=MEANS_INIT_SPECIFIER)
            .mean()
        )
        assert np.allclose(
            model.distribution_model.means_init,
            expected_init_means,
        )
        assert np.allclose(
            model.distribution_model.means_,
            [[-2.64241546, 0.19088505], [0.4338458, -0.21251283], [1.77550724, -0.03550646]],
        )
        assert (
            f"Initializing 3 gaussian kernel means on {MEANS_INIT_SPECIFIER} centroids"
            in caplog.text
        )

        # Number of means does not match n_kernels
        with pytest.raises(ValueError) as excinfo:
            GaussianAnomalyQuantifier.initialize(
                n_components=1,
                min_var_retained=0.95,
                random_state=1,
            ).fit(
                iris,
                gaussian_means_specifier=MEANS_INIT_SPECIFIER,
            )
        assert (
            "The number of mixture initialization means must match the number of mixture components, "
            f"got 3 means for 1 components" in str(excinfo.value)
        )

    def test_gaussian_means_specifier_not_found(self):
        with pytest.raises(ValueError):
            GaussianAnomalyQuantifier.initialize(
                min_var_retained=0.95,
                random_state=1,
            ).fit(
                iris,
                gaussian_means_specifier="non_existent_identifier",
            )


class TestHyperparameterTuner:
    def test_full_grid(self, caplog):
        n_components, covariance_type = GMMHyperparameterTuner(max_n_components=10).find_best_param(
            iris, random_state=1
        )
        assert (
            "Finding the best parameters in the ['spherical', 'tied', 'diag',"
            " 'full'] x [ 1  2  3  4  5  6  7  8  9 10] components grid" in caplog.text
        )
        assert n_components == 2
        assert covariance_type in ["full", "spherical"]

    def test_covariance_grid(self, caplog):
        n_components, covariance_type = GMMHyperparameterTuner().find_best_param(
            iris, n_components=1, random_state=1
        )
        assert (
            "Finding the best parameters in the ['spherical', 'tied', 'diag',"
            " 'full'] x [1] components grid" in caplog.text
        )
        assert n_components == 1
        assert covariance_type == "spherical"

    def test_gaussian_means_specifier(self, caplog):
        n_components, covariance_type = GMMHyperparameterTuner(
            gaussian_means_specifier=MEANS_INIT_SPECIFIER
        ).find_best_param(iris, n_components=3, random_state=1)
        assert n_components == 3
        assert (
            "Finding the best parameters in the ['spherical', 'tied', 'diag',"
            " 'full'] x [3] components grid" in caplog.text
        )

    def test_gaussian_means_specifier_inconsistent_for_some_fit(self, caplog):
        n_components, _ = GMMHyperparameterTuner(
            gaussian_means_specifier=MEANS_INIT_SPECIFIER, max_n_components=3
        ).find_best_param(iris, random_state=1)
        assert n_components == 3

    def test_inconsistent_gaussian_means_specifier_for_all_fit(self, caplog):
        with pytest.raises(ModelFittingFailure) as excinfo:
            GMMHyperparameterTuner(
                gaussian_means_specifier=MEANS_INIT_SPECIFIER, max_n_components=2
            ).find_best_param(iris, random_state=1)
