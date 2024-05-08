from sklearn.datasets import load_iris

from mltoolbox.anomaly_detection import (
    GaussianAnomalyQuantifier,
    GMMHyperparameterTuner,
)
from mltoolbox.latent_space import PCALatentSpace

iris = load_iris(as_frame=True)["data"]


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
        assert model.signal_tuner.n_pcomponents_ == 2

    def test_not_enough_components(self, caplog):
        model = GaussianAnomalyQuantifier.initialize(
            pca_model=PCALatentSpace.initialize(n_components=1),
            min_var_retained=0.95,
            random_state=1,
        ).fit(iris)
        assert model.signal_tuner.n_pcomponents_ == 1
        assert (
            "95.0% variance not reached with available components, consider setting PCA"
            " n_components > 1" in caplog.text
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
        assert covariance_type == "full"

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
