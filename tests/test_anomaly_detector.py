from sklearn.datasets import load_iris

from mltoolbox.anomaly_detection import (
    GaussianAnomalyQuantifier,
    GMMHyperparameterTuner,
)
from mltoolbox.latent_space import PCALatentSpace

iris = load_iris(as_frame=True)["data"]


class TestGaussianAnomalyQuantifier:
    def test_valid_construction(self):
        model = GaussianAnomalyQuantifier.initialize(n_components=1)
        assert model.var_explained is None
        assert model.var_retained_ is None
        assert model.n_pcomponents_ is None

    def test_model_specification(self, caplog):
        model = GaussianAnomalyQuantifier.initialize(var_explained=70)
        assert isinstance(model.pca_model, PCALatentSpace)
        assert "Setting a `PCALatentSpace` with `var-explained`=70" in caplog.text

    def test_model_fit(self):
        model = GaussianAnomalyQuantifier.initialize(var_explained=95).fit(iris)
        assert model.n_pcomponents_ == 2


class TestHyperparameterTuner:
    def test_full_grid(self, caplog):
        n_components, covariance_type = GMMHyperparameterTuner(max_n_components=10).find_best_param(
            iris
        )
        assert (
            "Finding the best parameters in the ['spherical', 'tied', 'diag',"
            " 'full'] x [ 1  2  3  4  5  6  7  8  9 10] components grid"
            in caplog.text
        )
        assert n_components == 2
        assert covariance_type == "full"

    def test_coviarance_grid(self, caplog):
        n_components, covariance_type = GMMHyperparameterTuner().find_best_param(
            iris, n_components=1
        )
        assert (
            "Finding the best parameters in the ['spherical', 'tied', 'diag',"
            " 'full'] x [1] components grid"
            in caplog.text
        )
        assert n_components == 1
        assert covariance_type == "spherical"
