import pytest
from sklearn.datasets import load_iris

from mltoolbox.anomaly_detection import GaussianAnomalyQuantifier
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

    def test_model(self):
        model = GaussianAnomalyQuantifier.initialize(var_explained=95).fit(iris)
        assert model.n_pcomponents_ == 2
