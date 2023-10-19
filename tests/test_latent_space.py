import numpy as np
from sklearn.datasets import load_iris

from mltoolbox.latent_space import PCALatentSpace, TSNELatentSpace

iris = load_iris(as_frame=True)["data"]


class TestPCALatentSpace:
    def test_valid_construction(self):
        model = PCALatentSpace.initialize(n_components=8, standardize=True)
        assert model.n_components == 8
        assert model.standard_scaler is not None

    def test_fit(self):
        model = PCALatentSpace.initialize(standardize=True).fit(iris)
        assert model.loadings is not None
        assert (model.pc_var_correlations == model.loadings).all()

    def test_reconstruct_variables(self):
        model = PCALatentSpace.initialize().fit(iris)
        assert not model.standardize
        assert np.allclose(
            PCALatentSpace.initialize().fit(iris).reconstruct_variables(iris, 100), iris
        )

    def test_project_data(self):
        model = PCALatentSpace.initialize(standardize=False).fit(iris)
        projected_data = model.project_data(iris)
        assert projected_data.shape[0] == iris.shape[0]
        assert projected_data.shape[1] == np.min(iris.shape)


class TestTSNELatentSpace:
    def test_valid_construction(self):
        model = TSNELatentSpace.initialize(n_components=3)
        assert model.n_components == 3
        assert model.standard_scaler is None

    def test_project_data(self):
        projected_data = TSNELatentSpace.initialize(standardize=False).fit_and_project_data(iris)
        assert projected_data.shape[0] == iris.shape[0]
        assert projected_data.shape[1] == 2
