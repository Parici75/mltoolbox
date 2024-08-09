import logging

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError

from mltoolbox.latent_space import PCALatentSpace, SignalTuner, TSNELatentSpace

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
        assert projected_data.shape[1] == min(iris.shape)


class TestTSNELatentSpace:
    def test_valid_construction(self):
        model = TSNELatentSpace.initialize(n_components=3)
        assert model.n_components == 3
        assert model.standard_scaler is None

    def test_project_data(self):
        projected_data = TSNELatentSpace.initialize(standardize=False).fit_and_project_data(iris)
        assert projected_data.shape[0] == iris.shape[0]
        assert projected_data.shape[1] == 2


class TestDenoiser:
    def test_tuning(self):
        denoiser = SignalTuner.initialize(min_var_retained=0.8)
        assert denoiser.tune(iris).shape[0] == iris.shape[0]
        assert denoiser.n_pcomponents_ == 1
        assert denoiser.tune(iris, orig_space=True).shape == iris.shape

    def test_no_tuning(self):
        denoiser = SignalTuner.initialize(min_var_retained=1)
        denoiser.tune(iris)
        assert denoiser.n_pcomponents_ == 4

    def test_precomputed_pca(self, caplog):
        denoiser = SignalTuner.initialize(
            min_var_retained=0.95, pca_model=PCALatentSpace.initialize(n_components=1)
        )
        denoiser.tune(iris)
        assert denoiser.n_pcomponents_ == 1
        assert (
            f"95.0% variance not reached with available components, consider setting PCA n_components > {denoiser.pca_model.model.n_components}"
            in caplog.text
        )
