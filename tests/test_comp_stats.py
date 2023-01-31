import numpy as np
import pytest

from mltoolbox.comp_stats import bootstrap_mean_ci, bootstrap_corr_pval

class TestBoostrapMeanCI:

    def test_valid_input(self):
        data = [1, 2, 3, 4, 5]
        ci = bootstrap_mean_ci(data, n_boot=100, ci=0.9)
        assert ci.shape == (2,)

    def test_invalid_input(self):
        data = [1, 2, 'a', 4, 5]
        with pytest.raises(TypeError):
            bootstrap_mean_ci(data, n_boot=100, ci=0.9)

    def test_scalar_input(self):
        data = 1
        with pytest.raises(ValueError):
            bootstrap_mean_ci(data, n_boot=100)

    def test_invalid_ci_level(self):
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            bootstrap_mean_ci(data, n_boot=100, ci=1.2)


class TestBoostrapCorrPval:

    def test_correlation(self):
        data_series1 = [1, 2, 3, 4, 5]
        data_series2 = [1, 2, 3, 4, 5]
        expected_result = 0.0
        n_boot = 100
        assert np.isclose(bootstrap_corr_pval(data_series1, data_series2, n_boot), expected_result)

    def test_anticorrelation(self):
        data_series1 = [1, 2, 3, 4, 5]
        data_series2 = [5, 4, 3, 2, 1]
        expected_result = 0.0
        n_boot = 100
        assert np.isclose(bootstrap_corr_pval(data_series1, data_series2, n_boot), expected_result)