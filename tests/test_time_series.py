import numpy as np
import pytest

from mltoolbox.time_series import ccf

class TestCcf:
    def test_ccf_without_ci(self):
        # Generates two waveform of period T=5
        fs = 20
        f = 1 / 5
        t = 10

        samples = np.arange(t * fs) / fs
        x_signal = np.sin(2 * np.pi * f * samples)
        y_signal = np.sin(2 * np.pi * f * samples + np.pi)  # Shift y_signal by half a period

        ccf_coeffs, bootstrap_ci = ccf(x_signal, y_signal)
        # Correlation coefficient should be maximum when signal is shifted of half a period
        assert np.argmax(ccf_coeffs) == np.round(1 / f / 2 * fs)
        assert bootstrap_ci is None

    def test_ccf_with_ci(self):
        x_signal = np.random.rand(100)
        y_signal = np.random.rand(100)

        lag = 10
        ccf_coeffs, bootstrap_ci = ccf(x_signal, y_signal, lag=lag, ci=0.95, n_boot=10)
        assert bootstrap_ci.shape == (lag, 2)

    def test_invalid_data_input(self, caplog):
        with pytest.raises(ValueError):
            x_signal = np.random.rand(5)
            y_signal = np.random.rand(5)
            lag = 10

            ccf(x_signal, y_signal, lag=lag)
            assert ("Invalid value for lag" in caplog.text)

        with pytest.raises(ValueError):
            x_signal = np.random.rand(10)
            y_signal = np.random.rand(5)
            lag = 10

            ccf(x_signal, y_signal, lag=lag)
            assert ("Data arrays must have the same length"in caplog.text)


