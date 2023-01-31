"""Main classes of the Latent space subpackage"""
from functools import wraps

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class LatentSpace(object):
    """Performs dimensionality reduction of a dataset.

    Attributes:
        model:
            The model used for dimensionality reduction. PCA by default.
        n_components:
            The number of components for latent space model. None by default.
        standardize:
            Wether to standardize data or not (i.e., compute model on covariance or correlation matrix).
    """

    def __init__(self, model='pca', n_components=None, standardize=False,
                 perplexity=10):
        self._algorithm = model
        self.n_components = n_components
        self.standardize = standardize

        if self.standardize:
            self._standard_scaler = StandardScaler()
        else:
            self._standard_scaler = None

        if self._algorithm == 'pca':
            self.model = PCA(n_components=self.n_components)
            self._score_dim = 'PC%d'
            self.fit = self.fit_pca
            self.project_data = self._project_in_pc_space
        elif self._algorithm == 'tsne':
            if not n_components:
                self.n_components = 2
            self.model = TSNE(n_components=self.n_components,
                              perplexity=perplexity,
                              init='pca')
            self._score_dim = 'Dimension %d'
            self.fit = self.fit_tsne
            self.project_data = self._project_in_tsne_space

    class Decorators(object):
        """Class to store the decorators."""

        @staticmethod
        def _fit_standardizer(fit_model_function):
            """"Optionnaly fit a standardizer to the data."""

            @wraps(fit_model_function)
            def inner(self, df, *args, **kwargs):
                if self._standard_scaler:
                    df = pd.DataFrame(self._standard_scaler.fit_transform(df),
                                      index=df.index, columns=df.columns)
                    return fit_model_function(self, df, *args, **kwargs)

                return fit_model_function(self, df, *args, **kwargs)

            return inner

        @staticmethod
        def _apply_standardizer(project_model_function):
            """Optionally transforms data with a standardizer."""

            @wraps(project_model_function)
            def inner(self, df, *args, **kwargs):
                if self._standard_scaler:
                    df = pd.DataFrame(self._standard_scaler.transform(df),
                                      index=df.index, columns=df.columns)
                    return project_model_function(self, df, *args, **kwargs)

                return project_model_function(self, df, *args, **kwargs)

            return inner

    def _to_score_df(self, score_matrix, index):
        """Wraps the score matrix in a DataFrame."""
        score_df = pd.DataFrame(score_matrix,
                                columns=[self._score_dim % (i + 1) for i in
                                         range(score_matrix.shape[1])],
                                index=index)
        return score_df

    def fit_transform(self, data_matrix):
        """Fit a model and returns the projected data."""
        self.fit(data_matrix)
        scores = self.project_data(data_matrix)

        return self._to_score_df(scores, data_matrix.index)

    @Decorators._fit_standardizer
    def fit_pca(self, data_matrix):
        """Fit the model to the data_matrix."""
        # Keep orig var
        self.orig_vars = np.array(data_matrix.columns)
        # Fit the scaler and the model
        self.model.fit(data_matrix)

        # Calculate the loadings
        self.loadings_ = self.model.components_.T * np.sqrt(
            self.model.explained_variance_)

        # Calculates correlation matrix between PC and original variables
        if self.standardize:
            # correlation between variable and the PC is given by the corresponding loading
            self.pc_var_correlations_ = self.loadings_
        else:
            # Divide by original variables standard deviation
            self.pc_var_correlations_ = self.loadings_ / data_matrix.std().values.reshape(
                -1, 1)

        return self

    @Decorators._fit_standardizer
    def fit_tsne(self, data_matrix):
        """Fit the model to the data_matrix."""
        # Keep orig var
        self.orig_vars = np.array(data_matrix.columns)
        # Fit the scaler and the model
        self.model.fit(data_matrix)

        return self

    @Decorators._apply_standardizer
    def _project_in_pc_space(self, data_matrix, *args):
        """Projects in PC space."""
        # Get the scores
        scores = self.model.transform(data_matrix)

        return self._to_score_df(scores, data_matrix.index)

    @Decorators._apply_standardizer
    def _project_in_tsne_space(self, data_matrix, params_dict={}):
        """Projects in TSNE space."""
        # Get the scores
        scores = self.model.set_params(**params_dict).fit_transform(
            data_matrix)

        return self._to_score_df(scores, data_matrix.index)

    def get_sorted_loadings(self, pc=1):
        """Returns the loadings of the variables onto the selected PC sorted by decreasing magnitude."""

        sorted_loading = np.sort(np.abs(self.loadings_[:, pc - 1]))[::-1]
        var_idx = self.orig_vars[
            np.argsort(np.abs(self.loadings_[:, pc - 1]))[::-1]]

        return pd.Series(sorted_loading,
                         index=pd.Index(var_idx, name='orig_variable'))

    def reconstruct_variables(self, data, var_explained):
        """Reconstructs original variable values from principal components.
        We sum squared correlation of the variables until reaching the desired amount of explained variance.
        Note that initial preprocessing is not reverted (i.e., data stays centered on 0 and SD=1 if standardization was applied).

        """
        # Get the scores
        pca_scores = self.project_data(data)

        # Loop through variables to denoise/reconstruct them
        reconstructed_variables = {}
        for variable in self.orig_vars:
            var_idx = np.where(self.orig_vars == variable)[0][0]
            sorted_pc_index = np.argsort(
                self.pc_var_correlations_[var_idx, :])[::-1]
            var_cumsum = 0
            pc_idx = []
            for idx in sorted_pc_index:
                var_cumsum += self.pc_var_correlations_[var_idx, idx] ** 2
                pc_idx.append(idx)
                if var_cumsum > (var_explained / 100):
                    break
            # Select the PCs that explain x percent of variance
            selected_pcs = self.model.components_[pc_idx, var_idx]
            reconstructed_variables[variable] = np.dot(
                pca_scores.iloc[:, pc_idx],
                selected_pcs)

        reconstructed_df = pd.DataFrame(reconstructed_variables,
                                        index=data.index,
                                        columns=data.columns)
        # "Decentering"
        if not self.standardize:
            reconstructed_df = reconstructed_df + self.model.mean_

        return reconstructed_df
