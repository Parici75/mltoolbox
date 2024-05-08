"""Main classes of the Latent space subpackage."""

import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, Self, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from openTSNE.sklearn import TSNE
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from mltoolbox.base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    PCA = "pca"
    TSNE = "tsne"


F = TypeVar("F", bound=Callable[..., Any])

STANDARD_MIN_VAR_RETAINED = 0.8


class LatentSpace(BaseModel, metaclass=ABCMeta):
    n_components: int | None = None
    standardize: bool
    standard_scaler: StandardScaler | None

    @property
    def orig_vars(self) -> list[str]:
        return self.model.feature_names_in_  # type: ignore

    class Decorators:
        @staticmethod
        def _projected_data_to_df(data_projection_function: F) -> F:
            @wraps(data_projection_function)
            def wrapper(self: Self, data_matrix: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
                """Wraps the projected data in a DataFrame."""
                projected_data = data_projection_function(self, data_matrix, *args, **kwargs)

                return pd.DataFrame(
                    projected_data,
                    columns=[
                        f"{self.projection_dimension} {i + 1}"  # type: ignore
                        for i in range(projected_data.shape[1])
                    ],
                    index=data_matrix.index,
                )

            return cast(F, wrapper)

        @staticmethod
        def _fit_standardizer(fit_model_function: F) -> F:
            """Fit a standardizer to the data."""

            @wraps(fit_model_function)
            def wrapper(self: Self, df: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
                if self.standardize:  # type: ignore
                    df = pd.DataFrame(
                        self.standard_scaler.fit_transform(df),  # type: ignore
                        index=df.index,
                        columns=df.columns,
                    )
                return fit_model_function(self, df, *args, **kwargs)

            return cast(F, wrapper)

        @staticmethod
        def _apply_standardizer(data_projection_function: F) -> F:
            """Apply a standardizer to the data."""

            @wraps(data_projection_function)
            def wrapper(self: Self, df: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
                if self.standardize:  # type: ignore
                    df = pd.DataFrame(
                        self.standard_scaler.transform(df),  # type: ignore
                        index=df.index,
                        columns=df.columns,
                    )
                return data_projection_function(self, df, *args, **kwargs)

            return cast(F, wrapper)

    @abstractmethod
    def _fit(self, data_matrix: pd.DataFrame) -> Self:
        pass

    @Decorators._fit_standardizer
    def fit(self, data_matrix: pd.DataFrame) -> Self:
        return self._fit(data_matrix)

    @abstractmethod
    def _project_data(self, data_matrix: pd.DataFrame) -> pd.DataFrame:
        pass

    @Decorators._apply_standardizer
    @Decorators._projected_data_to_df
    def project_data(self, data_matrix: pd.DataFrame) -> pd.DataFrame:
        return self._project_data(data_matrix)

    @Decorators._projected_data_to_df
    def fit_and_project_data(self, data_matrix: pd.DataFrame) -> pd.DataFrame:
        """Fit a model and returns the projected data."""
        self.fit(data_matrix)

        return self.project_data(data_matrix)

    @classmethod
    @abstractmethod
    def initialize(cls, standardize: bool, n_components: int | None, **kwargs: Any) -> Self:
        standard_scaler = StandardScaler() if standardize else None
        return cls(
            n_components=n_components,
            standardize=standardize,
            standard_scaler=standard_scaler,
            **kwargs,
        )


class PCALatentSpace(LatentSpace):
    latent_model_type: ModelType = ModelType.PCA
    model: PCA
    projection_dimension: str = "PC"
    _loadings: NDArray[np.float_]
    _pc_var_correlations_: NDArray[np.float_]

    @property
    def loadings(self) -> NDArray[np.float_]:
        return self._loadings

    @loadings.setter
    def loadings(self, value: NDArray[np.float_]) -> None:
        self._loadings = value

    @property
    def pc_var_correlations(self) -> NDArray[np.float_]:
        return self._pc_var_correlations_

    @pc_var_correlations.setter
    def pc_var_correlations(self, value: NDArray[np.float_]) -> None:
        self._pc_var_correlations_ = value

    def _fit(self, data_matrix: pd.DataFrame) -> Self:
        """Fit the model to the data_matrix."""
        # Fit the scaler and the model
        self.model.fit(data_matrix)

        # Calculate the loadings
        self.loadings = self.model.components_.T * np.sqrt(self.model.explained_variance_)

        # Calculates correlation matrix between PC and original variables
        if self.standardize:
            # Correlation between variable and the PC is given by the corresponding loading
            self.pc_var_correlations = self.loadings
        else:
            # Divide by original variables standard deviation
            self.pc_var_correlations = self.loadings / data_matrix.std().values.reshape(-1, 1)

        return self

    def _project_data(
        self,
        data_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Projects in PC space."""
        # Get the scores
        return self.model.transform(data_matrix)

    def get_sorted_loadings(self, pc: int = 1) -> pd.Series:
        """Returns the loadings of the variables onto the selected PC
        sorted by decreasing magnitude.
        """

        sorted_loading = np.sort(np.abs(self.loadings[:, pc - 1]))[::-1]
        var_idx = self.model.feature_names_in_[np.argsort(np.abs(self.loadings[:, pc - 1]))[::-1]]

        return pd.Series(sorted_loading, index=pd.Index(var_idx, name="original_variable"))

    def reconstruct_variables(self, data: pd.DataFrame, var_explained: float) -> pd.DataFrame:
        """Reconstructs original variable values from principal components.
        We sum squared correlation of the variables until reaching the desired amount
        of explained variance.
        Note that initial preprocessing is not reverted
        (i.e., data stays centered on 0 and SD=1 if standardization was applied).

        """

        # Get the scores
        pca_scores = self.project_data(data)

        # Loop through variables to reconstruct
        reconstructed_variables = {}
        for variable in self.model.feature_names_in_:
            var_idx = np.where(self.model.feature_names_in_ == variable)[0][0]
            sorted_pc_index = np.argsort(self.pc_var_correlations[var_idx, :])[::-1]
            var_cumsum = 0
            pc_idx = []
            for idx in sorted_pc_index:
                var_cumsum += self.pc_var_correlations[var_idx, idx] ** 2
                pc_idx.append(idx)
                if var_cumsum > var_explained:
                    break
            # Select the PCs that explain x percent of variance
            selected_pcs = self.model.components_[pc_idx, var_idx]
            reconstructed_variables[variable] = np.dot(pca_scores.iloc[:, pc_idx], selected_pcs)

        reconstructed_df = pd.DataFrame(
            reconstructed_variables, index=data.index, columns=data.columns
        )
        # Decentering
        if not self.standardize:
            reconstructed_df = reconstructed_df + self.model.mean_

        return reconstructed_df

    @classmethod
    def initialize(
        cls,
        standardize: bool = False,
        n_components: int | None = None,
        **kwargs: Any,
    ) -> Self:
        model = PCA(n_components=n_components, **kwargs)
        return super().initialize(
            n_components=n_components,
            standardize=standardize,
            model=model,
        )


class TSNELatentSpace(LatentSpace):
    latent_model_type: ModelType = ModelType.TSNE
    model: TSNE
    projection_dimension: str = "Dimension"
    n_components: int = 2

    def _fit(self, data_matrix: pd.DataFrame) -> Self:
        """Fit the model to the data_matrix."""
        # Fit the scaler and the model
        self.model.fit(data_matrix.values)  # openTSNE implementation requires a ndarray

        return self

    def _project_data(self, data_matrix: pd.DataFrame) -> pd.DataFrame:
        """Projects data in t-SNE embeddings."""
        return self.model.transform(data_matrix.values)

    @classmethod
    def initialize(
        cls,
        standardize: bool = False,
        n_components: int | None = 2,
        **kwargs: Any,
    ) -> Self:
        model = TSNE(n_components=n_components, n_jobs=-1, **kwargs)
        return super().initialize(
            n_components=n_components,
            standardize=standardize,
            model=model,
        )


class SignalTuner(BaseModel):
    """Uses PCA to tune the data to retain variance > `min_var_retained`."""

    pca_model: PCALatentSpace
    min_var_retained: float
    var_retained_: float | None = None
    n_pcomponents_: int | None = None

    def tune(self, data: pd.DataFrame, orig_space: bool = False) -> pd.DataFrame:
        try:
            check_is_fitted(self.pca_model.model)
        except NotFittedError:
            self.pca_model.fit(data)

        cumulative_variance = self.pca_model.model.explained_variance_ratio_.cumsum()
        try:
            pc_index = np.where(cumulative_variance >= self.min_var_retained)[0][0]
        except IndexError:
            logger.warning(
                f"{self.min_var_retained * 100}% variance not reached with available components,"
                f" consider setting PCA n_components > {self.pca_model.model.n_components}"
            )
            # We take all components
            pc_index = len(cumulative_variance) - 1

        self.var_retained_ = cumulative_variance[pc_index]
        self.n_pcomponents_ = pc_index + 1
        data_projection = self.pca_model.project_data(data).iloc[:, : self.n_pcomponents_]

        if orig_space:
            return np.dot(
                data_projection,
                self.pca_model.model.components_[: self.n_pcomponents_, :],
            )
        return data_projection

    @classmethod
    def initialize(
        cls,
        pca_model: PCALatentSpace | None = None,
        min_var_retained: float | None = None,
    ) -> Self:
        return cls(
            pca_model=pca_model if pca_model is not None else PCALatentSpace.initialize(),
            min_var_retained=(
                min_var_retained if min_var_retained is not None else STANDARD_MIN_VAR_RETAINED
            ),
        )
