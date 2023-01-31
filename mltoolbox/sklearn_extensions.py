from typing import List, Sequence, Tuple

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class PrefitVotingClassifier(object):
    """Stripped-down version of VotingClassifier that uses prefit estimators
    see https://gist.github.com/tomquisel/a421235422fdf6b51ec2ccc5e3dee1b4
    """

    def __init__(
        self,
        estimators: List[Tuple[str, sklearn.base.BaseEstimator]],
        voting: str = "hard",
        weights: Sequence | None = None,
    ):
        self.estimators = [e[1] for e in estimators]
        self.named_estimators = dict(estimators)
        self.voting = voting
        self.weights = weights

    def fit(
        self, X: Sequence, y: Sequence, sample_weight: Sequence | None = None
    ):
        raise NotImplementedError

    def predict(self, X: Sequence) -> np.ndarray:
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        check_is_fitted(self, "estimators")
        if self.voting == "soft":
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            # We need to encode the labels
            self.le_ = LabelEncoder().fit(predictions.ravel())

            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=self.le_.transform(predictions.ravel()).reshape(
                    predictions.shape
                ),
            )

            maj = self.le_.inverse_transform(maj)

        return maj

    def _collect_probas(self, X: Sequence) -> np.ndarray:
        """Collect results from estimator predict calls."""
        return np.asarray([clf.predict_proba(X) for clf in self.estimators])

    def predict_proba(self, X: Sequence) -> float:
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """

        if self.voting == "hard":
            raise AttributeError(
                f"predict_proba is not available when voting={self.voting}"
            )
        check_is_fitted(self, "estimators")
        return np.average(
            self._collect_probas(X), axis=0, weights=self.weights
        )

    def transform(self, X: Sequence) -> np.ndarray:
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          ndarray = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          ndarray = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """
        check_is_fitted(self, "estimators")
        if self.voting == "soft":
            return self._collect_probas(X)

        return self._predict(X)

    def _predict(self, X: Sequence) -> np.ndarray:
        """Collect results from estimator predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators]).T
