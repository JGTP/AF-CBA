from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class HeuristicType(Enum):
    MAJORITY = "majority"
    NEAREST_NEIGHBOUR = "nearest_neighbour"


class Heuristic(ABC):
    @abstractmethod
    def predict(self, focus_case: pd.Series) -> Any:
        pass

    @abstractmethod
    def fit(self, cases: pd.DataFrame, target_column: str) -> "Heuristic":
        pass


class MajorityHeuristic(Heuristic):
    def __init__(self):
        self.majority_outcome: Any = None
        self._is_fitted: bool = False

    def fit(self, cases: pd.DataFrame, target_column: str) -> "MajorityHeuristic":
        outcome_counts = cases[target_column].value_counts()
        self.majority_outcome = outcome_counts.idxmax()
        self._is_fitted = True
        return self

    def predict(self, focus_case: pd.Series) -> Any:
        if not self._is_fitted:
            raise RuntimeError("Heuristic must be fitted before predicting.")
        return self.majority_outcome


class NearestNeighbourHeuristic(Heuristic):
    def __init__(
        self,
        k: int = 3,
        metric: str = "euclidean",
        weights: str = "uniform",
    ):
        self.k = k
        self.metric = metric
        self.weights = weights
        self._nn: NearestNeighbors | None = None
        self._cases: pd.DataFrame | None = None
        self._target_column: str | None = None
        self._feature_columns: list[str] | None = None
        self._is_fitted: bool = False

    def fit(
        self, cases: pd.DataFrame, target_column: str
    ) -> "NearestNeighbourHeuristic":
        self._cases = cases.copy()
        self._target_column = target_column
        self._feature_columns = [col for col in cases.columns if col != target_column]
        X = cases[self._feature_columns].values
        effective_k = min(self.k, len(cases))
        self._nn = NearestNeighbors(
            n_neighbors=effective_k,
            metric=self.metric,
            algorithm="auto",
        )
        self._nn.fit(X)
        self._is_fitted = True
        return self

    def predict(self, focus_case: pd.Series) -> Any:
        if not self._is_fitted:
            raise RuntimeError("Heuristic must be fitted before predicting.")
        focus_features = focus_case[self._feature_columns].values.reshape(1, -1)
        distances, indices = self._nn.kneighbors(focus_features)
        distances = distances[0]
        indices = indices[0]
        neighbour_outcomes = self._cases.iloc[indices][self._target_column].values
        if self.weights == "distance":
            eps = 1e-10
            weights = 1.0 / (distances + eps)
            outcome_weights: dict[Any, float] = {}
            for outcome, weight in zip(neighbour_outcomes, weights, strict=False):
                outcome_weights[outcome] = outcome_weights.get(outcome, 0.0) + weight
            return max(outcome_weights, key=outcome_weights.get)
        else:
            unique, counts = np.unique(neighbour_outcomes, return_counts=True)
            return unique[np.argmax(counts)]


def create_heuristic(
    heuristic_type: HeuristicType | str,
    **kwargs,
) -> Heuristic:
    if isinstance(heuristic_type, str):
        heuristic_type = HeuristicType(heuristic_type.lower())
    if heuristic_type == HeuristicType.MAJORITY:
        return MajorityHeuristic()
    elif heuristic_type == HeuristicType.NEAREST_NEIGHBOUR:
        return NearestNeighbourHeuristic(**kwargs)
    else:
        raise ValueError(f"Unknown heuristic type: {heuristic_type}")
