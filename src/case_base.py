from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


class DimensionDirection(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class DimensionInfo:
    name: str
    direction: DimensionDirection
    coefficient: float = 0.0

    def is_at_least_as_good_as(
        self, value1: Any, value2: Any, for_outcome: Any
    ) -> bool:
        if self.direction == DimensionDirection.NEUTRAL:
            raise RuntimeError("Encountered neutral dimension.")
        if self.direction == DimensionDirection.POSITIVE:
            return value1 >= value2 if for_outcome == 1 else value1 <= value2
        else:
            return value1 <= value2 if for_outcome == 1 else value1 >= value2

    def is_better_than(self, value1: Any, value2: Any, for_outcome: Any) -> bool:
        if self.direction == DimensionDirection.NEUTRAL:
            raise RuntimeError("Encountered neutral dimension.")
        if self.direction == DimensionDirection.POSITIVE:
            return value1 > value2 if for_outcome == 1 else value1 < value2
        else:
            return value1 < value2 if for_outcome == 1 else value1 > value2

    def is_worse_than(self, value1: Any, value2: Any, for_outcome: Any) -> bool:
        return self.is_better_than(value2, value1, for_outcome)


class CaseBase:
    def __init__(
        self,
        cases: pd.DataFrame,
        target_column: str,
        min_correlation: float = 0.0,
        delta: float = 2.0,
        random_state: int = 42,
        conditional_checker: Any = None,
        config: Any = None,
    ):
        self.cases = cases.copy()
        self.target_column = target_column
        self.feature_columns = [col for col in cases.columns if col != target_column]
        self.min_correlation = min_correlation
        self.delta = delta
        self.random_state = random_state
        self.conditional_checker = conditional_checker
        auth_config = self._get_config_section(config, "authoritativeness")
        if auth_config and auth_config.get("enabled", False):
            self.auth_method = auth_config.get("method", "default")
        else:
            self.auth_method = "default"
        prefs_config = self._get_config_section(config, "conditional_preferences")
        if prefs_config:
            self.preference_method = prefs_config.get("method", "global")
            if "delta" in prefs_config:
                self.delta = prefs_config["delta"]
        else:
            self.preference_method = "global"
        self.dimensions = self._infer_dimensions()
        self.shap_importance = self._compute_shap_importance()
        self._correlation_matrix = self._compute_correlation_matrix()
        if self.auth_method != "default":
            self.calculate_alphas()

    def _compute_correlation_matrix(self) -> pd.DataFrame:
        return self.cases[self.feature_columns].corr()

    def _get_config_section(self, config: Any, section: str) -> dict | None:
        if config is None:
            return None
        if isinstance(config, dict):
            return config.get(section)
        if hasattr(config, section):
            return getattr(config, section)
        return None

    def calculate_alphas(self):
        if not hasattr(self, "auth_method") or self.auth_method == "default":
            return
        from authoritativeness import alpha

        alpha_scores = []
        for idx, case in tqdm(
            self.cases.iterrows(),
            total=len(self.cases),
            desc=f"Calculating {self.auth_method} authoritativeness",
        ):
            score = alpha(case, self, self.auth_method)
            alpha_scores.append(score)
        self.cases["alpha"] = alpha_scores

    def _infer_dimensions(self) -> dict[str, DimensionInfo]:
        dimensions = {}
        for feature in self.feature_columns:
            X = self.cases[[feature]].values.reshape(-1, 1)
            y = self.cases[self.target_column].values
            lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
            lr.fit(X, y)
            coefficient = lr.coef_[0][0]
            if abs(coefficient) < self.min_correlation:
                raise RuntimeError("Encountered neutral dimension.")
            direction = (
                DimensionDirection.POSITIVE
                if coefficient > 0
                else DimensionDirection.NEGATIVE
            )
            dimensions[feature] = DimensionInfo(
                name=feature, direction=direction, coefficient=coefficient
            )
        return dimensions

    def _compute_shap_importance(self) -> dict[str, float]:
        X = self.cases[self.feature_columns].values
        y = self.cases[self.target_column].values
        classifier = HistGradientBoostingClassifier(
            random_state=self.random_state, max_iter=100
        )
        classifier.fit(X, y)
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X)
        abs_shap = np.abs(shap_values.values)
        mean_abs_shap = abs_shap.mean(axis=0)
        return {
            name: float(importance)
            for name, importance in zip(
                self.feature_columns, mean_abs_shap, strict=False
            )
        }

    def calculate_differences(self, c: pd.Series, f: pd.Series) -> list[str]:
        D = []
        s = c[self.target_column]
        for dimension in self.feature_columns:
            if dimension not in self.dimensions:
                continue
            dim_info = self.dimensions[dimension]
            precedent_val = c[dimension]
            focus_val = f[dimension]
            if s == f[self.target_column]:
                if not dim_info.is_at_least_as_good_as(focus_val, precedent_val, s):
                    D.append(dimension)
            else:
                if not dim_info.is_at_least_as_good_as(
                    precedent_val, focus_val, f[self.target_column]
                ):
                    D.append(dimension)
        return D

    def get_distinguishing_dimensions(
        self, precedent: pd.Series, focus_case: pd.Series
    ) -> tuple[list[str], list[str]]:
        worse_dims = []
        better_dims = []
        focus_outcome = focus_case[self.target_column]
        for feature in self.feature_columns:
            if feature not in self.dimensions:
                continue
            dim_info = self.dimensions[feature]
            precedent_val = precedent[feature]
            focus_val = focus_case[feature]
            if precedent_val == focus_val:
                continue
            if dim_info.is_worse_than(focus_val, precedent_val, focus_outcome):
                worse_dims.append(feature)
            elif dim_info.is_better_than(focus_val, precedent_val, focus_outcome):
                better_dims.append(feature)
        return worse_dims, better_dims

    def can_compensate(
        self, better_dims: set[str], worse_dims: set[str], focus_case: pd.Series = None
    ) -> bool:
        if len(worse_dims) == 0:
            return True
        # if len(better_dims) == 0:
        #     return False
        method = getattr(self, "preference_method", None)
        if method == "conditional":
            if self.conditional_checker and focus_case is not None:
                result = self.conditional_checker.can_compensate(
                    focus_case, better_dims, worse_dims
                )
                return result if result is not None else False
            return False
        elif method == "global":
            return self._can_compensate_global(better_dims, worse_dims)
        else:
            raise ValueError(
                f"Unknown preference discovery method: {method}. "
                "Expected 'global' or 'conditional'."
            )

    def _can_compensate_global(
        self, better_dims: set[str], worse_dims: set[str]
    ) -> bool:
        if len(worse_dims) == 0:
            return True
        # if len(better_dims) == 0:
        #     return False
        better_importance = sum(
            self.shap_importance.get(dim, 0.0) for dim in better_dims
        )
        worse_importance = sum(self.shap_importance.get(dim, 0.0) for dim in worse_dims)
        if worse_importance == 0:
            return better_importance > 0
        if (better_importance - worse_importance) / worse_importance <= self.delta:
            return False
        return True

    def find_valid_compensation(
        self, better_dims: list[str], worse_dims: set[str], focus_case: pd.Series = None
    ) -> set[str] | None:
        if self.can_compensate(set(better_dims), worse_dims, focus_case):
            return set(better_dims)
        return None

    def can_transform(
        self, initial_precedent: pd.Series, focus: pd.Series
    ) -> tuple[bool, pd.Series]:
        worse_dims, better_dims = self.get_distinguishing_dimensions(
            initial_precedent, focus
        )
        # if len(better_dims) == 0 and len(worse_dims) == 0:
        #     return True, None
        if self.can_compensate(set(better_dims), set(worse_dims), focus):
            transformed = initial_precedent.copy()
            all_compensating_dims = set(worse_dims) | set(better_dims)
            for dim in all_compensating_dims:
                if dim in focus.index:
                    transformed[dim] = focus[dim]
            diffs_after = self.calculate_differences(transformed, focus)
            if len(diffs_after) == 0:
                return True, transformed
        return False, None

    def precedent_forces_focus(
        self, precedent_case: pd.Series, focus_case: pd.Series
    ) -> bool:
        precedent_outcome = precedent_case[self.target_column]
        for feature in self.feature_columns:
            if feature not in self.dimensions:
                continue
            dim_info = self.dimensions[feature]
            precedent_val = precedent_case[feature]
            focus_val = focus_case[feature]
            if not dim_info.is_at_least_as_good_as(
                focus_val, precedent_val, precedent_outcome
            ):
                return False
        return True

    def find_best_precedents(
        self, focus_case: pd.Series, difference_cache=None
    ) -> pd.DataFrame:
        if hasattr(self, "auth_method") and self.auth_method != "default":
            return self.find_best_precedents_with_alpha(focus_case, difference_cache)
        focus_outcome = focus_case[self.target_column]
        exclude_index = focus_case.name if hasattr(focus_case, "name") else None
        if difference_cache is not None and hasattr(
            difference_cache, "get_non_dominated_indices"
        ):
            non_dominated_indices = difference_cache.get_non_dominated_indices(
                focus_outcome
            )
            if exclude_index is not None:
                non_dominated_indices = [
                    idx for idx in non_dominated_indices if idx != exclude_index
                ]
            if not non_dominated_indices:
                return pd.DataFrame()
            return self.cases.loc[non_dominated_indices]
        candidates = self.cases[self.cases[self.target_column] == focus_outcome].copy()
        if exclude_index is not None and exclude_index in candidates.index:
            candidates = candidates.drop(exclude_index)
        if len(candidates) == 0:
            return pd.DataFrame()
        comparisons = []
        for idx, candidate in candidates.iterrows():
            if difference_cache is not None:
                differences = set(difference_cache.get(idx))
            else:
                differences = self.calculate_differences(candidate, focus_case)
                differences = set(differences) if differences else set()
            comparisons.append({"idx": idx, "differences": differences})
        dominated = set()
        for c1 in comparisons:
            for c2 in comparisons:
                if c1["idx"] != c2["idx"] and c1["differences"] < c2["differences"]:
                    dominated.add(c2["idx"])
        best_indices = [c["idx"] for c in comparisons if c["idx"] not in dominated]
        return candidates.loc[best_indices]

    def find_best_precedents_with_alpha(
        self, focus_case: pd.Series, difference_cache=None
    ) -> pd.DataFrame:
        focus_outcome = focus_case[self.target_column]
        exclude_index = focus_case.name if hasattr(focus_case, "name") else None
        if difference_cache is not None and hasattr(
            difference_cache, "get_non_dominated_indices"
        ):
            non_dominated_indices = difference_cache.get_non_dominated_indices(
                focus_outcome
            )
            if exclude_index is not None:
                non_dominated_indices = [
                    idx for idx in non_dominated_indices if idx != exclude_index
                ]
            if not non_dominated_indices:
                return pd.DataFrame()
            comparisons = []
            for idx in non_dominated_indices:
                case = self.cases.loc[idx]
                alpha_score = case.get("alpha", 0.0) if "alpha" in case else 0.0
                comparisons.append(
                    {
                        "idx": idx,
                        "differences": set(difference_cache.get(idx)),
                        "alpha": alpha_score,
                    }
                )
        else:
            candidates = self.cases[
                self.cases[self.target_column] == focus_outcome
            ].copy()
            if exclude_index is not None and exclude_index in candidates.index:
                candidates = candidates.drop(exclude_index)
            if len(candidates) == 0:
                return pd.DataFrame()
            comparisons = []
            for idx, candidate in candidates.iterrows():
                if difference_cache is not None:
                    diff_set = set(difference_cache.get(idx))
                else:
                    differences = self.calculate_differences(candidate, focus_case)
                    diff_set = set(differences) if differences else set()
                alpha_score = (
                    candidate.get("alpha", 0.0) if "alpha" in candidate else 0.0
                )
                comparisons.append(
                    {"idx": idx, "differences": diff_set, "alpha": alpha_score}
                )
            non_dominated = []
            for c in comparisons:
                is_dominated = False
                for other in comparisons:
                    if (
                        c["idx"] != other["idx"]
                        and other["differences"] < c["differences"]
                    ):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append(c)
            comparisons = non_dominated
        if not comparisons:
            return pd.DataFrame()
        if self.auth_method != "default":
            final_precedents = []
            for c in comparisons:
                is_beaten = False
                for other in comparisons:
                    if c["idx"] != other["idx"]:
                        if (
                            other["differences"] <= c["differences"]
                            and other["alpha"] > c["alpha"]
                        ):
                            is_beaten = True
                            break
                if not is_beaten:
                    final_precedents.append(c)
            best_indices = [p["idx"] for p in final_precedents]
        else:
            best_indices = [c["idx"] for c in comparisons]
        return self.cases.loc[best_indices]

    def check_consistency(self) -> tuple[float, int, list[int]]:
        case_indices = list(range(len(self.cases)))
        inconsistent_pairs = []
        for i in tqdm(case_indices, desc="Checking consistency"):
            case_i = self.cases.iloc[i]
            for j in case_indices:
                if i >= j:
                    continue
                case_j = self.cases.iloc[j]
                if case_i[self.target_column] != case_j[self.target_column] and (
                    self.precedent_forces_focus(case_i, case_j)
                    or self.precedent_forces_focus(case_j, case_i)
                ):
                    inconsistent_pairs.append((i, j))
        inconsistency_count = dict.fromkeys(case_indices, 0)
        for i, j in inconsistent_pairs:
            inconsistency_count[i] += 1
            inconsistency_count[j] += 1
        cases_to_remove = []
        remaining_pairs = set(inconsistent_pairs)
        while remaining_pairs:
            worst_case = max(inconsistency_count, key=inconsistency_count.get)
            if inconsistency_count[worst_case] == 0:
                break
            cases_to_remove.append(worst_case)
            new_remaining = set()
            for i, j in remaining_pairs:
                if i != worst_case and j != worst_case:
                    new_remaining.add((i, j))
                else:
                    inconsistency_count[i] -= 1
                    inconsistency_count[j] -= 1
            remaining_pairs = new_remaining
            inconsistency_count[worst_case] = 0
        n_cases = len(self.cases)
        consistency_percentage = ((n_cases - len(cases_to_remove)) / n_cases) * 100
        return consistency_percentage, len(cases_to_remove), cases_to_remove

    def __len__(self) -> int:
        return len(self.cases)


def create_case_base_from_data(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    conditional_checker: Any = None,
    config: Any = None,
) -> CaseBase:
    target_column = y.name if y.name else "target"
    cases = X.copy()
    cases[target_column] = y.values
    return CaseBase(
        cases,
        target_column,
        random_state=random_state,
        conditional_checker=conditional_checker,
        config=config,
    )
