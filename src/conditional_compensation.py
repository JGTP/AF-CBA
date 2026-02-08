import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap


def parse_ripper_value(value: Any) -> tuple[str, Any, Any]:
    if value is None:
        return "==", None, None
    str_val = str(value).strip()
    range_match = re.match(r"^(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)$", str_val)
    if range_match:
        lower = float(range_match.group(1))
        upper = float(range_match.group(2))
        return "range", lower, upper
    inequality_match = re.match(r"^([<>]=?)\s*(-?\d+\.?\d*)$", str_val)
    if inequality_match:
        op = inequality_match.group(1)
        val = float(inequality_match.group(2))
        return op, val, None
    try:
        numeric_val = float(str_val)
        return "==", numeric_val, None
    except (ValueError, TypeError):
        return "==", value, None


@dataclass
class RuleCondition:
    feature: str
    raw_operator: str
    raw_value: Any
    parsed_operator: str
    lower_bound: Any
    upper_bound: Any

    @classmethod
    def from_raw(cls, feature: str, operator: str, value: Any) -> "RuleCondition":
        parsed_op, lower, upper = parse_ripper_value(value)
        if operator == "==" and parsed_op != "==":
            effective_op = parsed_op
        else:
            effective_op = operator
        return cls(
            feature=feature,
            raw_operator=operator,
            raw_value=value,
            parsed_operator=effective_op,
            lower_bound=lower,
            upper_bound=upper,
        )

    def evaluate(self, case_value: Any) -> bool:
        if self.lower_bound is None:
            return case_value == self.raw_value
        if isinstance(self.lower_bound, str) and self.parsed_operator == "==":
            return str(case_value) == self.lower_bound
        try:
            numeric_case = float(case_value)
        except (ValueError, TypeError):
            logging.debug(
                f"Cannot compare non-numeric case value {case_value} "
                f"with numeric condition {self.parsed_operator} {self.lower_bound}"
            )
            return False
        if self.parsed_operator == "range":
            return self.lower_bound <= numeric_case <= self.upper_bound
        elif self.parsed_operator == "==":
            return numeric_case == self.lower_bound
        elif self.parsed_operator == "!=":
            return numeric_case != self.lower_bound
        elif self.parsed_operator == "<=":
            return numeric_case <= self.lower_bound
        elif self.parsed_operator == ">=":
            return numeric_case >= self.lower_bound
        elif self.parsed_operator == "<":
            return numeric_case < self.lower_bound
        elif self.parsed_operator == ">":
            return numeric_case > self.lower_bound
        else:
            logging.warning(
                f"Unknown operator {self.parsed_operator} in rule condition"
            )
            return False

    def create_mask(self, df: pd.DataFrame) -> pd.Series:
        if self.feature not in df.columns:
            logging.warning(f"Feature {self.feature} not found in data")
            return pd.Series(False, index=df.index)
        col = df[self.feature]
        if self.lower_bound is None:
            return col == self.raw_value
        if isinstance(self.lower_bound, str) and self.parsed_operator == "==":
            return col.astype(str) == self.lower_bound
        numeric_col = pd.to_numeric(col, errors="coerce")
        if self.parsed_operator == "range":
            return (numeric_col >= self.lower_bound) & (numeric_col <= self.upper_bound)
        elif self.parsed_operator == "==":
            return numeric_col == self.lower_bound
        elif self.parsed_operator == "!=":
            return numeric_col != self.lower_bound
        elif self.parsed_operator == "<=":
            return numeric_col <= self.lower_bound
        elif self.parsed_operator == ">=":
            return numeric_col >= self.lower_bound
        elif self.parsed_operator == "<":
            return numeric_col < self.lower_bound
        elif self.parsed_operator == ">":
            return numeric_col > self.lower_bound
        else:
            logging.warning(f"Unknown operator {self.parsed_operator}")
            return pd.Series(False, index=df.index)


@dataclass
class RuleContext:
    rule_id: str
    conditions: list[RuleCondition]

    def matches(self, case: pd.Series) -> bool:
        for cond in self.conditions:
            if cond.feature not in case.index:
                return False
            if not cond.evaluate(case[cond.feature]):
                return False
        return True

    def create_mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for cond in self.conditions:
            mask &= cond.create_mask(df)
        return mask

    @classmethod
    def from_legacy_format(cls, rule_id: str, conditions: list[dict]) -> "RuleContext":
        [{"feature": "name", "operator": "==", "value": "<2.0"}, ...]
        parsed_conditions = [
            RuleCondition.from_raw(
                feature=cond["feature"],
                operator=cond["operator"],
                value=cond["value"],
            )
            for cond in conditions
        ]
        return cls(rule_id=rule_id, conditions=parsed_conditions)


class ConditionalCompensationChecker:
    def __init__(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        delta: float = 2.0,
        min_support: int = 10,
        n_splits: int = 5,
    ):
        self.model = model
        self.X = X
        self.y = y
        self.delta = delta
        self.min_support = min_support
        self.feature_names = X.columns.tolist()
        logging.info("Extracting stable RIPPER rules for conditional compensation")
        from ripper import cross_validate_RIPPER

        stable_rules = cross_validate_RIPPER(X, y, n_splits=n_splits)
        self.rules = [self._extract_context(r) for r in stable_rules]
        logging.info(f"Extracted {len(self.rules)} stable rule contexts")
        self.explainer = shap.TreeExplainer(model)
        self._shap_cache: dict[str, dict[str, float]] = {}
        self._corr_cache: dict[str, pd.DataFrame] = {}
        self._support_cache: dict[str, int] = {}

    def _extract_context(self, rule) -> RuleContext:
        from ripper import determine_operator
        from utils import convert_to_serialisable

        conditions = []
        for cond in rule.conds:
            feature_idx = cond.feature
            feature_name = self.feature_names[feature_idx]
            operator = determine_operator(str(cond))
            value = convert_to_serialisable(cond.val)
            parsed_cond = RuleCondition.from_raw(feature_name, operator, value)
            conditions.append(parsed_cond)
        return RuleContext(rule_id=str(id(rule)), conditions=conditions)

    def can_compensate(
        self, focus_case: pd.Series, better_dims: set[str], worse_dims: set[str]
    ) -> bool | None:
        applicable_rules = [r for r in self.rules if r.matches(focus_case)]
        if not worse_dims:
            return True
        # if not better_dims:
        #     return False
        if not applicable_rules:
            return None
        found_rule_with_support = False
        for rule in applicable_rules:
            if rule.rule_id not in self._shap_cache:
                self._compute_conditional_shap(rule)
            if self._support_cache.get(rule.rule_id, 0) < self.min_support:
                continue
            shap_values = self._shap_cache.get(rule.rule_id, {})
            if not shap_values:
                continue
            found_rule_with_support = True
            if self._check_thresholds(rule, shap_values, better_dims, worse_dims):
                return True
        if not found_rule_with_support:
            return None
        return False

    def _compute_conditional_shap(self, rule: RuleContext):
        mask = rule.create_mask(self.X)
        X_subset = self.X[mask]
        support = len(X_subset)
        self._support_cache[rule.rule_id] = support
        if support < self.min_support:
            logging.info(
                f"Rule {rule.rule_id[:8]}... has insufficient support "
                f"({support} < {self.min_support})"
            )
            self._shap_cache[rule.rule_id] = {}
            self._corr_cache[rule.rule_id] = pd.DataFrame()
            return
        logging.info(
            f"Computing conditional SHAP for rule {rule.rule_id[:8]}... (support={support})"
        )
        shap_values = self.explainer(X_subset)
        abs_shap = np.abs(shap_values.values)
        mean_shap = abs_shap.mean(axis=0)
        self._shap_cache[rule.rule_id] = {
            name: float(imp)
            for name, imp in zip(self.feature_names, mean_shap, strict=False)
        }
        self._corr_cache[rule.rule_id] = X_subset.corr()

    def _check_thresholds(
        self,
        rule: RuleContext,
        shap_values: dict[str, float],
        better_dims: set[str],
        worse_dims: set[str],
    ) -> bool:
        # if len(worse_dims) == 0:
        #     return True
        # if len(better_dims) == 0:
        #     return False
        better_imp = sum(shap_values.get(d, 0.0) for d in better_dims)
        worse_imp = sum(shap_values.get(d, 0.0) for d in worse_dims)
        if worse_imp == 0:
            return better_imp > 0
        relative_diff = (better_imp - worse_imp) / worse_imp
        if relative_diff <= self.delta:
            return False
        return True

    def get_statistics(self) -> dict:
        return {
            "total_rules": len(self.rules),
            "rules_computed": len(self._shap_cache),
            "rules_with_sufficient_support": sum(
                1
                for support in self._support_cache.values()
                if support >= self.min_support
            ),
            "average_support": (
                np.mean(list(self._support_cache.values()))
                if self._support_cache
                else 0
            ),
        }
