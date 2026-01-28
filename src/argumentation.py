from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import (
    AbstractArgumentationFramework,
)
from py_arg.abstract_argumentation_classes.argument import Argument
from py_arg.abstract_argumentation_classes.defeat import Defeat
from py_arg.algorithms.semantics.get_grounded_extension import (
    get_grounded_extension as pyarg_grounded,
)

if TYPE_CHECKING:
    from case_base import CaseBase


@dataclass(frozen=True)
class AFCBAArgument:
    name: str
    content: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "pyarg_argument", Argument(self.name))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, AFCBAArgument) and self.name == other.name

    def __repr__(self):
        return f"AFCBAArgument('{self.name}')"

    @classmethod
    def citation(
        cls,
        case: pd.Series,
        focus_case: pd.Series,
        case_name: str,
        introduced_by: str = None,
        responding_to: str = None,
    ):
        if responding_to:
            name = f"Counterexample({case_name}->{responding_to})"
        else:
            name = f"Citation({case_name})"
        content = json.dumps(
            {
                "type": "Citation",
                "case_name": str(case_name),
                "case_data": case.to_dict(),
                "focus_data": focus_case.to_dict(),
                "introduced_by": introduced_by,
                "responding_to": responding_to,
            }
        )
        return cls(name, content)

    @classmethod
    def worse(
        cls,
        precedent: pd.Series,
        focus_case: pd.Series,
        worse_dims: set[str],
        precedent_name: str,
    ):
        dims_str = ",".join(sorted(worse_dims))
        name = f"Worse({precedent_name},[{dims_str}])"
        content = json.dumps(
            {
                "type": "Worse",
                "precedent_name": precedent_name,
                "precedent_data": precedent.to_dict(),
                "focus_data": focus_case.to_dict(),
                "worse_dimensions": list(worse_dims),
            }
        )
        return cls(name, content)

    @classmethod
    def compensates(
        cls,
        precedent: pd.Series,
        focus_case: pd.Series,
        worse_dims: set[str],
        better_dims: set[str],
        precedent_name: str,
    ):
        worse_str = ",".join(sorted(worse_dims))
        better_str = ",".join(sorted(better_dims)) if better_dims else "∅"
        name = f"Compensates({precedent_name},[{better_str}]for[{worse_str}])"
        content = json.dumps(
            {
                "type": "Compensates",
                "precedent_name": precedent_name,
                "precedent_data": precedent.to_dict(),
                "focus_data": focus_case.to_dict(),
                "worse_dimensions": list(worse_dims),
                "better_dimensions": list(better_dims),
            }
        )
        return cls(name, content)

    @classmethod
    def transformed(
        cls, original_name: str, transformation_series: Any, focus: pd.Series
    ):
        name = f"Transformed({original_name})"
        content = json.dumps(
            {
                "type": "Transformed",
                "original_precedent_name": original_name,
                "transformed_dimensions": str(transformation_series),
                "focus_data": focus.to_dict(),
            }
        )
        return cls(name, content)

    def get_content_data(self) -> dict:
        if not self.content:
            return {}
        try:
            return json.loads(self.content)
        except (json.JSONDecodeError, KeyError):
            return {}


@dataclass(frozen=True)
class AFCBADefeat:
    from_arg: AFCBAArgument
    to_arg: AFCBAArgument

    def __post_init__(self):
        object.__setattr__(
            self,
            "pyarg_defeat",
            Defeat(self.from_arg.pyarg_argument, self.to_arg.pyarg_argument),
        )

    def __repr__(self):
        return f"AFCBADefeat({self.from_arg.name} -> {self.to_arg.name})"


class DifferenceCache:
    def __init__(self, case_base: CaseBase, focus_case: pd.Series):
        self.case_base = case_base
        self.focus_case = focus_case
        self.target_column = case_base.target_column
        self._cache: dict[int, frozenset[str]] = {}
        self._non_dominated_by_outcome: dict[Any, list[int]] = {}
        self._precompute_all()

    def _precompute_all(self):
        for idx, case in self.case_base.cases.iterrows():
            diffs = self.case_base.calculate_differences(case, self.focus_case)
            self._cache[idx] = frozenset(diffs)
        indices_by_outcome: dict[Any, list[int]] = {}
        for idx, case in self.case_base.cases.iterrows():
            outcome = case[self.target_column]
            if outcome not in indices_by_outcome:
                indices_by_outcome[outcome] = []
            indices_by_outcome[outcome].append(idx)
        for outcome, indices in indices_by_outcome.items():
            self._non_dominated_by_outcome[outcome] = self._compute_non_dominated(
                indices
            )

    def _compute_non_dominated(self, indices: list[int]) -> list[int]:
        if not indices:
            return []
        dominated = set()
        for i, idx1 in enumerate(indices):
            if idx1 in dominated:
                continue
            diffs1 = self._cache[idx1]
            for idx2 in indices[i + 1 :]:
                if idx2 in dominated:
                    continue
                diffs2 = self._cache[idx2]
                if diffs1 < diffs2:
                    dominated.add(idx2)
                elif diffs2 < diffs1:
                    dominated.add(idx1)
                    break
        return [idx for idx in indices if idx not in dominated]

    def get(self, case_idx: int) -> frozenset[str]:
        return self._cache[case_idx]

    def compute(self, case: pd.Series) -> frozenset[str]:
        diffs = self.case_base.calculate_differences(case, self.focus_case)
        return frozenset(diffs)

    def get_non_dominated_indices(self, outcome: Any) -> list[int]:
        return self._non_dominated_by_outcome.get(outcome, [])


class CounterexampleGenerator:
    def __init__(
        self,
        target_case: pd.Series,
        target_case_name: str,
        focus_case: pd.Series,
        case_base: CaseBase,
        introduced_by: str,
        difference_cache: DifferenceCache | None = None,
        cited_case_indices: set[int] | None = None,
    ):
        self.target_case = target_case
        self.target_case_name = target_case_name
        self.focus_case = focus_case
        self.case_base = case_base
        self.introduced_by = introduced_by
        self.tried_indices: set[int] = set()
        self.target_outcome = target_case[case_base.target_column]
        self.difference_cache = difference_cache
        self.cited_case_indices = (
            cited_case_indices if cited_case_indices is not None else set()
        )
        if difference_cache:
            self.target_diffs = difference_cache.compute(target_case)
        else:
            self.target_diffs = frozenset(
                case_base.calculate_differences(target_case, focus_case)
            )
        self._candidate_indices = self._build_candidate_indices()

    def _build_candidate_indices(self) -> list[int]:
        candidates = []
        if self.difference_cache is not None:
            for outcome in self.difference_cache._non_dominated_by_outcome:
                if outcome != self.target_outcome:
                    non_dominated = self.difference_cache.get_non_dominated_indices(
                        outcome
                    )
                    candidates.extend(non_dominated)
        else:
            for idx, case in self.case_base.cases.iterrows():
                outcome = case[self.case_base.target_column]
                if outcome != self.target_outcome:
                    candidates.append(idx)
        return candidates

    def get_next_counterexample(self) -> AFCBAArgument | None:
        for case_idx in self._candidate_indices:
            if case_idx in self.tried_indices:
                continue
            if self._is_valid_attack(case_idx):
                self.tried_indices.add(case_idx)
                attacking_case = self.case_base.cases.loc[case_idx]
                return AFCBAArgument.citation(
                    attacking_case,
                    self.focus_case,
                    case_idx,
                    introduced_by=self.introduced_by,
                    responding_to=self.target_case_name,
                )
        return None

    def get_all_valid_counterexamples(
        self,
    ) -> list[tuple[AFCBAArgument, frozenset[str]]]:
        results = []
        for case_idx in self._candidate_indices:
            if case_idx in self.tried_indices:
                continue
            if case_idx in self.cited_case_indices:
                continue
            if self.difference_cache:
                attacking_diffs = self.difference_cache.get(case_idx)
            else:
                attacking_case = self.case_base.cases.loc[case_idx]
                attacking_diffs = frozenset(
                    self.case_base.calculate_differences(
                        attacking_case, self.focus_case
                    )
                )
            if self.introduced_by == "PRO":
                is_valid = attacking_diffs < self.target_diffs
            else:
                is_valid = not (self.target_diffs < attacking_diffs)
            if is_valid:
                self.tried_indices.add(case_idx)
                attacking_case = self.case_base.cases.loc[case_idx]
                arg = AFCBAArgument.citation(
                    attacking_case,
                    self.focus_case,
                    case_idx,
                    introduced_by=self.introduced_by,
                    responding_to=self.target_case_name,
                )
                results.append((arg, attacking_diffs))
        return results

    def _is_valid_attack(self, attacking_case_idx: int) -> bool:
        if attacking_case_idx in self.cited_case_indices:
            return False
        if self.difference_cache:
            attacking_diffs = self.difference_cache.get(attacking_case_idx)
        else:
            attacking_case = self.case_base.cases.loc[attacking_case_idx]
            attacking_diffs = frozenset(
                self.case_base.calculate_differences(attacking_case, self.focus_case)
            )
        if self.introduced_by == "PRO":
            return attacking_diffs < self.target_diffs
        else:
            return not (self.target_diffs < attacking_diffs)


def _filter_dominated_counterexamples(
    candidates: list[tuple[AFCBAArgument, frozenset[str]]],
) -> list[AFCBAArgument]:
    if not candidates:
        return []
    n = len(candidates)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        diffs_i = candidates[i][1]
        for j in range(i + 1, n):
            if dominated[j]:
                continue
            diffs_j = candidates[j][1]
            if diffs_i < diffs_j:
                dominated[j] = True
            elif diffs_j < diffs_i:
                dominated[i] = True
                break
    non_dominated = [candidates[i] for i in range(n) if not dominated[i]]
    non_dominated.sort(key=lambda x: len(x[1]))
    return [arg for arg, _ in non_dominated]


class AFCBAFramework:
    def __init__(
        self,
        case_base: CaseBase,
        focus_case: pd.Series,
        difference_cache: DifferenceCache | None = None,
    ):
        self.case_base = case_base
        self.focus_case = focus_case
        self.arguments: dict[str, AFCBAArgument] = {}
        self.defeats: list[AFCBADefeat] = []
        self.expanded_arguments: set[str] = set()
        self.counterexample_generators: dict[str, CounterexampleGenerator] = {}
        self._cited_case_indices: set[int] = set()
        if difference_cache is not None:
            self.difference_cache = difference_cache
        else:
            self.difference_cache = DifferenceCache(case_base, focus_case)
        self._distinguishing_cache: dict[str, tuple[list[str], list[str]]] = {}
        self._pyarg_framework: AbstractArgumentationFramework | None = None
        self._grounded_extension: set[AFCBAArgument] | None = None
        self._transformed_citations: set[str] = set()

    def add_argument(self, argument: AFCBAArgument):
        self.arguments[argument.name] = argument
        self._invalidate_cache()

    def add_defeat(self, from_arg_name: str, to_arg_name: str):
        from_arg = self.arguments[from_arg_name]
        to_arg = self.arguments[to_arg_name]
        defeat = AFCBADefeat(from_arg, to_arg)
        if defeat not in self.defeats:
            self.defeats.append(defeat)
            self._invalidate_cache()

    def has_transformation_for_citation(self, citation_name: str) -> bool:
        return citation_name in self._transformed_citations

    def get_attackers(self, argument: AFCBAArgument) -> list[AFCBAArgument]:
        if argument.name not in self.expanded_arguments:
            self._generate_attackers(argument)
            self.expanded_arguments.add(argument.name)
        attackers = []
        for defeat in self.defeats:
            if defeat.to_arg == argument:
                attackers.append(defeat.from_arg)
        return attackers

    def _generate_attackers(self, argument: AFCBAArgument):
        content_data = argument.get_content_data()
        arg_type = content_data.get("type", "")
        if arg_type == "Citation":
            self._generate_citation_attackers(argument, content_data)
        elif arg_type == "Worse":
            self._generate_worse_attackers(argument, content_data)

    def _generate_citation_attackers(self, argument: AFCBAArgument, content_data: dict):
        case = pd.Series(content_data["case_data"])
        focus = pd.Series(content_data["focus_data"])
        case_name = content_data["case_name"]
        introduced_by = content_data.get("introduced_by")
        case_outcome = case[self.case_base.target_column]
        focus_outcome = focus[self.case_base.target_column]
        if introduced_by == "PRO":
            try:
                case_idx = int(case_name)
                self._cited_case_indices.add(case_idx)
            except (ValueError, TypeError):
                pass
        if case_outcome == focus_outcome:
            worse_dims, _ = self._get_distinguishing_dimensions(case, case_name)
            if worse_dims:
                worse_arg = AFCBAArgument.worse(case, focus, set(worse_dims), case_name)
                self.add_argument(worse_arg)
                self.add_defeat(worse_arg.name, argument.name)
        if introduced_by == "CON":
            initial_citation_name = self._find_initial_citation(argument)
            if initial_citation_name is not None:
                initial_case_idx = int(initial_citation_name)
                initial_case = self.case_base.cases.loc[initial_case_idx]
                can_transform, transformed_series = self.case_base.can_transform(
                    initial_case, focus
                )
                if can_transform:
                    # Check if counterexample could also transform (for its outcome)
                    # focus_opposite = focus.copy()
                    # focus_opposite[self.case_base.target_column] = case[
                    #     self.case_base.target_column
                    # ]
                    # counterexample_can_transform, _ = self.case_base.can_transform(
                    #     case, focus_opposite
                    # )
                    # if not counterexample_can_transform:
                    transform_arg = AFCBAArgument.transformed(
                        initial_citation_name, transformed_series, focus
                    )
                    if transform_arg.name not in self.arguments:
                        self.add_argument(transform_arg)
                    self.add_defeat(transform_arg.name, argument.name)
        if argument.name not in self.counterexample_generators:
            will_be_introduced_by = "PRO" if introduced_by == "CON" else "CON"
            self.counterexample_generators[argument.name] = CounterexampleGenerator(
                case,
                case_name,
                focus,
                self.case_base,
                will_be_introduced_by,
                difference_cache=self.difference_cache,
                cited_case_indices=self._cited_case_indices,
            )
        generator = self.counterexample_generators[argument.name]
        candidates = generator.get_all_valid_counterexamples()
        # non_dominated = _filter_dominated_counterexamples(candidates)
        for counterexample, _ in candidates:
            self.add_argument(counterexample)
            self.add_defeat(counterexample.name, argument.name)

    def _find_initial_citation(self, counterexample: AFCBAArgument) -> str | None:
        ce_data = counterexample.get_content_data()
        current_case_name = ce_data.get("responding_to")
        if current_case_name is None:
            return None
        case_name_to_data: dict[str, dict] = {}
        for arg in self.arguments.values():
            arg_data = arg.get_content_data()
            if arg_data.get("type") == "Citation":
                case_name = arg_data.get("case_name")
                if case_name is not None:
                    case_name_to_data[case_name] = arg_data
        visited = set()
        while current_case_name in case_name_to_data:
            if current_case_name in visited:
                break
            visited.add(current_case_name)
            current_data = case_name_to_data[current_case_name]
            parent_name = current_data.get("responding_to")
            if parent_name is None:
                return current_case_name
            current_case_name = parent_name
        return current_case_name

    def _generate_worse_attackers(self, argument: AFCBAArgument, content_data: dict):
        precedent = pd.Series(content_data["precedent_data"])
        focus = pd.Series(content_data["focus_data"])
        worse_dims = set(content_data["worse_dimensions"])
        precedent_name = content_data["precedent_name"]
        _, better_dims = self._get_distinguishing_dimensions(precedent, precedent_name)
        compensation = self.case_base.find_valid_compensation(
            better_dims, worse_dims, focus
        )
        if compensation is not None:
            comp_arg = AFCBAArgument.compensates(
                precedent, focus, worse_dims, compensation, precedent_name
            )
            self.add_argument(comp_arg)
            self.add_defeat(comp_arg.name, argument.name)

    def _get_distinguishing_dimensions(
        self, precedent: pd.Series, precedent_name: str
    ) -> tuple[list[str], list[str]]:
        if precedent_name in self._distinguishing_cache:
            return self._distinguishing_cache[precedent_name]
        worse_dims, better_dims = self.case_base.get_distinguishing_dimensions(
            precedent, self.focus_case
        )
        self._distinguishing_cache[precedent_name] = (worse_dims, better_dims)
        return worse_dims, better_dims

    def _invalidate_cache(self):
        self._pyarg_framework = None
        self._grounded_extension = None

    def _build_pyarg_framework(self) -> AbstractArgumentationFramework:
        if self._pyarg_framework is not None:
            return self._pyarg_framework
        pyarg_args = frozenset(arg.pyarg_argument for arg in self.arguments.values())
        pyarg_defeats = frozenset(defeat.pyarg_defeat for defeat in self.defeats)
        self._pyarg_framework = AbstractArgumentationFramework(
            arguments=pyarg_args, defeats=pyarg_defeats
        )
        return self._pyarg_framework

    def get_grounded_extension(self) -> set[AFCBAArgument]:
        if self._grounded_extension is not None:
            return self._grounded_extension
        pyarg_framework = self._build_pyarg_framework()
        pyarg_grounded_args = pyarg_grounded(pyarg_framework)
        name_to_arg = {arg.name: arg for arg in self.arguments.values()}
        self._grounded_extension = set()
        for pyarg_arg in pyarg_grounded_args:
            if pyarg_arg.name in name_to_arg:
                self._grounded_extension.add(name_to_arg[pyarg_arg.name])
        return self._grounded_extension
