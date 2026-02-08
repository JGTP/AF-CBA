from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from case_base import CaseBase


def n_agreement(case: pd.Series, case_base: "CaseBase") -> int:
    same_outcome = case_base.cases[
        case_base.cases[case_base.target_column] == case[case_base.target_column]
    ]
    count = 0
    for idx, other_case in same_outcome.iterrows():
        if case_base.precedent_forces_focus(case, other_case):
            count += 1
    return count


def n_disagreement(case: pd.Series, case_base: "CaseBase") -> int:
    other_outcome = case_base.cases[
        case_base.cases[case_base.target_column] != case[case_base.target_column]
    ]
    count = 0
    for idx, other_case in other_outcome.iterrows():
        if case_base.precedent_forces_focus(case, other_case):
            count += 1
    return count


def relative_authoritativeness(case: pd.Series, case_base: "CaseBase") -> float:
    n_a = n_agreement(case, case_base)
    n_d = n_disagreement(case, case_base)
    if n_a + n_d == 0:
        return 0.0
    return n_a / (n_a + n_d)


def absolute_authoritativeness(case: pd.Series, case_base: "CaseBase") -> float:
    n_a = n_agreement(case, case_base)
    return n_a / len(case_base.cases)


def product_authoritativeness(case: pd.Series, case_base: "CaseBase") -> float:
    rels = relative_authoritativeness(case, case_base)
    abss = absolute_authoritativeness(case, case_base)
    return rels * abss


def harmonic_authoritativeness(
    case: pd.Series, case_base: "CaseBase", beta: float = 1.0
) -> float:
    rels = relative_authoritativeness(case, case_base)
    abss = absolute_authoritativeness(case, case_base)
    if rels == 0 and abss == 0:
        return 0.0
    numerator = (1 + beta**2) * (rels * abss)
    denominator = (beta**2 * rels) + abss
    if denominator == 0:
        return 0.0
    return numerator / denominator


def alpha(case: pd.Series, case_base: "CaseBase", method: str) -> float:
    if method == "relative":
        return relative_authoritativeness(case, case_base)
    elif method == "absolute":
        return absolute_authoritativeness(case, case_base)
    elif method == "product":
        return product_authoritativeness(case, case_base)
    elif method.startswith("harmonic"):
        try:
            beta = float(method.split("_")[1])
        except (IndexError, ValueError):
            raise ValueError(
                f"Invalid harmonic method format: {method}. "
                "Expected format: 'harmonic_X' where X is a number (e.g., 'harmonic_1')"
            )
        return harmonic_authoritativeness(case, case_base, beta)
    else:
        raise ValueError(
            f"Unknown authoritativeness method: {method}. "
            "Valid methods: 'relative', 'absolute', 'product', 'harmonic_X'"
        )
