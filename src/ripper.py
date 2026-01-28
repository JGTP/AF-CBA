import numpy as np
import pandas as pd
import wittgenstein as lw
from sklearn.model_selection import StratifiedKFold

from utils import convert_to_serialisable, save_json_results


def train_RIPPER_classifier(X_train, y_train):
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
    ripper = lw.RIPPER(random_state=42)
    try:
        ripper.fit(trainset=X_train_array, y=y_train_array)
    except Exception as e:
        print(f"Error during training: {e!s}")
        return None, None
    ruleset = ripper.ruleset_
    for rule in ruleset.rules:
        tuple = rule.class_ns_
        rule.predicted_class = 0 if tuple[0] > tuple[1] else 1
    return ripper, ruleset


def analyse_rulesets_globally(all_rulesets, full_dataset, target_name):
    global_rule_summary = {}
    for fold, ruleset in enumerate(all_rulesets, start=1):
        ruleset_str = str(ruleset)
        if ruleset_str not in global_rule_summary:
            global_rule_summary[ruleset_str] = {
                "ruleset": ruleset,
                "folds": [],
                "fold_positions": [],
                "rules": [],
            }
        for rule in ruleset.rules:
            global_rule_summary[ruleset_str]["rules"].append(rule)
        global_rule_summary[ruleset_str]["folds"].append(fold)
        global_rule_summary[ruleset_str]["fold_positions"].append(
            len(global_rule_summary[ruleset_str]["fold_positions"]) + 1
        )
    total_folds = len(all_rulesets)
    for ruleset_str, summary in global_rule_summary.items():
        summary["stability"] = {
            "fold_frequency": len(summary["folds"]) / total_folds,
            "avg_position": (
                np.mean(summary["fold_positions"]) if summary["fold_positions"] else 0
            ),
            "position_std": (
                np.std(summary["fold_positions"])
                if len(summary["fold_positions"]) > 1
                else 0
            ),
        }
    return global_rule_summary


def cross_validate_RIPPER(X, y, n_splits=5, progress_logger=None):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rule_occurrences = {}
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        ripper, ruleset = train_RIPPER_classifier(X_train, y_train)
        if ruleset is None:
            continue
        for rule in ruleset.rules:
            rule_str = str(rule)
            if rule_str not in rule_occurrences:
                rule_occurrences[rule_str] = {"count": 0, "rule": rule}
            rule_occurrences[rule_str]["count"] += 1
        if progress_logger:
            progress_logger.update_progress("ripper", 1)
    return [info["rule"] for info in rule_occurrences.values() if info["count"] > 1]


def format_rule_for_json(rule, metrics, feature_names):
    def format_condition(cond):
        feature_name = feature_names[cond.feature]
        operator = determine_operator(str(cond))
        value = convert_to_serialisable(cond.val)
        return {"feature": feature_name, "operator": operator, "value": value}

    return {
        "rule_structure": {
            "conditions": [format_condition(cond) for cond in rule.conds],
            "predicted_class": rule.predicted_class,
        },
        "raw_rule_str": str(rule),
        "stability": metrics["stability"],
    }


def format_rule_condition(condition, feature_names):
    feature_name = feature_names[condition.feature]
    cond_str = str(condition)
    if "<=" in cond_str:
        operator = "<="
    elif ">=" in cond_str:
        operator = ">="
    elif "=" in cond_str:
        operator = "="
    elif "<" in cond_str:
        operator = "<"
    elif ">" in cond_str:
        operator = ">"
    else:
        operator = "="
    value = convert_to_serialisable(condition.val)
    if isinstance(value, (int, float)):
        formatted_value = "MISSING" if value in [-99, -999] else f"{value:g}"
    else:
        formatted_value = str(value).strip()
    return f"{feature_name} {operator} {formatted_value}"


def determine_operator(cond):
    cond_str = str(cond)
    if "<=" in cond_str:
        return "<="
    elif ">=" in cond_str:
        return ">="
    elif "=" in cond_str:
        return "=="
    elif "<" in cond_str:
        return "<"
    elif ">" in cond_str:
        return ">"
    return "=="


def export_analysis_results(
    rule_analysis=None, feature_names=None, output_dir="results"
):
    if not (rule_analysis and feature_names):
        return {}
    formatted_rulesets = [
        {
            "raw_ruleset_str": ruleset_str,
            "rules": [
                format_rule_for_json(
                    rule, {"stability": summary["stability"]}, feature_names
                )
                for rule in summary["rules"]
            ],
            "stability": summary["stability"],
        }
        for ruleset_str, summary in rule_analysis.items()
    ]
    ruleset_data = {
        "metadata": {
            "total_rulesets": len(formatted_rulesets),
        },
        "rulesets": formatted_rulesets,
    }
    output_path = save_json_results(ruleset_data, output_dir, "ruleset_analysis")
    return {"ruleset_analysis_path": output_path}
