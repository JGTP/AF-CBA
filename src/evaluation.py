from collections import Counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from case_base import CaseBase
from game import AFCBAClassifier
from utils import compute_classification_metrics


def evaluate_consistency(case_base: CaseBase) -> dict:
    consistency_percentage, n_removals, removal_indices = case_base.check_consistency()
    removed_cases = []
    if removal_indices:
        for idx in removal_indices:
            case_data = case_base.cases.iloc[idx].to_dict()
            case_data["original_index"] = idx
            case_data["original_dataframe_index"] = case_base.cases.index[idx]
            removed_cases.append(case_data)
    total_cases = len(case_base)
    removal_percentage = (n_removals / total_cases) * 100 if total_cases > 0 else 0
    return {
        "consistency_analysis": {
            "total_cases": total_cases,
            "consistency_percentage": round(consistency_percentage, 2),
            "inconsistent_cases": n_removals,
            "removal_percentage": round(removal_percentage, 2),
        },
        "cases_to_remove": {
            "count": n_removals,
            "indices": removal_indices,
            "detailed_cases": removed_cases,
        },
        "summary": {
            "action_required": f"Remove {n_removals} cases ({removal_percentage:.1f}% of dataset) to achieve consistency",
            "resulting_dataset_size": total_cases - n_removals,
        },
    }


def analyse_dataset_consistency(processors: list) -> dict:
    from case_base import create_case_base_from_data

    results = {}
    print(f"Analysing {len(processors)} dataset(s) for consistency...")
    for processor in processors:
        dataset_name = processor.get_dataset_name()
        print(f"\nAnalysing {dataset_name}...")
        try:
            X, y = processor.load_and_preprocess()
            print(f"Loaded: {len(X):,} samples, {len(X.columns)} features")
            case_base = create_case_base_from_data(X, y)
            consistency_report = evaluate_consistency(case_base)
            class_counts = Counter(y)
            class_balance_ratio = min(class_counts.values()) / max(
                class_counts.values()
            )
            results[dataset_name] = {
                "total_cases": len(case_base),
                "feature_count": len(case_base.feature_columns),
                "class_distribution": dict(class_counts),
                "class_balance_ratio": round(class_balance_ratio, 4),
                "consistency_percentage": consistency_report["consistency_analysis"][
                    "consistency_percentage"
                ],
                "cases_to_remove": consistency_report["consistency_analysis"][
                    "inconsistent_cases"
                ],
                "inconsistency_details": consistency_report,
            }
            consistency_pct = consistency_report["consistency_analysis"][
                "consistency_percentage"
            ]
            n_remove = consistency_report["consistency_analysis"]["inconsistent_cases"]
            if n_remove > 0:
                print(
                    f"   Consistency: {consistency_pct:.2f}% (remove {n_remove} cases)"
                )
                print(f"   Class balance: {class_balance_ratio:.2f}")
            else:
                print(f"   Consistency: {consistency_pct:.2f}% (perfect)")
                print(f"   Class balance: {class_balance_ratio:.2f}")
        except Exception as e:
            print(f"Failed to analyse {dataset_name}: {e}")
            continue
    return results


def benchmark_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
) -> dict:
    from case_base import create_case_base_from_data

    results = {}
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(
            random_state=random_state, n_estimators=100
        ),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_classification_metrics(y_test, y_pred)
        results[name] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "mcc": metrics["mcc"],
            "type": "sklearn",
        }
    case_base = create_case_base_from_data(X_train, y_train, random_state=random_state)
    classifier = AFCBAClassifier(case_base)
    outcome_counts = case_base.cases[case_base.target_column].value_counts()
    default_outcome = outcome_counts.idxmax()
    non_default_outcome = next(o for o in outcome_counts.index if o != default_outcome)
    predictions = []
    default_count = 0
    for _, test_case in tqdm(
        X_test.iterrows(),
        total=len(X_test),
        desc="AF-CBA Classification",
        mininterval=1,
    ):
        prediction, is_default = classifier.classify(
            test_case, default_outcome, non_default_outcome
        )
        predictions.append(prediction)
        if is_default:
            default_count += 1
    predictions = pd.Series(predictions, index=y_test.index)
    metrics = compute_classification_metrics(y_test, predictions)
    results["AF-CBA"] = {
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "mcc": metrics["mcc"],
        "justified_case_fraction": (len(predictions) - default_count)
        / len(predictions),
        "default_cases": default_count,
        "total_cases": len(predictions),
        "type": "afcba",
    }
    return results


def create_results_summary(results: dict) -> pd.DataFrame:
    summary_data = []
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "Type": metrics.get("type", "unknown"),
            "Accuracy": metrics.get("accuracy", 0),
            "F1-Score": metrics.get("f1_score", 0),
            "MCC": metrics.get("mcc", 0),
        }
        if model_name == "AF-CBA":
            row["Justified fraction"] = metrics.get("justified_case_fraction", 0)
            row["Default Cases"] = metrics.get("default_cases", 0)
        else:
            row["Justified fraction"] = 1.0
            row["Default Cases"] = 0
        summary_data.append(row)
    return pd.DataFrame(summary_data)


def print_performance_comparison(results: dict):
    print(f"\n{'=' * 60}")
    print("Performance Comparison")
    print(f"{'=' * 60}")
    summary_df = create_results_summary(results)
    print(
        f"\n{'Model':<15} {'Accuracy':<10} {'F1-Score':<10} {'MCC':<8} {'Justified':<10}"
    )
    print("-" * 60)
    for _, row in summary_df.iterrows():
        print(
            f"{row['Model']:<15} {row['Accuracy']:<10.2f} {row['F1-Score']:<10.2f} "
            f"{row['MCC']:<8.2f} {row['Justified fraction']:<10.2f}"
        )
    best_model = summary_df.loc[summary_df["Accuracy"].idxmax()]
    print(
        f"\nBest performing model: {best_model['Model']} "
        f"(Accuracy: {best_model['Accuracy']:.2f})"
    )
