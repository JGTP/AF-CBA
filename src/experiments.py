import json
import multiprocessing
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

from case_base import create_case_base_from_data
from game import AFCBAClassifier
from preprocessing import CaseBaseConfig, DatasetProcessor, RuntimeConfig
from utils import compute_classification_metrics, convert_to_serialisable
import wittgenstein as lw
from sklearn.ensemble import HistGradientBoostingClassifier
from conditional_compensation import ConditionalCompensationChecker

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_numerical = [f for f in numerical_features if f in X_train.columns]
    if not existing_numerical:
        return X_train, X_test
    X_train = X_train.copy()
    X_test = X_test.copy()
    scaler = StandardScaler()
    X_train[existing_numerical] = scaler.fit_transform(X_train[existing_numerical])
    X_test[existing_numerical] = scaler.transform(X_test[existing_numerical])
    return X_train, X_test


def apply_undersampling(
    X: pd.DataFrame,
    y: pd.Series,
    runtime_config: RuntimeConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    if not runtime_config.undersampling:
        return X, y
    sampling_strategy = 1.0 / runtime_config.undersampling_ratio
    print(f"   Undersampling: {dict(y.value_counts())}", end="")
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=runtime_config.random_state,
    )
    X_resampled, y_resampled = rus.fit_resample(X, y)
    y_resampled = pd.Series(y_resampled)
    print(f" → {dict(y_resampled.value_counts())}")
    return pd.DataFrame(X_resampled, columns=X.columns), y_resampled


def get_class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts().to_dict()
    total = len(y)
    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)
    ratio = (
        counts[majority_class] / counts[minority_class]
        if counts[minority_class] > 0
        else float("inf")
    )
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "total": total,
        "minority_class": int(minority_class),
        "majority_class": int(majority_class),
        "imbalance_ratio": round(ratio, 3),
    }


def enforce_consistency(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, dict]:
    temp_case_base = create_case_base_from_data(X, y, random_state=random_state)
    consistency_percentage, n_removals, removal_indices = (
        temp_case_base.check_consistency()
    )
    consistency_info = {
        "original_size": len(X),
        "consistency_percentage": consistency_percentage,
        "cases_removed": n_removals,
        "removal_percentage": (n_removals / len(X)) * 100 if len(X) > 0 else 0,
        "final_size": len(X) - n_removals,
    }
    if n_removals == 0:
        return X, y, consistency_info
    cases_df = temp_case_base.cases
    actual_indices_to_remove = cases_df.index[removal_indices].tolist()
    X_consistent = X.drop(actual_indices_to_remove)
    y_consistent = y.drop(actual_indices_to_remove)
    return X_consistent, y_consistent, consistency_info


def calculate_job_allocation(total_jobs: int, n_folds: int) -> tuple[int, int]:
    if total_jobs <= 0:
        total_jobs = multiprocessing.cpu_count()
    if total_jobs <= n_folds:
        return total_jobs, 1
    outer_jobs = min(n_folds, total_jobs)
    inner_jobs = max(1, total_jobs // outer_jobs)
    return outer_jobs, inner_jobs


def evaluate_ML_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
    return_predictions: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], dict[str, pd.Series]]:
    results = {}
    predictions = {}

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(
            random_state=random_state, n_estimators=100
        ),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_series = pd.Series(y_pred, index=y_test.index)
        metrics = compute_classification_metrics(y_test, y_pred)
        results[name] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "mcc": metrics["mcc"],
        }
        predictions[name] = y_pred_series

    try:
        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
        X_test_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        ripper = lw.RIPPER(random_state=42)
        ripper.fit(trainset=X_train_arr, y=y_train_arr)
        y_pred = ripper.predict(X_test_arr)
        y_pred_series = pd.Series(y_pred, index=y_test.index)

        metrics = compute_classification_metrics(y_test, y_pred)
        ruleset = ripper.ruleset_
        ruleset_size = len(ruleset.rules) if ruleset else 0

        results["RIPPER"] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "mcc": metrics["mcc"],
            "ruleset_size": ruleset_size,
        }
        predictions["RIPPER"] = y_pred_series
    except Exception as e:
        print(f"   ⚠ RIPPER evaluation failed: {e}")

    if return_predictions:
        return results, predictions
    return results


def compute_sklearn_metrics_on_subset(
    predictions: dict[str, pd.Series],
    y_true: pd.Series,
    indices: list,
) -> dict[str, dict]:
    if not indices:
        return {}

    subset_indices = list(set(indices) & set(y_true.index))
    if not subset_indices:
        return {}

    y_true_subset = y_true.loc[subset_indices]
    results = {}

    for model_name, y_pred in predictions.items():
        y_pred_subset = y_pred.loc[subset_indices]
        metrics = compute_classification_metrics(y_true_subset, y_pred_subset)
        results[model_name] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "mcc": metrics["mcc"],
            "n_cases": len(subset_indices),
        }

    return results


def evaluate_afcba_performance(
    case_base,
    test_cases: pd.DataFrame,
    true_outcomes: pd.Series,
    heuristic: str = "majority",
    n_jobs: int = 1,
    timeout_seconds: float | None = None,
    max_moves: int | None = None,
    collect_complexity: bool = True,
    collect_games: bool = False,
    **heuristic_kwargs,
) -> dict:
    from game import AFCBAClassifier
    from joblib import Parallel, delayed

    classifier = AFCBAClassifier(
        case_base,
        heuristic=heuristic,
        timeout_seconds=timeout_seconds,
        max_moves=max_moves,
        **heuristic_kwargs,
    )
    should_collect_games = collect_complexity or collect_games

    predictions = []
    fallback_mask = []  # True if heuristic was used (either both-justified or unjustified)
    winning_games = []
    case_ids = []  # Track case IDs for matching with both_justified_cases

    if n_jobs == 1:
        for idx, test_case in tqdm(
            test_cases.iterrows(), total=len(test_cases), desc="Classifying"
        ):
            case_ids.append(idx)
            if should_collect_games:
                pred, used_heuristic, game = classifier.classify(
                    test_case, case_id=idx, return_game=True
                )
                if game is not None:
                    winning_games.append(game)
            else:
                pred, used_heuristic = classifier.classify(test_case, case_id=idx)

            predictions.append(pred)
            fallback_mask.append(used_heuristic)
    else:
        def classify_single_case(test_case_idx):
            test_case = test_cases.iloc[test_case_idx]
            original_idx = test_cases.index[test_case_idx]
            prediction, used_heuristic = classifier.classify(
                test_case, case_id=original_idx
            )
            return original_idx, prediction, used_heuristic

        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=10)(
            delayed(classify_single_case)(idx)
            for idx in tqdm(range(len(test_cases)), desc="Classifying")
        )
        case_ids = [r[0] for r in results]
        predictions = [r[1] for r in results]
        fallback_mask = [r[2] for r in results]

    predictions = pd.Series(predictions, index=true_outcomes.index)
    fallback_mask = pd.Series(fallback_mask, index=true_outcomes.index)

    # Get both-justified cases from classifier
    both_justified_ids = set(classifier.both_justified_cases)

    # Create masks for the three categories
    both_justified_mask = pd.Series(
        [idx in both_justified_ids for idx in true_outcomes.index],
        index=true_outcomes.index
    )
    # Single-justified: heuristic was NOT used
    single_justified_mask = ~fallback_mask
    # Unjustified: heuristic was used AND not both-justified
    unjustified_mask = fallback_mask & ~both_justified_mask

    # Compute counts
    total_cases = len(predictions)
    single_justified_count = int(single_justified_mask.sum())
    both_justified_count = int(both_justified_mask.sum())
    unjustified_count = int(unjustified_mask.sum())

    # Compute fractions
    justified_fraction = single_justified_count / total_cases
    both_justified_fraction = both_justified_count / total_cases
    unjustified_fraction = unjustified_count / total_cases

    metrics = compute_classification_metrics(true_outcomes, predictions)

    # Compute justified-only metrics (only single-justified cases)
    justified_only_metrics = None
    if single_justified_mask.any():
        justified_preds = predictions[single_justified_mask]
        justified_true = true_outcomes[single_justified_mask]
        jm = compute_classification_metrics(justified_true, justified_preds)
        justified_only_metrics = {
            "accuracy": jm["accuracy"],
            "f1_score": jm["f1_score"],
            "mcc": jm["mcc"],
            "n_cases": single_justified_count,
        }

    timeout_count = len(classifier.timeout_cases)

    result = {
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "mcc": metrics["mcc"],
        "justified_fraction": justified_fraction,
        "both_justified_fraction": both_justified_fraction,
        "unjustified_fraction": unjustified_fraction,
        "total_cases": total_cases,
        "justified_count": single_justified_count,
        "both_justified_count": both_justified_count,
        "unjustified_count": unjustified_count,
        # Backwards compatibility
        "justified_cases": single_justified_count,
        "heuristic_cases": both_justified_count + unjustified_count,
        "heuristic": heuristic,
        "justified_only_metrics": justified_only_metrics,
        "justified_indices": true_outcomes.index[single_justified_mask].tolist(),
        "both_justified_indices": true_outcomes.index[both_justified_mask].tolist(),
    }
    if timeout_count > 0:
        result["timeout_count"] = timeout_count
        result["timeout_cases"] = classifier.timeout_cases
    if both_justified_count > 0:
        result["both_justified_cases"] = classifier.both_justified_cases
    if collect_complexity and winning_games:
        from complexity import aggregate_game_complexity
        result["complexity"] = aggregate_game_complexity(winning_games)
    if should_collect_games:
        result["winning_games"] = winning_games

    return result


def run_single_fold(
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    runtime_config: RuntimeConfig,
    inner_n_jobs: int,
    heuristic: str = "majority",
    timeout_seconds: float | None = None,
    max_moves: int | None = None,
    numerical_features: list[str] | None = None,
    **heuristic_kwargs,
) -> dict:
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if runtime_config.standardise_numerical and numerical_features:
        X_train, X_test = scale_features(X_train, X_test, numerical_features)
    X_train, y_train = apply_undersampling(X_train, y_train, runtime_config)
    consistency_info = None
    if runtime_config.make_consistent:
        X_train, y_train, consistency_info = enforce_consistency(
            X_train, y_train, runtime_config.random_state
        )
        if len(X_train) == 0:
            raise ValueError(f"Fold {fold}: All training cases removed!")
    case_base_config = CaseBaseConfig.from_runtime_config(runtime_config)

    conditional_checker = None
    if runtime_config.conditional:
        model = HistGradientBoostingClassifier(
            random_state=runtime_config.random_state, max_iter=100
        )
        model.fit(X_train, y_train)
        conditional_checker = ConditionalCompensationChecker(
            model=model,
            X=X_train,
            y=y_train,
            delta=runtime_config.delta,
            min_support=runtime_config.min_support,
            n_splits=runtime_config.n_splits,
        )

    case_base = create_case_base_from_data(
        X_train,
        y_train,
        random_state=runtime_config.random_state,
        conditional_checker=conditional_checker,
        config=case_base_config,
    )
    sklearn_results, sklearn_predictions = evaluate_ML_models(
        X_train, X_test, y_train, y_test, runtime_config.random_state,
        return_predictions=True,
    )
    afcba_results = evaluate_afcba_performance(
        case_base,
        X_test,
        y_test,
        heuristic=heuristic,
        n_jobs=inner_n_jobs,
        timeout_seconds=timeout_seconds,
        max_moves=max_moves,
        **heuristic_kwargs,
    )

    justified_indices = afcba_results.get("justified_indices", [])
    sklearn_justified_only = compute_sklearn_metrics_on_subset(
        sklearn_predictions, y_test, justified_indices
    )

    result = {
        "fold": fold,
        "sklearn_results": sklearn_results,
        "sklearn_justified_only": sklearn_justified_only,
        "afcba_results": afcba_results,
        "class_distribution": {
            "train": get_class_distribution(y_train),
            "test": get_class_distribution(y_test),
        },
    }
    if consistency_info:
        result["consistency_info"] = consistency_info
    return result


def serialise_game_to_dict(game, winning_strategy: list) -> dict:
    case_id = getattr(game, "case_id", None)
    outcome = getattr(game, "predicted_outcome", None)
    framework = getattr(game, "framework", None)

    focus_case_data = None
    best_precedents_info = []
    case_differences = {}

    if framework is not None:
        focus_case = framework.focus_case
        focus_case_data = focus_case.to_dict()

        difference_cache = getattr(framework, "difference_cache", None)
        case_base = framework.case_base

        cited_cases = set()
        for move in game.moves.values():
            if move.argument:
                content = move.argument.get_content_data()
                if content.get("type") == "Citation":
                    case_name = content.get("case_name")
                    if case_name:
                        try:
                            cited_cases.add(int(case_name))
                        except (ValueError, TypeError):
                            pass

        for case_idx in cited_cases:
            try:
                if difference_cache is not None:
                    diffs = list(difference_cache.get(case_idx))
                else:
                    case_data = case_base.cases.loc[case_idx]
                    diffs = case_base.calculate_differences(case_data, focus_case)
                case_differences[str(case_idx)] = sorted(diffs)
            except (KeyError, IndexError):
                pass

        for move in winning_strategy:
            if (
                move.target_move_id == 0
                and move.player.value == "PRO"
                and move.argument
            ):
                content = move.argument.get_content_data()
                if content.get("type") == "Citation":
                    case_name = content.get("case_name")
                    diffs = case_differences.get(case_name, [])
                    best_precedents_info.append(
                        {
                            "case_name": case_name,
                            "differences_vs_focus": diffs,
                            "num_differences": len(diffs),
                        }
                    )

    def serialise_move(move) -> dict:
        move_dict = {
            "move_id": move.move_id,
            "player": move.player.value,
            "role": move.role.value,
            "target_move_id": move.target_move_id,
            "is_backtracked": move.is_backtracked,
            "description": move.description(),
        }

        if move.argument:
            arg_content = move.argument.get_content_data()
            move_dict["argument"] = {
                "name": move.argument.name,
                "content": arg_content,
            }

            if arg_content.get("type") == "Citation":
                case_name = arg_content.get("case_name")
                if case_name and case_name in case_differences:
                    move_dict["argument"]["differences_vs_focus"] = case_differences[
                        case_name
                    ]
                    move_dict["argument"]["num_differences"] = len(
                        case_differences[case_name]
                    )

        if move.target_argument:
            target_content = move.target_argument.get_content_data()
            move_dict["target_argument"] = {
                "name": move.target_argument.name,
                "content": target_content,
            }

            if target_content.get("type") == "Citation":
                target_case_name = target_content.get("case_name")
                if target_case_name and target_case_name in case_differences:
                    move_dict["target_argument"]["differences_vs_focus"] = (
                        case_differences[target_case_name]
                    )
                    move_dict["target_argument"]["num_differences"] = len(
                        case_differences[target_case_name]
                    )

            if (
                move.argument
                and move.argument.get_content_data().get("type") == "Citation"
                and target_content.get("type") == "Citation"
            ):
                attacker_name = move.argument.get_content_data().get("case_name")
                target_name = target_content.get("case_name")

                if (
                    attacker_name in case_differences
                    and target_name in case_differences
                ):
                    attacker_diffs = set(case_differences[attacker_name])
                    target_diffs = set(case_differences[target_name])

                    move_dict["counterexample_analysis"] = {
                        "attacker_case": attacker_name,
                        "target_case": target_name,
                        "attacker_num_diffs": len(attacker_diffs),
                        "target_num_diffs": len(target_diffs),
                        "is_valid_attack": not (target_diffs < attacker_diffs),
                        "attacker_dominates_target": attacker_diffs < target_diffs,
                        "diffs_only_in_attacker": sorted(attacker_diffs - target_diffs),
                        "diffs_only_in_target": sorted(target_diffs - attacker_diffs),
                    }

        return move_dict

    return {
        "case_id": case_id,
        "predicted_outcome": outcome,
        "total_moves": len(game.moves),
        "winning_strategy_length": len(winning_strategy),
        "focus_case": focus_case_data,
        "best_precedents": best_precedents_info,
        "all_case_differences": case_differences,
        "winning_strategy": [serialise_move(m) for m in winning_strategy],
        "all_moves": [
            serialise_move(game.moves[mid]) for mid in sorted(game.moves.keys())
        ],
    }


def visualise_winning_strategies(
    winning_games: list,
    output_dir: Path,
    dataset_name: str,
    max_visualisations: int = 100,
    skip_trivial: bool = True,
) -> list[Path]:
    from visualisation import visualise_game_result

    if not winning_games:
        print("No winning games to visualise.")
        return []
    vis_dir = output_dir / "visualisations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    visualised_count = 0
    skipped_trivial = 0
    print(f"\n🎨 Visualising up to {max_visualisations} winning strategies...")
    if skip_trivial:
        print("   (skipping trivial single-citation strategies)")
    for game in winning_games:
        if visualised_count >= max_visualisations:
            break
        winning_strategy = game.get_winning_strategy()
        if not winning_strategy:
            continue
        if skip_trivial and len(winning_strategy) <= 1:
            skipped_trivial += 1
            continue
        visualised_count += 1
        png_filename = f"{dataset_name}_strategy_{visualised_count:02d}.png"
        json_filename = f"{dataset_name}_strategy_{visualised_count:02d}.json"
        png_path = vis_dir / png_filename
        json_path = vis_dir / json_filename
        case_id = getattr(game, "case_id", visualised_count)
        outcome = getattr(game, "predicted_outcome", "unknown")
        num_moves = len(winning_strategy)
        title = f"Strategy {visualised_count} (Case {case_id}, Outcome: {outcome}, {num_moves} moves)"
        try:
            visualise_game_result(game, png_path, title=title)
            created_files.append(png_path)
            print(f"  ✓ Saved: {png_path}")
        except Exception as e:
            print(f"  ✗ Failed to visualise strategy {visualised_count}: {e}")
        try:
            game_data = serialise_game_to_dict(game, winning_strategy)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(convert_to_serialisable(game_data), f, indent=2)
            created_files.append(json_path)
            print(f"  ✓ Saved: {json_path}")
        except Exception as e:
            print(f"  ✗ Failed to export JSON for strategy {visualised_count}: {e}")
    if skipped_trivial > 0:
        print(f"\n   Skipped {skipped_trivial} trivial strategies")
    if visualised_count < max_visualisations:
        print(f"   Only found {visualised_count} non-trivial strategies")
    print(f"\n📁 Visualisations saved to: {vis_dir}")
    return created_files


class Experiment:
    def __init__(
        self,
        processor: DatasetProcessor,
        runtime_config: RuntimeConfig,
        heuristic: str = "majority",
        timeout_seconds: float | None = None,
        max_moves: int | None = None,
        **heuristic_kwargs,
    ):
        self.processor = processor
        self.runtime_config = runtime_config
        self.heuristic = heuristic
        self.heuristic_kwargs = heuristic_kwargs
        self.timeout_seconds = timeout_seconds
        self.max_moves = max_moves
        self.results = {}
        self._X_dev = None
        self._y_dev = None
        self._X_holdout = None
        self._y_holdout = None

    def _get_numerical_features(self) -> list[str]:
        return self.processor.dataset_config.features.get("numerical", [])

    def _print_configuration_summary(self):
        print("\n📋 Configuration:")
        print(f"   Dataset: {self.processor.get_dataset_name()}")
        print(f"   Heuristic: {self.heuristic}")
        if self.heuristic_kwargs:
            print(f"   Heuristic params: {self.heuristic_kwargs}")
        print(f"   Consistency enforcement: {self.runtime_config.make_consistent}")
        print(f"   Undersampling: {self.runtime_config.undersampling}")
        print(f"   Authoritativeness: {self.runtime_config.authoritativeness}")
        if self.runtime_config.authoritativeness:
            print(f"     - method: {self.runtime_config.auth_method}")
        print(f"   Preference method: {self.runtime_config.get_preference_method()}")
        if self.runtime_config.conditional:
            print(f"     - delta: {self.runtime_config.delta}")
        if self.timeout_seconds is not None or self.max_moves is not None:
            print(f"   Timeout: {self.timeout_seconds}s / {self.max_moves} moves")

    def _prepare_data_splits(self):
        if self._X_dev is not None:
            return
        X, y = self.processor.load_and_preprocess()
        print(f"\n📈 Full dataset: {len(X)} samples, {len(X.columns)} features")
        print(f"   Class distribution: {y.value_counts().to_dict()}")
        self._X_dev, self._X_holdout, self._y_dev, self._y_holdout = train_test_split(
            X,
            y,
            test_size=self.runtime_config.test_size,
            stratify=y,
            random_state=self.runtime_config.random_state,
        )
        print(
            f"\n🔀 Initial stratified split (holdout_size={self.runtime_config.test_size}):"
        )
        print(f"   Development set: {len(self._X_dev)} samples (for CV)")
        print(f"   Holdout set: {len(self._X_holdout)} samples (final evaluation)")
        print(f"   Dev class distribution: {self._y_dev.value_counts().to_dict()}")
        print(
            f"   Holdout class distribution: {self._y_holdout.value_counts().to_dict()}"
        )

    def run_full_experiment(
        self,
        cv_folds: int = 5,
        n_jobs: int = -1,
        use_parallel: bool = True,
        force_sequential_inner: bool = False,
        evaluate_complexity: bool = True,
        visualise_strategies: bool = False,
    ) -> dict[str, Any]:
        print(f"\n{'=' * 80}")
        print("AF-CBA Full Experiment")
        print(f"{'=' * 80}")
        self._print_configuration_summary()
        self._prepare_data_splits()
        full_class_distribution = get_class_distribution(
            pd.concat([self._y_dev, self._y_holdout])
        )
        print(f"\n{'=' * 80}")
        print("Phase 1: Cross-Validation on Development Set")
        print(f"{'=' * 80}")
        cv_results = self._run_cross_validation_on_data(
            self._X_dev,
            self._y_dev,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            use_parallel=use_parallel,
            force_sequential_inner=force_sequential_inner,
        )
        print(f"\n{'=' * 80}")
        print("Phase 2: Final Evaluation on Holdout Set")
        print(f"{'=' * 80}")
        holdout_results = self._run_holdout_evaluation_on_data(
            self._X_dev,
            self._y_dev,
            self._X_holdout,
            self._y_holdout,
            evaluate_complexity=evaluate_complexity,
            visualise_strategies=visualise_strategies,
        )
        self.results = {
            "cross_validation": cv_results,
            "holdout_evaluation": holdout_results,
            "data_split_info": {
                "total_samples": len(self._X_dev) + len(self._X_holdout),
                "development_samples": len(self._X_dev),
                "holdout_samples": len(self._X_holdout),
                "holdout_fraction": self.runtime_config.test_size,
                "cv_folds": cv_folds,
            },
            "class_distribution": {
                "full_dataset": full_class_distribution,
                "development": get_class_distribution(self._y_dev),
                "holdout": get_class_distribution(self._y_holdout),
            },
        }
        return self.results

    def _run_cross_validation_on_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        n_jobs: int = -1,
        use_parallel: bool = True,
        force_sequential_inner: bool = False,
    ) -> dict[str, Any]:
        print(f"   CV data: {len(X)} samples")
        print(f"   CV Folds: {cv_folds}")
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        outer_n_jobs, inner_n_jobs = calculate_job_allocation(n_jobs, cv_folds)
        if force_sequential_inner:
            inner_n_jobs = 1
        print(f"   Parallelisation: {outer_n_jobs} outer × {inner_n_jobs} inner")
        skf = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.runtime_config.random_state,
        )
        folds = list(skf.split(X, y))
        numerical_features = self._get_numerical_features()
        if use_parallel and outer_n_jobs > 1:
            print(f"\n🔄 Running {cv_folds}-fold CV in parallel...")
            start_time = time.time()
            fold_results = Parallel(n_jobs=outer_n_jobs, verbose=10)(
                delayed(run_single_fold)(
                    fold,
                    train_idx,
                    test_idx,
                    X,
                    y,
                    self.runtime_config,
                    inner_n_jobs,
                    self.heuristic,
                    self.timeout_seconds,
                    self.max_moves,
                    numerical_features,
                    **self.heuristic_kwargs,
                )
                for fold, (train_idx, test_idx) in enumerate(folds, 1)
            )
            elapsed = time.time() - start_time
            print(f"\n✓ CV completed in {elapsed:.1f}s")
        else:
            print(f"\n🔄 Running {cv_folds}-fold CV sequentially...")
            fold_results = []
            for fold, (train_idx, test_idx) in enumerate(folds, 1):
                print(f"\n--- Fold {fold}/{cv_folds} ---")
                result = run_single_fold(
                    fold,
                    train_idx,
                    test_idx,
                    X,
                    y,
                    self.runtime_config,
                    inner_n_jobs,
                    self.heuristic,
                    self.timeout_seconds,
                    self.max_moves,
                    numerical_features,
                    **self.heuristic_kwargs,
                )
                fold_results.append(result)
        sklearn_results = [r["sklearn_results"] for r in fold_results]
        afcba_results = [r["afcba_results"] for r in fold_results]
        sklearn_justified_only = [
            r.get("sklearn_justified_only", {}) for r in fold_results
        ]
        consistency_infos = [
            r["consistency_info"] for r in fold_results if "consistency_info" in r
        ]
        fold_distributions = [r["class_distribution"] for r in fold_results]
        results = self._calculate_cv_averages(
            sklearn_results, afcba_results, sklearn_justified_only
        )
        results["fold_class_distributions"] = fold_distributions
        if consistency_infos:
            results["consistency_summary"] = {
                "total_removed": sum(c["cases_removed"] for c in consistency_infos),
                "avg_removal_percentage": np.mean(
                    [c["removal_percentage"] for c in consistency_infos]
                ),
                "per_fold": consistency_infos,
            }
        total_timeouts = sum(r.get("timeout_count", 0) for r in afcba_results)
        if total_timeouts > 0:
            results["timeout_summary"] = {
                "total_timeouts": total_timeouts,
                "per_fold": [
                    {"fold": i + 1, "timeout_count": r.get("timeout_count", 0)}
                    for i, r in enumerate(afcba_results)
                    if r.get("timeout_count", 0) > 0
                ],
            }
        total_both_justified = sum(
            r.get("both_justified_count", 0) for r in afcba_results
        )
        if total_both_justified > 0:
            results["both_justified_summary"] = {
                "total_both_justified": total_both_justified,
                "per_fold": [
                    {
                        "fold": i + 1,
                        "both_justified_count": r.get("both_justified_count", 0),
                    }
                    for i, r in enumerate(afcba_results)
                    if r.get("both_justified_count", 0) > 0
                ],
            }
        self._print_cv_summary(results)
        return results

    def _print_cv_summary(self, cv_averages: dict):
        print("\n" + "=" * 80)
        print("Cross-Validation Results (mean ± std)")
        print("=" * 80)
        print(
            "\n{:<20} {:>14} {:>14} {:>14} {:>14}".format(
                "Model", "Accuracy", "F1", "MCC", "Note"
            )
        )
        print("-" * 80)

        for model_name, metrics in cv_averages["sklearn"].items():
            note = "--"
            if model_name == "RIPPER" and "ruleset_size" in metrics:
                note = f"{metrics['ruleset_size']} rules"
            print(
                "{:<20} {:>14} {:>14} {:>14} {:>14}".format(
                    model_name,
                    metrics["accuracy"],
                    metrics["f1_score"],
                    metrics["mcc"],
                    note,
                )
            )

        afcba_metrics = cv_averages["afcba"]["AF-CBA"]
        note = f"j:{afcba_metrics['justified_fraction']}"
        if "both_justified_fraction" in afcba_metrics:
            note += f" b:{afcba_metrics['both_justified_fraction']}"
        print(
            "{:<20} {:>14} {:>14} {:>14} {:>14}".format(
                "AF-CBA",
                afcba_metrics["accuracy"],
                afcba_metrics["f1_score"],
                afcba_metrics["mcc"],
                note,
            )
        )


    def _calculate_cv_averages(
        self,
        sklearn_results: list[dict],
        afcba_results: list[dict],
        sklearn_justified_only: list[dict] | None = None,
    ) -> dict[str, Any]:
        def mean_pm_std(values):
            mean = np.mean(values)
            std = np.std(values)
            return f"{mean:.2f}±{std:.2f}"

        def mean_pm_std_int(values):
            mean = np.mean(values)
            std = np.std(values)
            return f"{mean:.1f}±{std:.1f}"

        sklearn_avg = {}
        all_model_names = set()
        for fold_result in sklearn_results:
            all_model_names.update(fold_result.keys())

        for model_name in all_model_names:
            model_metrics = {}
            folds_with_model = [r for r in sklearn_results if model_name in r]
            if not folds_with_model:
                continue

            for metric in folds_with_model[0][model_name].keys():
                values = [r[model_name][metric] for r in folds_with_model]
                if metric == "ruleset_size":
                    model_metrics[metric] = mean_pm_std_int(values)
                else:
                    model_metrics[metric] = mean_pm_std(values)

            sklearn_avg[model_name] = model_metrics

        afcba_metrics = [
            "accuracy", "f1_score", "mcc",
            "justified_fraction", "both_justified_fraction", "unjustified_fraction"
        ]
        afcba_avg = {}
        for metric in afcba_metrics:
            values = [r[metric] for r in afcba_results if metric in r]
            if values:
                afcba_avg[metric] = mean_pm_std(values)

        afcba_justified_only_list = [
            r.get("justified_only_metrics") for r in afcba_results
            if r.get("justified_only_metrics") is not None
        ]
        if afcba_justified_only_list:
            for metric in ["accuracy", "f1_score", "mcc"]:
                values = [r[metric] for r in afcba_justified_only_list]
                afcba_avg[f"justified_only_{metric}"] = mean_pm_std(values)

        result = {
            "sklearn": sklearn_avg,
            "afcba": {"AF-CBA": afcba_avg},
        }

        if sklearn_justified_only:
            sklearn_justified_avg = {}
            all_justified_models = set()
            for fold_result in sklearn_justified_only:
                if fold_result:
                    all_justified_models.update(fold_result.keys())

            for model_name in all_justified_models:
                model_metrics = {}
                folds_with_model = [
                    r for r in sklearn_justified_only
                    if r and model_name in r
                ]
                if not folds_with_model:
                    continue

                for metric in ["accuracy", "f1_score", "mcc"]:
                    values = [r[model_name][metric] for r in folds_with_model]
                    model_metrics[metric] = mean_pm_std(values)

                sklearn_justified_avg[model_name] = model_metrics

            if sklearn_justified_avg:
                result["sklearn_justified_only"] = sklearn_justified_avg

        return result

    def _run_holdout_evaluation_on_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        evaluate_complexity: bool = True,
        visualise_strategies: bool = False,
    ) -> dict[str, Any]:
        print(f"   Training on: {len(X_train)} samples (full development set)")
        print(f"   Testing on: {len(X_test)} samples (holdout set)")
        if self.runtime_config.standardise_numerical:
            numerical_features = self._get_numerical_features()
            X_train, X_test = scale_features(X_train, X_test, numerical_features)
        X_train, y_train = apply_undersampling(X_train, y_train, self.runtime_config)
        consistency_info = None
        if self.runtime_config.make_consistent:
            print("\n🔧 Applying consistency enforcement to training set...")
            X_train, y_train, consistency_info = enforce_consistency(
                X_train, y_train, self.runtime_config.random_state
            )
            if len(X_train) == 0:
                raise ValueError("All training cases removed for consistency!")
        case_base_config = CaseBaseConfig.from_runtime_config(self.runtime_config)

        conditional_checker = None
        if self.runtime_config.conditional:
            model = HistGradientBoostingClassifier(
                random_state=self.runtime_config.random_state, max_iter=100
            )
            model.fit(X_train, y_train)
            conditional_checker = ConditionalCompensationChecker(
                model=model,
                X=X_train,
                y=y_train,
                delta=self.runtime_config.delta,
                min_support=self.runtime_config.min_support,
                n_splits=self.runtime_config.n_splits,
            )

        case_base = create_case_base_from_data(
            X_train,
            y_train,
            random_state=self.runtime_config.random_state,
            conditional_checker=conditional_checker,
            config=case_base_config,
        )
        sklearn_results, sklearn_predictions = evaluate_ML_models(
            X_train, X_test, y_train, y_test, self.runtime_config.random_state,
            return_predictions=True,
        )
        collect_games = evaluate_complexity or visualise_strategies
        afcba_results = evaluate_afcba_performance(
            case_base,
            X_test,
            y_test,
            heuristic=self.heuristic,
            n_jobs=1,
            timeout_seconds=self.timeout_seconds,
            max_moves=self.max_moves,
            collect_complexity=evaluate_complexity,
            collect_games=collect_games,
            **self.heuristic_kwargs,
        )

        justified_indices = afcba_results.get("justified_indices", [])
        sklearn_justified_only = compute_sklearn_metrics_on_subset(
            sklearn_predictions, y_test, justified_indices
        )

        results = {
            "sklearn": sklearn_results,
            "sklearn_justified_only": sklearn_justified_only,
            "afcba": {"AF-CBA": afcba_results},
            "class_distribution": {
                "train": get_class_distribution(y_train),
                "test": get_class_distribution(y_test),
            },
        }
        if consistency_info:
            results["consistency_info"] = consistency_info
        if evaluate_complexity and "complexity" in afcba_results:
            from complexity import get_decision_tree_complexity

            print("\n📐 Structural complexity analysis...")
            decision_tree = DecisionTreeClassifier(
                random_state=self.runtime_config.random_state
            )
            decision_tree.fit(X_train, y_train)
            complexity_results = {
                "afcba": afcba_results["complexity"],
                "decision_tree": get_decision_tree_complexity(decision_tree),
            }
            results["complexity"] = complexity_results
            self._print_complexity_summary(complexity_results)
        if visualise_strategies and "winning_games" in afcba_results:
            winning_games = afcba_results["winning_games"]
            output_dir = Path("visualisations")
            dataset_name = self.processor.get_dataset_name().lower()
            visualisation_paths = visualise_winning_strategies(
                winning_games,
                output_dir,
                dataset_name,
                max_visualisations=100,
            )
            results["visualisations"] = {
                "count": len(visualisation_paths),
                "paths": [str(p) for p in visualisation_paths],
            }
        if "winning_games" in afcba_results:
            del afcba_results["winning_games"]
        self._print_holdout_summary(results)
        return results

    def _print_complexity_summary(self, results: dict):
        print("\n" + "=" * 70)
        print("Structural Complexity Analysis (Table 5.3)")
        print("=" * 70)
        print(
            "\n{:<20} {:>12} {:>12} {:>12} {:>12}".format(
                "", "Avg Depth", "Max Depth", "Avg Breadth", "Max Breadth"
            )
        )
        print("-" * 70)
        afcba = results["afcba"]
        print(
            "{:<20} {:>12.2f} {:>12} {:>12.2f} {:>12}".format(
                "AF-CBA",
                afcba["mean_depth"],
                int(afcba["max_depth"]),
                afcba["mean_breadth"],
                int(afcba["max_breadth"]),
            )
        )
        dt = results["decision_tree"]
        print(
            "{:<20} {:>12.2f} {:>12} {:>12.2f} {:>12}".format(
                "Decision Tree",
                dt["mean_depth"],
                int(dt["max_depth"]),
                dt["mean_breadth"],
                int(dt["max_breadth"]),
            )
        )

    def _print_holdout_summary(self, results: dict):
        print("\n" + "=" * 80)
        print("Holdout Evaluation Results")
        print("=" * 80)
        print(
            "\n{:<20} {:>12} {:>12} {:>12} {:>14}".format(
                "Model", "Accuracy", "F1", "MCC", "Note"
            )
        )
        print("-" * 80)

        for model_name, metrics in results["sklearn"].items():
            note = "--"
            if model_name == "RIPPER" and "ruleset_size" in metrics:
                note = f"{metrics['ruleset_size']} rules"
            print(
                "{:<20} {:>12.2f} {:>12.2f} {:>12.2f} {:>14}".format(
                    model_name,
                    metrics["accuracy"],
                    metrics["f1_score"],
                    metrics["mcc"],
                    note,
                )
            )

        afcba_metrics = results["afcba"]["AF-CBA"]
        note = f"j:{afcba_metrics['justified_fraction']:.2f}"
        if "both_justified_fraction" in afcba_metrics:
            note += f" b:{afcba_metrics['both_justified_fraction']:.2f}"
        print(
            "{:<20} {:>12.2f} {:>12.2f} {:>12.2f} {:>14}".format(
                "AF-CBA",
                afcba_metrics["accuracy"],
                afcba_metrics["f1_score"],
                afcba_metrics["mcc"],
                note,
            )
        )

    def export_results(self, output_dir: Path = Path("results")):
        if not self.results:
            print("No results to export.")
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = self.processor.get_dataset_name().lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{timestamp}.json"
        configuration = {
            "dataset": dataset_name,
            "random_state": self.runtime_config.random_state,
            "heuristic": self.heuristic,
            "make_consistent": self.runtime_config.make_consistent,
            "undersampling": self.runtime_config.undersampling,
            "undersampling_ratio": self.runtime_config.undersampling_ratio,
            "sample_size": self.runtime_config.sample_size,
            "preference_method": self.runtime_config.get_preference_method(),
            "authoritativeness": self.runtime_config.authoritativeness,
            "auth_method": self.runtime_config.auth_method,
        }
        output_data = {
            "configuration": configuration,
            "results": self.results,
        }
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(convert_to_serialisable(output_data), f, indent=2)
        print(f"\n📁 Results exported to: {output_path}")