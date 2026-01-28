import argparse
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

from case_base import create_case_base_from_data
from evaluation import analyse_dataset_consistency
from experiments import Experiment
from game import AFCBAClassifier
from latex import generate_latex_tables
from preprocessing import (
    AdmissionProcessor,
    ChurnProcessor,
    COMPASProcessor,
    GTDProcessor,
    RuntimeConfig,
)

DATASETS = ["admission", "churn", "compas", "gtd"]


def get_processor(dataset_name: str, runtime_config: RuntimeConfig):
    processors = {
        "admission": AdmissionProcessor,
        "churn": ChurnProcessor,
        "gtd": GTDProcessor,
        "compas": COMPASProcessor,
    }
    factory = processors.get(dataset_name.lower())
    return factory(runtime_config=runtime_config) if factory else None


def build_runtime_config(args) -> RuntimeConfig:
    return RuntimeConfig(
        one_hot_categoricals=not getattr(args, "no_one_hot", False),
        standardise_numerical=not getattr(args, "no_standardise", False),
        sample_size=getattr(args, "sample_size", None),
        make_consistent=getattr(args, "make_consistent", False),
        undersampling=getattr(args, "undersampling", False),
        undersampling_ratio=getattr(args, "undersampling_ratio", 2.0),
        authoritativeness=getattr(args, "authoritativeness", True),
        auth_method=getattr(args, "auth_method", "harmonic_1"),
        conditional=getattr(args, "conditional", False),
        delta=getattr(args, "delta", 2.0),
        min_support=getattr(args, "min_support", 10),
        n_splits=getattr(args, "n_splits", 5),
        random_state=getattr(args, "random_state", 42),
        test_size=getattr(args, "test_size", 0.2),
    )


def build_heuristic_kwargs(args) -> dict:
    if getattr(args, "heuristic", "majority") == "nearest_neighbour":
        return {"k": getattr(args, "heuristic_k", 3)}
    return {}


def run_single_experiment(args, dataset_name: str = None):
    dataset_name = dataset_name or args.dataset
    runtime_config = build_runtime_config(args)
    processor = get_processor(dataset_name, runtime_config)
    if processor is None:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {', '.join(DATASETS)}")
        return None
    experiment = Experiment(
        processor,
        runtime_config=runtime_config,
        heuristic=args.heuristic,
        timeout_seconds=args.timeout,
        max_moves=args.max_moves,
        **build_heuristic_kwargs(args),
    )
    print(f"\n{'=' * 80}")
    print(f"Experiment: {processor.get_dataset_name()}")
    print(f"{'=' * 80}")
    results = experiment.run_full_experiment(
        cv_folds=args.folds,
        n_jobs=args.n_jobs,
        use_parallel=not getattr(args, "no_parallel", False),
        force_sequential_inner=getattr(args, "sequential_inner", False),
        evaluate_complexity=getattr(args, "complexity", False),
        visualise_strategies=getattr(args, "visualise", False),
    )
    experiment.results = results
    experiment.export_results()
    return results


def run_delta_sweep(args) -> dict:
    """Run experiments across multiple delta values to find optimal threshold."""
    delta_values = args.delta_values
    if delta_values is None:
        delta_values = [
            0.0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "delta_values": delta_values,
        "folds": args.folds,
        "experiments": [],
    }
    print(f"\n{'=' * 60}")
    print(f"Delta Sweep: {args.dataset.upper()}")
    print(f"{'=' * 60}")
    print(f"Testing delta values: {delta_values}")
    for delta in delta_values:
        print(f"\n[Delta = {delta}]")
        if args.dry_run:
            print(f"  Would run with delta={delta}")
            all_results["experiments"].append({"delta": delta, "results": None})
            continue
        try:
            runtime_config = RuntimeConfig(
                one_hot_categoricals=not getattr(args, "no_one_hot", False),
                standardise_numerical=not getattr(args, "no_standardise", False),
                sample_size=getattr(args, "sample_size", None),
                make_consistent=getattr(args, "make_consistent", False),
                undersampling=getattr(args, "undersampling", False),
                undersampling_ratio=getattr(args, "undersampling_ratio", 2.0),
                authoritativeness=getattr(args, "authoritativeness", True),
                auth_method=getattr(args, "auth_method", "harmonic_1"),
                conditional=True,
                delta=delta,
                min_support=getattr(args, "min_support", 10),
                n_splits=getattr(args, "n_splits", 5),
                random_state=args.random_state,
                test_size=args.test_size,
            )
            processor = get_processor(args.dataset, runtime_config)
            experiment = Experiment(
                processor,
                runtime_config=runtime_config,
                heuristic=args.heuristic,
                timeout_seconds=args.timeout,
                max_moves=args.max_moves,
                **build_heuristic_kwargs(args),
            )
            results = experiment.run_full_experiment(
                cv_folds=args.folds,
                n_jobs=args.n_jobs,
            )
            all_results["experiments"].append({"delta": delta, "results": results})
        except Exception as e:
            print(f"  Experiment failed: {e}")
            all_results["experiments"].append(
                {"delta": delta, "results": None, "error": str(e)}
            )
    if not args.dry_run:
        _print_delta_sweep_summary(all_results)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.output_dir / f"delta_sweep_{args.dataset}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults written to {output_file}")
    return all_results


def _print_delta_sweep_summary(results: dict):
    """Print a summary table of delta sweep results."""
    print(f"\n{'=' * 80}")
    print("Delta Sweep Summary")
    print(f"{'=' * 80}")
    print(
        f"\n{'Delta':<10} {'CV Acc':<14} {'CV F1':<14} {'HO Acc':<10} {'HO F1':<10} {'Justified':<10}"
    )
    print("-" * 80)
    best_cv_f1 = None
    best_delta = None
    for exp in results["experiments"]:
        delta = exp["delta"]
        res = exp.get("results")
        if res is None:
            print(f"{delta:<10} {'ERROR':<14}")
            continue
        cv = res.get("cross_validation", {}).get("afcba", {}).get("AF-CBA", {})
        ho = res.get("holdout_evaluation", {}).get("afcba", {}).get("AF-CBA", {})
        cv_acc = cv.get("accuracy", "--")
        cv_f1 = cv.get("f1_score", "--")
        ho_acc = ho.get("accuracy", "--")
        ho_f1 = ho.get("f1_score", "--")
        justified = ho.get("justified_fraction", "--")
        if isinstance(ho_acc, (int, float)) and isinstance(ho_f1, (int, float)):
            print(
                f"{delta:<10} {cv_acc:<14} {cv_f1:<14} "
                f"{ho_acc:<10.2f} {ho_f1:<10.2f} {justified:<10.2f}"
            )
        else:
            print(
                f"{delta:<10} {cv_acc:<14} {cv_f1:<14} "
                f"{ho_acc!s:<10} {ho_f1!s:<10} {justified!s:<10}"
            )
        # Track best CV F1 (parse from "mean±std" format)
        if isinstance(cv_f1, str) and "±" in cv_f1:
            mean_f1 = float(cv_f1.split("±")[0])
            if best_cv_f1 is None or mean_f1 > best_cv_f1:
                best_cv_f1 = mean_f1
                best_delta = delta
    if best_delta is not None:
        print(f"\nBest delta by CV F1: {best_delta} (F1 = {best_cv_f1:.2f})")


def run_consistency_analysis(args):
    runtime_config = build_runtime_config(args)
    processors = [get_processor(d, runtime_config) for d in DATASETS]
    results = analyse_dataset_consistency(processors)
    print(f"\n{'=' * 80}")
    print("Consistency Analysis Summary")
    print(f"{'=' * 80}")
    for dataset_name, data in results.items():
        total = data["total_cases"]
        to_remove = data["cases_to_remove"]
        print(f"\n{dataset_name}:")
        print(f"  Total cases: {total:,}")
        print(f"  Consistency: {data['consistency_percentage']:.2f}%")
        print(f"  Cases to remove: {to_remove} ({to_remove / total * 100:.1f}%)")
        print(f"  Class balance: {data['class_balance_ratio']:.2f}")
    return results


def run_demo(args):
    runtime_config = build_runtime_config(args)
    processor = get_processor(args.dataset, runtime_config)
    if processor is None:
        print(f"Unknown dataset: {args.dataset}")
        return
    X, y = processor.load_and_preprocess()
    case_base = create_case_base_from_data(X, y, random_state=args.random_state)
    focus_case = X.iloc[args.case]
    true_outcome = y.iloc[args.case]
    print(f"\n{'=' * 60}")
    print(f"Justification Demo: {processor.get_dataset_name()}")
    print(f"{'=' * 60}")
    print(f"Case index: {args.case}")
    print(f"True outcome: {true_outcome}")
    print("Focus case features:")
    for col, val in focus_case.items():
        print(f"  {col}: {val}")
    classifier = AFCBAClassifier(case_base, verbose=True)
    outcomes = (
        [args.outcome]
        if args.outcome is not None
        else list(case_base.cases[case_base.target_column].unique())
    )
    print(f"\nTrying to justify outcome(s): {outcomes}")
    for outcome in outcomes:
        justification = classifier.find_justification(focus_case, outcome)
        if justification:
            print(f"\n? Found justification for outcome {outcome}:")
            print(f"Winning strategy ({len(justification)} moves):")
            for i, move in enumerate(justification, 1):
                print(f"  {i}. {move.description()}")
        else:
            print(f"\n? Could not justify outcome {outcome}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    experiment_parent = argparse.ArgumentParser(add_help=False)
    experiment_parent.add_argument("--folds", type=int, default=5)
    experiment_parent.add_argument("--test-size", type=float, default=0.2)
    experiment_parent.add_argument("--random-state", type=int, default=42)
    experiment_parent.add_argument("--n-jobs", type=int, default=-1)
    experiment_parent.add_argument("--timeout", type=float, default=120)
    experiment_parent.add_argument("--max-moves", type=int, default=None)
    experiment_parent.add_argument(
        "--heuristic",
        choices=["majority", "nearest_neighbour"],
        default="majority",
    )
    experiment_parent.add_argument("--heuristic-k", type=int, default=3)
    experiment_parent.add_argument(
        "--no-one-hot",
        action="store_true",
        help="Disable one-hot encoding of categorical features",
    )
    experiment_parent.add_argument(
        "--no-standardise",
        action="store_true",
        help="Disable standardisation of numerical features",
    )
    experiment_parent.add_argument(
        "--sample-size",
        type=int,
        default=3500,
        help="Limit dataset to this many samples",
    )
    experiment_parent.add_argument(
        "--make-consistent",
        action="store_true",
        help="Enforce case base consistency",
    )
    experiment_parent.add_argument(
        "--undersampling",
        action="store_true",
        help="Enable undersampling of majority class",
    )
    experiment_parent.add_argument(
        "--undersampling-ratio",
        type=float,
        default=2.0,
        help="Target ratio of majority to minority class (default: 2.0)",
    )
    experiment_parent.add_argument(
        "--authoritativeness",
        action="store_true",
        default=True,
        help="Enable authoritativeness-based precedent selection",
    )
    experiment_parent.add_argument(
        "--auth-method",
        type=str,
        default="harmonic_1",
        help="Authoritativeness method: relative, absolute, product, harmonic_X",
    )
    experiment_parent.add_argument(
        "--conditional",
        action="store_true",
        help="Use conditional preferences (vs global SHAP-based)",
    )
    experiment_parent.add_argument(
        "--delta",
        type=float,
        default=2,
        help="Delta threshold for preferences",
    )
    experiment_parent.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Minimum support for conditional preference rules",
    )
    experiment_parent.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits for RIPPER rule extraction",
    )
    parser = argparse.ArgumentParser(
        description="AF-CBA: A Fortiori Case-Based Argumentation Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/main.py experiment --dataset admission --visualise
  python src/main.py experiment --dataset gtd --visualise
  python src/main.py experiment --dataset churn --undersampling --conditional
  python src/main.py experiment --dataset compas --authoritativeness --auth-method harmonic_1
  python src/main.py sweep --datasets admission compas gtd churn
  python src/main.py sweep --datasets gtd --dry-run
  python src/main.py sweep --datasets admission --complexity
  python src/main.py delta-sweep --dataset admission
  python src/main.py delta-sweep --dataset churn --delta-values 0.0 0.1 0.2 0.3
  python src/main.py demo --dataset admission --case 5
  """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    exp_parser = subparsers.add_parser(
        "experiment",
        parents=[experiment_parent],
        help="Run experiment on a single dataset",
    )
    exp_parser.add_argument("--dataset", required=True, choices=DATASETS)
    exp_parser.add_argument("--no-parallel", action="store_true")
    exp_parser.add_argument("--sequential-inner", action="store_true")
    exp_parser.add_argument(
        "--complexity",
        action="store_true",
        help="Evaluate structural complexity metrics",
    )
    exp_parser.add_argument(
        "--visualise",
        action="store_true",
        help="Generate strategy visualisations",
    )
    sweep_parser = subparsers.add_parser(
        "sweep",
        parents=[experiment_parent],
        help="Run experiments across configuration combinations",
    )
    sweep_parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, choices=DATASETS
    )
    sweep_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    sweep_parser.add_argument("--dry-run", action="store_true")
    sweep_parser.add_argument(
        "--complexity",
        action="store_true",
        help="Evaluate structural complexity metrics",
    )
    delta_parser = subparsers.add_parser(
        "delta-sweep",
        parents=[experiment_parent],
        help="Run experiments across multiple delta values",
    )
    delta_parser.add_argument("--dataset", required=True, choices=DATASETS)
    delta_parser.add_argument(
        "--delta-values",
        nargs="+",
        type=float,
        default=None,
        help="Delta values to test (default: 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)",
    )
    delta_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    delta_parser.add_argument("--dry-run", action="store_true")
    demo_parser = subparsers.add_parser("demo", help="Demonstrate justification")
    demo_parser.add_argument("--dataset", required=True, choices=DATASETS)
    demo_parser.add_argument("--case", type=int, default=0)
    demo_parser.add_argument("--outcome", type=int)
    demo_parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    start_time = time.time()
    try:
        if args.command == "experiment":
            run_single_experiment(args)
        elif args.command == "sweep":
            run_sweep(args)
        elif args.command == "delta-sweep":
            run_delta_sweep(args)
        elif args.command == "consistency":
            run_consistency_analysis(args)
        elif args.command == "demo":
            run_demo(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        raise
    print(f"\nCompleted in {time.time() - start_time:.1f} seconds")


def run_sweep(args) -> dict:
    configurations = list(
        product(
            [False],  # undersampling
            [True, False],  # conditional
            [True],  # authoritative
        )
    )
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "folds": args.folds,
        "heuristic": args.heuristic,
        "heuristic_k": (
            args.heuristic_k if args.heuristic == "nearest_neighbour" else None
        ),
        "evaluate_complexity": getattr(args, "complexity", False),
        "experiments": [],
    }
    for dataset in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'=' * 60}")
        for undersampling, conditional, authoritativeness in configurations:
            label = (
                f"{'Under, ' if undersampling else ''}"
                f"{'Cond' if conditional else 'Glob'}, "
                f"{'Auth' if authoritativeness else 'Default'}"
            )
            print(f"\n[{label}]")
            config_info = {
                "dataset": dataset,
                "undersampling": undersampling,
                "conditional": conditional,
                "authoritativeness": authoritativeness,
            }
            if args.dry_run:
                print(f"  Would run: {dataset} with {label}")
                all_results["experiments"].append({**config_info, "results": None})
                continue
            try:
                runtime_config = RuntimeConfig(
                    one_hot_categoricals=not getattr(args, "no_one_hot", False),
                    standardise_numerical=not getattr(args, "no_standardise", False),
                    sample_size=getattr(args, "sample_size", None),
                    make_consistent=getattr(args, "make_consistent", False),
                    undersampling=undersampling,
                    undersampling_ratio=getattr(args, "undersampling_ratio", 2.0),
                    authoritativeness=authoritativeness,
                    auth_method=getattr(args, "auth_method", "harmonic_1"),
                    conditional=conditional,
                    delta=getattr(args, "delta", 0.1),
                    min_support=getattr(args, "min_support", 10),
                    n_splits=getattr(args, "n_splits", 5),
                    random_state=args.random_state,
                    test_size=args.test_size,
                )
                processor = get_processor(dataset, runtime_config)
                experiment = Experiment(
                    processor,
                    runtime_config=runtime_config,
                    heuristic=args.heuristic,
                    timeout_seconds=args.timeout,
                    max_moves=args.max_moves,
                    **build_heuristic_kwargs(args),
                )
                results = experiment.run_full_experiment(
                    cv_folds=args.folds,
                    n_jobs=args.n_jobs,
                    evaluate_complexity=getattr(args, "complexity", False),
                )
                experiment.results = results
                experiment.export_results()
                all_results["experiments"].append({**config_info, "results": results})
            except Exception as e:
                print(f"  Experiment failed: {e}")
                all_results["experiments"].append(
                    {**config_info, "results": None, "error": str(e)}
                )
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.output_dir / f"sweep_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSweep results written to {output_file}")
        latex_output = generate_latex_tables(all_results)
        latex_file = args.output_dir / f"sweep_{timestamp}.tex"
        with open(latex_file, "w") as f:
            f.write(latex_output)
        print(f"LaTeX tables written to {latex_file}")
    return all_results


if __name__ == "__main__":
    if len(sys.argv) == 1:
        default_args = [
            "sweep",
            "--datasets",
            "admission",
            "churn",
            "gtd",
            "compas",
            "--complexity",
        ]
        main(default_args)
    else:
        main()
# python src/main.py experiment --dataset admission --conditional --visualise --complexity
