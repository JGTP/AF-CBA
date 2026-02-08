import json
import sys
from collections import defaultdict
from pathlib import Path


def fmt(x):
    if x is None:
        return "--"
    if isinstance(x, str):
        if x.startswith("0."):
            x = x[1:]
        return x.replace("±", "$\\pm$")
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0") else s


def fmt_int(x):
    if x is None:
        return "--"
    if isinstance(x, str):
        return x.replace("±", "$\\pm$")
    if isinstance(x, float):
        return str(int(round(x)))
    return str(x)


def get(results, *keys):
    for k in keys:
        if results is None or k not in results:
            return None
        results = results[k]
    return results


def format_row(
    model, cv, ho, note_cv=None, note_ho=None, note_is_int=False, note_str=None
):
    acc = f"{fmt(get(cv, 'accuracy'))} ({fmt(get(ho, 'accuracy'))})"
    f1 = f"{fmt(get(cv, 'f1_score'))} ({fmt(get(ho, 'f1_score'))})"
    mcc = f"{fmt(get(cv, 'mcc'))} ({fmt(get(ho, 'mcc'))})"
    if note_str is not None:
        note = note_str
    elif note_cv is not None:
        if note_is_int:
            note = (
                f"{fmt_int(note_cv)} ({fmt_int(note_ho)})"
                if note_ho is not None
                else fmt_int(note_cv)
            )
        else:
            note = (
                f"{fmt(note_cv)} ({fmt(note_ho)})"
                if note_ho is not None
                else fmt(note_cv)
            )
    else:
        note = "--"
    return (
        f"                         & {model:<26} "
        f"& {acc:<16} & {f1:<16} & {mcc:<16} & {note} \\\\"
    )


def format_afcba_note(cv, ho):
    """Format the Note column for AF-CBA with justified and both-justified fractions."""
    j_cv = get(cv, "justified_fraction")
    j_ho = get(ho, "justified_fraction")
    b_cv = get(cv, "both_justified_fraction")
    b_ho = get(ho, "both_justified_fraction")
    if j_cv is None and j_ho is None:
        return "--"
    parts = []
    if j_cv is not None or j_ho is not None:
        j_str = f"j:{fmt(j_cv)}" if j_cv is not None else "j:--"
        if j_ho is not None:
            j_str += f" ({fmt(j_ho)})"
        parts.append(j_str)
    if b_cv is not None or b_ho is not None:
        b_str = f"b:{fmt(b_cv)}" if b_cv is not None else "b:--"
        if b_ho is not None:
            b_str += f" ({fmt(b_ho)})"
        parts.append(b_str)
    return " ".join(parts) if parts else "--"


def generate_table(
    by_dataset: dict,
    caption: str,
    label: str,
) -> list[str]:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"        \centering")
    lines.append(f"        \\caption{{{caption}}}")
    lines.append(f"        \\label{{{label}}}")
    lines.append(r"        \begin{tabular}{llcccc}")
    lines.append(r"                \toprule")
    lines.append(
        r"                Data set & Model                    & Accuracy              & F1                    & MCC                   & Note \\"
    )
    lines.append(r"                \midrule")
    for dataset, exps in by_dataset.items():
        if not exps:
            continue
        label_text = dataset.upper()
        n_rows = 5
        lines.append(
            rf"                \multirow{{{n_rows}}}{{*}}{{\rotatebox{{90}}{{{label_text}}}}}"
        )
        base = exps[0]["results"]
        lines.append(
            format_row(
                "Decision Tree",
                get(base, "cross_validation", "sklearn", "DecisionTree"),
                get(base, "holdout_evaluation", "sklearn", "DecisionTree"),
            )
        )
        lines.append(
            format_row(
                "Random Forest",
                get(base, "cross_validation", "sklearn", "RandomForest"),
                get(base, "holdout_evaluation", "sklearn", "RandomForest"),
            )
        )
        ripper_cv = get(base, "cross_validation", "sklearn", "RIPPER")
        ripper_ho = get(base, "holdout_evaluation", "sklearn", "RIPPER")
        if ripper_cv is not None:
            ruleset_size_cv = get(ripper_cv, "ruleset_size")
            ruleset_size_ho = get(ripper_ho, "ruleset_size")
            lines.append(
                format_row(
                    "RIPPER",
                    ripper_cv,
                    ripper_ho,
                    ruleset_size_cv,
                    ruleset_size_ho,
                    note_is_int=True,
                )
            )
        else:
            lines.append(
                "                         & RIPPER                     "
                "& --               & --               & --               & -- \\\\"
            )
        cond = next(
            (e for e in exps if e["conditional"] and e["authoritativeness"]),
            None,
        )
        if cond:
            cv = get(cond, "results", "cross_validation", "afcba", "AF-CBA")
            ho = get(cond, "results", "holdout_evaluation", "afcba", "AF-CBA")
            note = format_afcba_note(cv, ho)
            lines.append(
                format_row(
                    "AF-CBA (Cond., Harmonic)",
                    cv,
                    ho,
                    note_str=note,
                )
            )
        else:
            lines.append(
                "                         & AF-CBA (Cond., Harmonic)   "
                "& --               & --               & --               & -- \\\\"
            )
        glob = next(
            (e for e in exps if not e["conditional"] and e["authoritativeness"]),
            None,
        )
        if glob:
            cv = get(glob, "results", "cross_validation", "afcba", "AF-CBA")
            ho = get(glob, "results", "holdout_evaluation", "afcba", "AF-CBA")
            note = format_afcba_note(cv, ho)
            lines.append(
                format_row(
                    "AF-CBA (Glob., Harmonic)",
                    cv,
                    ho,
                    note_str=note,
                )
            )
        else:
            lines.append(
                "                         & AF-CBA (Glob., Harmonic)   "
                "& --               & --               & --               & -- \\\\"
            )
        lines.append(r"                \midrule")
    if lines[-1] == r"                \midrule":
        lines[-1] = r"                \bottomrule"
    else:
        lines.append(r"                \bottomrule")
    lines.append(r"        \end{tabular}")
    lines.append(r"\end{table}")
    return lines


def generate_complexity_table(
    by_dataset: dict,
    caption: str,
    label: str,
) -> list[str]:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"        \centering")
    lines.append(f"        \\caption{{{caption}}}")
    lines.append(f"        \\label{{{label}}}")
    lines.append(r"        \begin{tabular}{llcccc}")
    lines.append(r"                \toprule")
    lines.append(
        r"                Data set & Model                    & Avg Depth & Max Depth & Avg Breadth & Max Breadth \\"
    )
    lines.append(r"                \midrule")
    for dataset, exps in by_dataset.items():
        if not exps:
            continue
        exps_with_complexity = [
            e
            for e in exps
            if get(e, "results", "holdout_evaluation", "complexity") is not None
        ]
        if not exps_with_complexity:
            continue
        label_text = dataset.upper()
        n_afcba_rows = len(exps_with_complexity)
        n_rows = 1 + n_afcba_rows
        lines.append(
            rf"                \multirow{{{n_rows}}}{{*}}{{\rotatebox{{90}}{{{label_text}}}}}"
        )
        dt_complexity = get(
            exps_with_complexity[0],
            "results",
            "holdout_evaluation",
            "complexity",
            "decision_tree",
        )
        if dt_complexity:
            lines.append(
                f"                         & {'Decision Tree':<26} "
                f"& {fmt(get(dt_complexity, 'mean_depth')):<9} "
                f"& {fmt_int(get(dt_complexity, 'max_depth')):<9} "
                f"& {fmt(get(dt_complexity, 'mean_breadth')):<11} "
                f"& {fmt_int(get(dt_complexity, 'max_breadth'))} \\\\"
            )
        for exp in exps_with_complexity:
            afcba_complexity = get(
                exp, "results", "holdout_evaluation", "complexity", "afcba"
            )
            if not afcba_complexity:
                continue
            cond_str = "Cond." if exp.get("conditional") else "Glob."
            auth_str = "Harmonic" if exp.get("authoritativeness") else "Default"
            model_name = f"AF-CBA ({cond_str}, {auth_str})"
            lines.append(
                f"                         & {model_name:<26} "
                f"& {fmt(get(afcba_complexity, 'mean_depth')):<9} "
                f"& {fmt_int(get(afcba_complexity, 'max_depth')):<9} "
                f"& {fmt(get(afcba_complexity, 'mean_breadth')):<11} "
                f"& {fmt_int(get(afcba_complexity, 'max_breadth'))} \\\\"
            )
        lines.append(r"                \midrule")
    if lines[-1] == r"                \midrule":
        lines[-1] = r"                \bottomrule"
    else:
        lines.append(r"                \bottomrule")
    lines.append(r"        \end{tabular}")
    lines.append(r"\end{table}")
    return lines


def _format_justified_row(model: str, cv: dict | None, ho: dict | None) -> str:
    if cv is None and ho is None:
        return (
            f"                         & {model:<26} "
            f"& --               & --               & -- \\\\"
        )
    acc = f"{fmt(get(cv, 'accuracy'))} ({fmt(get(ho, 'accuracy'))})"
    f1 = f"{fmt(get(cv, 'f1_score'))} ({fmt(get(ho, 'f1_score'))})"
    mcc = f"{fmt(get(cv, 'mcc'))} ({fmt(get(ho, 'mcc'))})"
    return f"                         & {model:<26} & {acc:<16} & {f1:<16} & {mcc} \\\\"


def generate_justified_only_table(
    by_dataset: dict,
    afcba_config: str,
    caption: str,
    label: str,
) -> list[str]:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"        \centering")
    lines.append(f"        \\caption{{{caption}}}")
    lines.append(f"        \\label{{{label}}}")
    lines.append(r"        \begin{tabular}{llccc}")
    lines.append(r"                \toprule")
    lines.append(
        r"                Data set & Model                    & Accuracy              & F1                    & MCC \\"
    )
    lines.append(r"                \midrule")
    for dataset, exps in by_dataset.items():
        if not exps:
            continue
        if afcba_config == "conditional":
            target_exp = next(
                (e for e in exps if e["conditional"] and e["authoritativeness"]),
                None,
            )
        else:
            target_exp = next(
                (e for e in exps if not e["conditional"] and e["authoritativeness"]),
                None,
            )
        if target_exp is None:
            continue
        label_text = dataset.upper()
        n_rows = 4
        lines.append(
            rf"                \multirow{{{n_rows}}}{{*}}{{\rotatebox{{90}}{{{label_text}}}}}"
        )
        cv_sklearn_just = get(
            target_exp, "results", "cross_validation", "sklearn_justified_only"
        )
        ho_sklearn_just = get(
            target_exp, "results", "holdout_evaluation", "sklearn_justified_only"
        )
        dt_cv = get(cv_sklearn_just, "DecisionTree") if cv_sklearn_just else None
        dt_ho = get(ho_sklearn_just, "DecisionTree") if ho_sklearn_just else None
        lines.append(_format_justified_row("Decision Tree", dt_cv, dt_ho))
        rf_cv = get(cv_sklearn_just, "RandomForest") if cv_sklearn_just else None
        rf_ho = get(ho_sklearn_just, "RandomForest") if ho_sklearn_just else None
        lines.append(_format_justified_row("Random Forest", rf_cv, rf_ho))
        rip_cv = get(cv_sklearn_just, "RIPPER") if cv_sklearn_just else None
        rip_ho = get(ho_sklearn_just, "RIPPER") if ho_sklearn_just else None
        lines.append(_format_justified_row("RIPPER", rip_cv, rip_ho))
        afcba_cv = get(target_exp, "results", "cross_validation", "afcba", "AF-CBA")
        afcba_ho = get(target_exp, "results", "holdout_evaluation", "afcba", "AF-CBA")
        afcba_cv_just = None
        if afcba_cv:
            afcba_cv_just = {
                "accuracy": get(afcba_cv, "justified_only_accuracy"),
                "f1_score": get(afcba_cv, "justified_only_f1_score"),
                "mcc": get(afcba_cv, "justified_only_mcc"),
            }
        afcba_ho_just = None
        if afcba_ho:
            afcba_ho_metrics = get(afcba_ho, "justified_only_metrics")
            if afcba_ho_metrics:
                afcba_ho_just = afcba_ho_metrics
        cond_str = "Cond." if afcba_config == "conditional" else "Glob."
        model_name = f"AF-CBA ({cond_str}, Harmonic)"
        lines.append(_format_justified_row(model_name, afcba_cv_just, afcba_ho_just))
        lines.append(r"                \midrule")
    if lines[-1] == r"                \midrule":
        lines[-1] = r"                \bottomrule"
    else:
        lines.append(r"                \bottomrule")
    lines.append(r"        \end{tabular}")
    lines.append(r"\end{table}")
    return lines


def generate_latex_tables(sweep_data: dict) -> str:
    experiments = sweep_data.get("experiments", [])
    normal_by_dataset = defaultdict(list)
    undersampled_by_dataset = defaultdict(list)
    for exp in experiments:
        if exp.get("results") is None:
            continue
        dataset = exp["dataset"]
        if exp.get("undersampling", False):
            undersampled_by_dataset[dataset].append(exp)
        else:
            normal_by_dataset[dataset].append(exp)
    output_lines = []
    if any(normal_by_dataset.values()):
        normal_table = generate_table(
            normal_by_dataset,
            caption=(
                "Classification performance on the original datasets comparing "
                "AF-CBA configurations with traditional machine learning models, "
                "using accuracy, macro F1-score, and MCC. "
                "The Note column shows ruleset size for RIPPER and justification fractions for AF-CBA "
                "(j=singly justified, b=both justified). "
                "Cross-validation results are reported as $\\mu\\pm\\sigma$; "
                "holdout performance is parenthesised."
            ),
            label="tab:original-results",
        )
        output_lines.extend(normal_table)
        output_lines.append("")
        for afcba_config, config_label in [("conditional", "cond"), ("global", "glob")]:
            config_name = "Conditional" if afcba_config == "conditional" else "Global"
            justified_table = generate_justified_only_table(
                normal_by_dataset,
                afcba_config=afcba_config,
                caption=(
                    f"Classification performance on singly justified cases only (original datasets), "
                    f"using the {config_name} AF-CBA configuration to determine justified cases. "
                    f"This provides a fairer comparison as all models are evaluated on the same "
                    f"subset of cases that AF-CBA could uniquely justify. "
                    f"Cross-validation results are reported as $\\mu\\pm\\sigma$; "
                    f"holdout performance is parenthesised."
                ),
                label=f"tab:original-justified-{config_label}",
            )
            output_lines.extend(justified_table)
            output_lines.append("")
    if any(undersampled_by_dataset.values()):
        undersampled_table = generate_table(
            undersampled_by_dataset,
            caption=(
                "Classification performance on undersampled datasets comparing "
                "AF-CBA configurations with traditional machine learning models, "
                "using accuracy, macro F1-score, and MCC. "
                "The Note column shows ruleset size for RIPPER and justification fractions for AF-CBA "
                "(j=singly justified, b=both justified). "
                "Cross-validation results are reported as $\\mu\\pm\\sigma$; "
                "holdout performance is parenthesised."
            ),
            label="tab:undersampled-results",
        )
        output_lines.extend(undersampled_table)
        output_lines.append("")
        for afcba_config, config_label in [("conditional", "cond"), ("global", "glob")]:
            config_name = "Conditional" if afcba_config == "conditional" else "Global"
            justified_table = generate_justified_only_table(
                undersampled_by_dataset,
                afcba_config=afcba_config,
                caption=(
                    f"Classification performance on singly justified cases only (undersampled datasets), "
                    f"using the {config_name} AF-CBA configuration to determine justified cases. "
                    f"This provides a fairer comparison as all models are evaluated on the same "
                    f"subset of cases that AF-CBA could uniquely justify. "
                    f"Cross-validation results are reported as $\\mu\\pm\\sigma$; "
                    f"holdout performance is parenthesised."
                ),
                label=f"tab:undersampled-justified-{config_label}",
            )
            output_lines.extend(justified_table)
            output_lines.append("")
    if sweep_data.get("evaluate_complexity", False):
        has_complexity = any(
            get(exp, "results", "holdout_evaluation", "complexity") is not None
            for exp in experiments
            if exp.get("results") is not None
        )
        if has_complexity:
            if any(normal_by_dataset.values()):
                normal_complexity_table = generate_complexity_table(
                    normal_by_dataset,
                    caption=(
                        "Structural complexity comparison between AF-CBA dispute trees "
                        "and decision trees on the original datasets. "
                        "Metrics include average and maximum depth and breadth."
                    ),
                    label="tab:original-complexity",
                )
                output_lines.extend(normal_complexity_table)
                output_lines.append("")
            if any(undersampled_by_dataset.values()):
                undersampled_complexity_table = generate_complexity_table(
                    undersampled_by_dataset,
                    caption=(
                        "Structural complexity comparison between AF-CBA dispute trees "
                        "and decision trees on the undersampled datasets. "
                        "Metrics include average and maximum depth and breadth."
                    ),
                    label="tab:undersampled-complexity",
                )
                output_lines.extend(undersampled_complexity_table)
    return "\n".join(output_lines)


def generate_from_file(path: Path) -> str:
    with open(path) as f:
        sweep_data = json.load(f)
    return generate_latex_tables(sweep_data)


def main(path: Path):
    output = generate_from_file(path)
    print(output)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python latex.py path/to/sweep.json", file=sys.stderr)
        sys.exit(1)
    main(Path(sys.argv[1]))
