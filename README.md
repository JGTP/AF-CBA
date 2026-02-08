# AF-CBA

A Fortiori Case-Based Argumentation as an explainable classifier.

## Getting started (Windows)
```
python -m venv .venv
source .venv/Scripts/activate 
python -m pip install --upgrade pip
pip install -e .
```

## Getting started (UNIX)
```
python -m venv .venv
source .venv/bin/activate 
pip install --upgrade pip
pip install -e .
```

Using ```pip install -e ".[dev]"``` installs Ruff as well.

## Usage

### Single experiment
```
python src/main.py experiment --dataset admission
python src/main.py experiment --dataset churn --undersampling --conditional
python src/main.py experiment --dataset compas --authoritativeness --auth-method harmonic_1 --complexity
```

### Sweep
Run experiments across multiple configuration combinations:
```
python src/main.py sweep --datasets admission
python src/main.py sweep --datasets admission churn compas gtd
python src/main.py sweep --datasets gtd --dry-run
```

The sweep command outputs two files to the results directory:
- `sweep_TIMESTAMP.json` — full experiment results
- `sweep_TIMESTAMP.tex` — LaTeX tables for inclusion in papers

The LaTeX output contains two separate tables:
- `tab:original-results` — results on the original (non-undersampled) datasets
- `tab:undersampled-results` — results on undersampled datasets

### Demo
```
python src/main.py demo --dataset admission --case 5
```

## Arguments

### Shared arguments (experiment and sweep)

| Argument | Default | Description |
|----------|---------|-------------|
| `--folds` | 5 | Number of cross-validation folds |
| `--test-size` | 0.2 | Fraction of data for holdout set |
| `--random-state` | 42 | Random seed |
| `--n-jobs` | -1 | Number of parallel jobs (-1 for all cores) |
| `--timeout` | None | Timeout in seconds per game |
| `--max-moves` | None | Maximum moves per game |
| `--heuristic` | majority | Fallback heuristic: `majority` or `nearest_neighbour` |
| `--heuristic-k` | 3 | Number of neighbours for nearest_neighbour heuristic |

### Preprocessing

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-one-hot` | False | Disable one-hot encoding of categorical features |
| `--no-standardise` | False | Disable standardisation of numerical features |
| `--sample-size` | None | Limit dataset to this many samples |
| `--make-consistent` | False | Enforce case base consistency |
| `--undersampling` | False | Enable undersampling of majority class |
| `--undersampling-ratio` | 2.0 | Target ratio of majority to minority class |

### AF-CBA configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--authoritativeness` | True | Enable authoritativeness-based precedent selection |
| `--auth-method` | harmonic_1 | Authoritativeness method: `relative`, `absolute`, `product`, `harmonic_X` |
| `--conditional` | False | Use conditional preferences (vs global SHAP-based) |
| `--delta` | 0.1 | Delta threshold for conditional preferences |
| `--min-support` | 1 | Minimum support for conditional preference rules |
| `--n-splits` | 5 | Number of CV splits for RIPPER rule extraction |

### Experiment-only arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | Dataset: `admission`, `churn`, `compas`, `gtd` |
| `--no-parallel` | False | Disable parallel execution |
| `--sequential-inner` | False | Force sequential inner loop |
| `--complexity` | False | Evaluate structural complexity |
| `--visualise` | False | Generate strategy visualisations |

### Sweep-only arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | all | Datasets to include |
| `--output-dir` | results | Output directory |
| `--dry-run` | False | Show configurations without running |
| `--complexity` | False | Evaluate structural complexity metrics |

### Demo-only arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | Dataset: `admission`, `churn`, `compas`, `gtd` |
| `--case` | 0 | Case index to justify |
| `--outcome` | None | Specific outcome to justify (both if omitted) |
| `--random-state` | 42 | Random seed |