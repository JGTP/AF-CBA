from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


@dataclass
class RuntimeConfig:
    one_hot_categoricals: bool = True
    standardise_numerical: bool = True
    sample_size: int | None = None
    make_consistent: bool = False
    undersampling: bool = False
    undersampling_ratio: float = 2.0
    authoritativeness: bool = True
    auth_method: str = "harmonic_1"
    conditional: bool = False
    delta: float = 2.0
    min_support: int = 10
    n_splits: int = 5
    random_state: int = 42
    test_size: float = 0.2

    def get_preference_method(self) -> str:
        return "conditional" if self.conditional else "global"

    def to_case_base_config(self) -> dict:
        return {
            "authoritativeness": {
                "enabled": self.authoritativeness,
                "method": self.auth_method,
            },
            "conditional_preferences": {
                "method": self.get_preference_method(),
                "delta": self.delta,
                "min_support": self.min_support,
                "n_splits": self.n_splits,
            },
        }


@dataclass
class CaseBaseConfig:
    authoritativeness: dict = field(default_factory=dict)
    conditional_preferences: dict = field(default_factory=dict)

    @classmethod
    def from_runtime_config(cls, runtime_config: RuntimeConfig) -> "CaseBaseConfig":
        config_dict = runtime_config.to_case_base_config()
        return cls(
            authoritativeness=config_dict["authoritativeness"],
            conditional_preferences=config_dict["conditional_preferences"],
        )


class DatasetConfig:
    def __init__(self, config_dict: dict):
        self.target = config_dict["target"]
        self.features = config_dict.get("features", {})
        self.preprocessing = config_dict.get("preprocessing", {})
        for category in ["numerical", "categorical", "binary", "ordinal", "exclude"]:
            if category not in self.features:
                self.features[category] = []

    @classmethod
    def from_yaml(cls, config_path: Path) -> "DatasetConfig":
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)


class DataPreprocessor:
    def __init__(self, dataset_config: DatasetConfig, runtime_config: RuntimeConfig):
        self.dataset_config = dataset_config
        self.runtime_config = runtime_config
        self.encoders = {}

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        if self.dataset_config.target not in df.columns:
            raise ValueError(f"Target column '{self.dataset_config.target}' not found")
        df = self._apply_dataset_specific_preprocessing(df)
        df = self._handle_missing_values(df)
        df = self._select_features(df)
        y = df[self.dataset_config.target].copy()
        X = df.drop(columns=[self.dataset_config.target])
        X = self._transform_features(X)
        y, indices_to_drop = self._transform_target(y)
        if indices_to_drop:
            X = X.drop(index=indices_to_drop)
        if self.runtime_config.sample_size:
            X, y = self._sample_data(X, y)
        return X, y

    def _apply_dataset_specific_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_codes = self.dataset_config.preprocessing.get("missing_value_codes", {})
        if missing_codes:
            rows_to_drop = set()
            for code, columns in missing_codes.items():
                code_value = int(code) if code.lstrip("-").isdigit() else code
                for col in columns:
                    if col in df.columns:
                        missing_mask = df[col] == code_value
                        rows_to_drop.update(df[missing_mask].index)
            if rows_to_drop:
                df = df.drop(index=rows_to_drop)
                print(f"Removed {len(rows_to_drop)} rows with missing values")
        return df.fillna(0)

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_to_keep = set([self.dataset_config.target])
        for category in ["numerical", "categorical", "binary", "ordinal"]:
            features_to_keep.update(self.dataset_config.features.get(category, []))
        excluded = set(self.dataset_config.features.get("exclude", []))
        features_to_keep -= excluded
        columns_to_keep = [col for col in features_to_keep if col in df.columns]
        return df[columns_to_keep]

    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        categorical_features = self.dataset_config.features.get("categorical", [])
        if categorical_features and self.runtime_config.one_hot_categoricals:
            X = self._encode_categoricals(X, categorical_features)
        ordinal_features = self.dataset_config.features.get("ordinal", [])
        for feature in ordinal_features:
            if feature in X.columns:
                X[feature] = pd.to_numeric(X[feature], errors="coerce")
        return X

    def _encode_categoricals(
        self, X: pd.DataFrame, categorical_features: list[str]
    ) -> pd.DataFrame:
        for feature in categorical_features:
            if feature not in X.columns:
                continue
            encoder = OneHotEncoder(
                sparse_output=False, drop=None, handle_unknown="ignore"
            )
            encoded = encoder.fit_transform(X[[feature]])
            feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
            self.encoders[feature] = encoder
            X = pd.concat([X.drop(columns=[feature]), encoded_df], axis=1)
        return X

    def _transform_target(self, y: pd.Series) -> tuple[pd.Series, list]:
        indices_to_drop = []
        if pd.api.types.is_numeric_dtype(y):
            y = y.astype(int)
        unique_values = y.unique()
        if len(unique_values) > 2:
            valid_values = {0, 1}
            mask = y.isin(valid_values)
            indices_to_drop = y[~mask].index.tolist()
            y = y[mask]
            unique_values = y.unique()
            print(f"Dropped {len(indices_to_drop)} rows with invalid target values")
        if len(unique_values) == 2:
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            return y.map(mapping), indices_to_drop
        else:
            raise ValueError("No cases left after transforming target.")

    def _sample_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        sample_size = self.runtime_config.sample_size
        if sample_size >= len(X):
            return X, y
        X_sampled, _, y_sampled, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=self.runtime_config.random_state,
        )
        print(f"Stratified sample: {len(X)} → {len(X_sampled)}")
        return X_sampled, y_sampled


class DatasetProcessor:
    def __init__(
        self,
        config_path: Path,
        data_path: Path,
        runtime_config: RuntimeConfig | None = None,
    ):
        self.config_path = config_path
        self.data_path = data_path
        self.dataset_config = DatasetConfig.from_yaml(config_path)
        self.runtime_config = runtime_config or RuntimeConfig()
        self._cached_data = None

    def get_dataset_name(self) -> str:
        return self.config_path.stem

    def set_runtime_config(self, runtime_config: RuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self._cached_data = None

    def load_and_preprocess(self) -> tuple[pd.DataFrame, pd.Series]:
        if self._cached_data is None:
            df = self._load_raw_data()
            df = self._apply_specific_preprocessing(df)
            preprocessor = DataPreprocessor(self.dataset_config, self.runtime_config)
            X, y = preprocessor.fit_transform(df)
            self._cached_data = (X, y)
            print(
                f"Loaded {self.get_dataset_name()}: {len(X)} samples, "
                f"{len(X.columns)} features"
            )
        return self._cached_data

    def _load_raw_data(self) -> pd.DataFrame:
        if self.data_path.suffix.lower() == ".csv":
            return pd.read_csv(self.data_path)
        elif self.data_path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _apply_specific_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def prepare_for_experiment(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X, y = self.load_and_preprocess()
        return train_test_split(
            X,
            y,
            test_size=self.runtime_config.test_size,
            stratify=y,
            random_state=self.runtime_config.random_state,
        )


class AdmissionProcessor(DatasetProcessor):
    def __init__(self, runtime_config: RuntimeConfig | None = None):
        super().__init__(
            config_path=Path("config/admission.yaml"),
            data_path=Path("data/admission.csv"),
            runtime_config=runtime_config,
        )

    def _apply_specific_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Chance of Admit " in df.columns:
            df = df.copy()
            df["Chance of Admit "] = df["Chance of Admit "].round(0).astype(int)
        return df


class ChurnProcessor(DatasetProcessor):
    def __init__(self, runtime_config: RuntimeConfig | None = None):
        super().__init__(
            config_path=Path("config/churn.yaml"),
            data_path=Path("data/churn.csv"),
            runtime_config=runtime_config,
        )

    def _apply_specific_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
        df = df.dropna(subset=["TotalCharges"])
        df["TotalCharges"] = df["TotalCharges"].astype(float)
        binary_mappings = {
            "Partner": {"Yes": 1, "No": 0},
            "gender": {"Male": 1, "Female": 0},
            "Dependents": {"Yes": 1, "No": 0},
            "PhoneService": {"Yes": 1, "No": 0},
            "PaperlessBilling": {"Yes": 1, "No": 0},
            "Churn": {"Yes": 1, "No": 0},
        }
        for column, mapping in binary_mappings.items():
            if column in df.columns:
                df[column] = df[column].map(mapping)
        return df


class GTDProcessor(DatasetProcessor):
    def __init__(self, runtime_config: RuntimeConfig | None = None):
        super().__init__(
            config_path=Path("config/GTD.yaml"),
            data_path=Path("data/gtd.xlsx"),
            runtime_config=runtime_config,
        )


class COMPASProcessor(DatasetProcessor):
    def __init__(self, runtime_config: RuntimeConfig | None = None):
        super().__init__(
            config_path=Path("config/COMPAS.yaml"),
            data_path=Path("data/compas.csv"),
            runtime_config=runtime_config,
        )
