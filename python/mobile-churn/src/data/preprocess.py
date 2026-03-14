import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data.EDA import perform_eda

def check_unique_values(df):
    """
    Iterates through all columns in the dataframe and prints unique values.
    Helpful for identifying dirty data before ColumnTransformation.
    """
    print(f"{'COLUMN':<25} | {'TYPE':<10} | {'UNIQUE VALUES / INFO'}")
    print("-" * 80)

    for col in df.columns:
        col_type = str(df[col].dtype)
        unique_vals = df[col].unique()
        n_unique = len(unique_vals)

        # If it's a feature with many numbers, we just show the range and a few samples
        if n_unique > 10 and col_type != "object":
            # sample = sorted([x for x in unique_vals if pd.notna(x)])[:3]
            print(
                f"{col:<25} | {col_type:<10} | {n_unique} unique values (Min: {df[col].min()}, Max: {df[col].max()})"
            )

        # If it's categorical or binary, we show all unique values
        else:
            print(f"{col:<25} | {col_type:<10} | {unique_vals}")


class Preprocess:
    """End-to-end preprocessing pipeline for machine learning datasets.

    Handles data loading, feature splitting, imputation, StandardScaler/OneHotEncoder,
    train/test/val split, and preprocessor serialization.

    Attributes:
        datasource (dict): Data source configuration with dataset path and preprocessor filename.
        dataset_info (dict): Dataset metadata including split ratios and target column name.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, datasource: dict, dataset_info: dict, random_state: int = 42,  show_unique_values: bool = False):
        self.datasource = datasource
        self.dataset_info = dataset_info
        self.random_state = random_state
        self.show_unique_values = show_unique_values
        self.dataset_path = datasource["data_path"]["dataset_path"]
        self.preprocessor_filename = datasource["data_path"]["preprocessor_filename"]
        self.split_ratio = dataset_info["split_ratio"]
        self.cat_cols = dataset_info["features"]["cat_columns"]
        self.num_cols = dataset_info["features"]["num_columns"]
        self.binary_cols = dataset_info["features"]["binary_columns"]
        self.target_column = dataset_info["features"]["target"]

        self.df: pd.DataFrame

        self.y = None, None, None

        self.preprocessor = None

        self.x_train, self.x_test, self.x_val = None, None, None
        self.y_train, self.y_test, self.y_val = None, None, None

    def run(self):
        """Execute the complete preprocessing pipeline in order."""
        self.load()
        self.split_columns()
        self.cleaning()
        self.preprocessor_setup()
        self.split()
        self.preprocess_fit()
        self.preprocess_transform()
        self.rebuild_dataframes()

    def run_and_save(self):
        """Execute the complete pipeline and save the fitted preprocessor to disk."""
        self.run()
        self.save_preprocessor()

    def load(self):
        """Load dataset from CSV and normalize column names (trim whitespace)."""
        # load dataframe from disk
        df = pd.read_csv(self.dataset_path)
        # normalize column names (trim whitespace / invisible chars) to avoid KeyError
        try:
            df.columns = df.columns.str.strip()
        except Exception:
            pass
        self.df = df

    def split_columns(self):
        """
        Extracts the target variable and isolates features from the raw dataframe.

        Normalizes the target column by mapping various truthy/falsy strings
        (e.g., 'yes', 'no', 'true', 'false') to binary integers (1, 0).
        Rows with unmappable or missing target values are dropped to ensure
        supervised learning integrity.

        Sets:
            self.y (pd.Series): The cleaned binary target vector.
            self.df (pd.DataFrame): The filtered dataframe containing only valid rows.
        """
        target_column = self.target_column
        df = self.df
        if target_column not in df.columns:
            raise KeyError(
                f"Target column '{target_column}' not found in dataframe columns. Available columns: {list(df.columns)}"
            )

        # target: normalize and map to binary
        target = df[target_column].copy()
        target = target.astype(str).str.strip().str.lower()
        mapping = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
        target = target.map(mapping)

        # remove rows with invalid target values (unmapped)
        mask = target.notna()
        if not mask.all():
            print(
                f"Deleting {(~mask).sum()} rows with invalid target : {df.loc[~mask, target_column].unique()}"
            )
            df = df.loc[mask].copy()
            target = target[mask].copy()
        self.df = df
        self.y = target

    def cleaning(self):
        """
        Performs type-specific data imputation and business logic corrections.

        Iterates through numerical, categorical, and binary column lists to:
        1. Fix business anomalies (e.g., replacing age 0 with median).
        2. Standardize categorical text and handle 'NA' strings as a new 'unknown' class.
        3. Force binary features into strict integer format (0/1).

        Finalizes the feature engineering process by consolidating all relevant
        features into a single matrix.

        Sets:
            self.X (pd.DataFrame): The final cleaned feature matrix ready for scaling/encoding.
        """
        print("Starting data cleaning process...")

        # 1. NUMERICAL FEATURES CLEANING
        if hasattr(self, "num_cols") and self.num_cols:

            # Logic Rule: Prices and Income should not be negative
            monetary_cols = [
                c
                for c in ["disposable_income", "current_mobile_price"]
                if c in self.num_cols
            ]
            for col in monetary_cols:
                self.df[col] = self.df[col].clip(lower=0)

            # Imputation: Fill NaNs with median for all numerical columns
            self.df[self.num_cols] = self.df[self.num_cols].fillna(
                self.df[self.num_cols].median()
            )

       

        # 3. BINARY FEATURES CLEANING
        if hasattr(self, "binary_cols") and self.binary_cols:
            # Map various truthy/falsy strings to strict 1/0 integers
            binary_mapping = {
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "t": 1,
                "f": 0,
                "1": 1,
                "0": 0,
                "1.0": 1,
                "0.0": 0,
            }

            for col in self.binary_cols:
                # Convert to string and normalize to ensure mapping works
                self.df[col] = (
                    self.df[col].astype(str).str.lower().str.strip().map(binary_mapping)
                )

                # Imputation: Fill missing binary values with the column mode (most frequent)
                col_mode = self.df[col].mode()
                if not col_mode.empty:
                    self.df[col] = self.df[col].fillna(col_mode.iloc[0])
                else:
                    # Fallback if the whole column is NaN
                    self.df[col] = self.df[col].fillna(0)

            # Ensure final type is integer for binary columns
            self.df[self.binary_cols] = self.df[self.binary_cols].astype(int)

        # 4. FINAL ASSEMBLY
        # Create self.X containing only the features cleaned (excluding target and ID)
        all_features = self.num_cols + self.binary_cols #+ self.cat_cols
        self.X = self.df[all_features].copy()
           
        print(
            f"Cleaning completed. Feature matrix 'self.X' ready. Shape: {self.X.shape}"
        )
        if self.show_unique_values:
            check_unique_values(self.X)

    def preprocessor_setup(self):
        """Initialize ColumnTransformer with StandardScaler for numerical and OneHotEncoder for categorical features."""
        cat_cols = self.cat_cols
        num_cols = self.num_cols
        binary_cols = self.binary_cols
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                #("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("bin", "passthrough", binary_cols),
            ]
        )

    def split(self):
        """Split data into train/test/val sets using stratified sampling by target."""
        train_size = 1 - self.split_ratio
        test_size = self.split_ratio / 2
        val_size = self.split_ratio / 2
        x = self.X
        y = self.y
        # first split: train vs temp (test+val)
        x_train, x_temp, y_train, y_temp = train_test_split(
            x, y, train_size=train_size, random_state=self.random_state, stratify=y
        )
        # second split: test vs val from temp set
        x_test, x_val, y_test, y_val = train_test_split(
            x_temp,
            y_temp,
            test_size=val_size / (test_size + val_size),
            random_state=self.random_state,
            stratify=y_temp,
        )
        print(f"Training set size: {len(x_train)}")
        print(f"Test set size: {len(x_test)}")
        print(f"Validation set size: {len(x_val)}")
        self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = (
            x_train,
            x_test,
            x_val,
            y_train,
            y_test,
            y_val,
        )

    def preprocess_fit(self):
        """Fit the preprocessor (StandardScaler, OneHotEncoder) on training data only."""
        if self.preprocessor is None:
            raise
        self.preprocessor.fit(self.x_train)

    def preprocess_transform(self):
        """Apply the fitted preprocessor to transform train/test/val sets."""
        if self.preprocessor is None:
            raise
        self.x_train = self.preprocessor.transform(self.x_train)
        self.x_test = self.preprocessor.transform(self.x_test)
        self.x_val = self.preprocessor.transform(self.x_val)

    def rebuild_dataframes(self):
        """Rebuild train/test/val sets as DataFrames with feature names from preprocessor output."""
        if self.preprocessor is None:
            raise
        self.x_train = pd.DataFrame(
            self.x_train, columns=self.preprocessor.get_feature_names_out()
        )
        self.x_test = pd.DataFrame(
            self.x_test, columns=self.preprocessor.get_feature_names_out()
        )
        self.x_val = pd.DataFrame(
            self.x_val, columns=self.preprocessor.get_feature_names_out()
        )
        print(
            f"Preprocessing completed. Training set: {self.x_train.shape}, Test set: {self.x_test.shape}, Validation set: {self.x_val.shape}"
        )

    def save_preprocessor(self):
        """Serialize fitted preprocessor to disk using joblib for later inference."""
        preprocessor = self.preprocessor
        preprocessor_filename = self.preprocessor_filename
        joblib.dump(
            preprocessor,
            "artifacts/preprocess/{preprocessor_filename}".format(
                preprocessor_filename=preprocessor_filename
            ),
        )
        print(
            "Preprocessor saved to artifacts/preprocess/{preprocessor_filename}".format(
                preprocessor_filename=preprocessor_filename
            )
        )


if __name__ == "__main__":
    """Entry point: load config and execute preprocessing pipeline with serialization."""
    # To execute this file in test mode, run from the project root(mobile-churn) with:
    # python src/data/preprocess.py --config 0
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from utils.config import Config, parse_args

    config_index = parse_args().config
    config = Config(config_index)
    data = config.data
    data_source = data["data_source"]
    dataset_info = data["dataset_info"]
    x = Preprocess(data_source, dataset_info, show_unique_values=True)
    x.load()
    x.split_columns()
    x.cleaning()
    perform_eda(x.df)
    x.preprocessor_setup()
    x.split()
    x.preprocess_fit()
    x.preprocess_transform()
    x.rebuild_dataframes()# success
    
