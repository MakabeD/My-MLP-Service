import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class preprocess:
    def __init__(self, datasource: dict, dataset_info: dict, random_state: int = 42):
        self.datasource = datasource
        self.dataset_info = dataset_info
        self.df = pd.read_csv(self.datasource["data_path"]["dataset_path"])
        # normalize column names (trim whitespace / invisible chars) to avoid KeyError
        try:
            self.df.columns = self.df.columns.str.strip()
        except Exception:
            pass

        cat_x, num_x, self.y = self.split_columns(
            self.df, self.dataset_info["features"]["target"]
        )
        cat_x, num_x = self.cleaning(cat_x, num_x)
        self.preprocessor = self.preprocessor_setup(cat_x, num_x)
        self.df = cat_x.join(num_x)
        self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = self.split(self.df, self.y)

        # preprocessed data (X) for training and validation. And (y) for training and validation.
        
        self.x_train = self.preprocessor.fit_transform(self.x_train)
        self.x_test = self.preprocessor.transform(self.x_test)
        self.x_val = self.preprocessor.transform(self.x_val)
        # save preprocessor to disk (e.g., using joblib) for later use in inference pipeline
        joblib_name = datasource["data_path"]["preprocessor_filename"]
        joblib.dump(
            self.preprocessor,
            "artifacts/preprocess/{joblib_name}".format(joblib_name=joblib_name),
        )
        # log
        print(
            "Preprocessor saved to artifacts/preprocess/{joblib_name}".format(
                joblib_name=joblib_name
            )
        )
        # to dataframes
        self.x_train = pd.DataFrame(self.x_train, columns=self.preprocessor.get_feature_names_out())
        self.x_test = pd.DataFrame(self.x_test, columns=self.preprocessor.get_feature_names_out())
        self.x_val = pd.DataFrame(self.x_val, columns=self.preprocessor.get_feature_names_out())
        #final log
        print(f"Preprocessing completed. Training set: {self.x_train.shape}, Test set: {self.x_test.shape}, Validation set: {self.x_val.shape}")
    @staticmethod
    def split(
        df: pd.DataFrame,
        y: pd.Series,
        train_size: float = 0.8,
        test_size: float = 0.1,
        val_size: float = 0.1,
    ):
        print(y.isna().sum())
        x_train, x_temp, y_train, y_temp = train_test_split(
            df, y, train_size=train_size, random_state=42, stratify=y
        )
        x_test, x_val, y_test, y_val = train_test_split(
            x_temp, y_temp, test_size=val_size / (test_size + val_size), random_state=42, stratify=y_temp
        )
        print(f"Training set size: {len(x_train)}")
        print(f"Test set size: {len(x_test)}")
        print(f"Validation set size: {len(x_val)}")
        return x_train, x_test, x_val, y_train, y_test, y_val

    @staticmethod
    def cleaning(cat_df, num_df):
        cat_df = cat_df.fillna("unknown")
        return cat_df, num_df

    @staticmethod
    def preprocessor_setup(cat_df: pd.DataFrame, num_df: pd.DataFrame):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_df.columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_df.columns),
            ]
        )
        return preprocessor

    @staticmethod
    def split_columns(df: pd.DataFrame, target_column: str):
        """return (categorical_df, numerical_df).

        - `categorical_df`: categorical columns (dtype != number)
        - `numerical_df`: numeric columns (dtype == number)
        - `target`: target column (dtype can be either number or non-number)
        """
        if target_column not in df.columns:
            raise KeyError(
                f"Target column '{target_column}' not found in dataframe columns. Available columns: {list(df.columns)}"
            )
        # target
        target = df[target_column].copy()
        target = target.astype(str).str.strip().str.lower()
        mapping = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
        target = target.map(mapping)
        # en split_columns, después del .map(...)
        mask = target.notna()
        if not mask.all():
            print(f"Eliminando { (~mask).sum() } filas con target inválido: {df.loc[~mask, target_column].unique()}")
            df = df.loc[mask].copy()
            target = target[mask].copy()
        
        
        # x
        df = df.drop(columns=target_column)
        numerical_df = df.select_dtypes(include=["number"]).copy()
        categorical_df = df.select_dtypes(exclude=["number"]).copy()
        return categorical_df, numerical_df, target


if __name__ == "__main__":
    # To execute this file test mode, run:
    # from the root of the project:
    # python src/data/preprocess.py
    # Parser for command line arguments
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from utils.config import Config, parse_args

    config_index = parse_args().config
    config = Config(config_index)
    data = config.data
    data_source = data["data_source"]
    dataset_info = data["dataset_info"]
    x = preprocess(data_source, dataset_info)
    # usage test for preprocessor python src/data/preprocess.py --test
