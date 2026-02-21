import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocess:
    def __init__(self, datasource: dict, dataset_info: dict, random_state: int = 42):
        self.datasource = datasource
        self.dataset_info = dataset_info
        self.random_state = random_state
        self.dataset_path = datasource["data_path"]["dataset_path"]
        self.preprocessor_filename = datasource["data_path"]["preprocessor_filename"]
        self.split_ratio = dataset_info["split_ratio"]
        self.target_column = dataset_info["features"]["target"]

        self.df = None

        self.cat_x, self.num_x, self.y = None, None, None

        self.preprocessor = None

        self.x_train, self.x_test, self.x_val = None, None, None
        self.y_train, self.y_test, self.y_val = None, None, None

    def run(self):
        self.load()
        self.split_columns()
        self.intputting()
        self.preprocessor_setup()
        self.join_cat_num()
        self.split()
        self.preprocess_fit()
        self.preprocess_transform()
        self.rebuild_dataframes()

    def run_and_save(self):
        self.run()
        self.save_preprocessor()

    def load(self):
        # load dataframe from disk
        df = pd.read_csv(self.dataset_path)
        # normalize column names (trim whitespace / invisible chars) to avoid KeyError
        try:
            df.columns = df.columns.str.strip()
        except Exception:
            pass
        self.df = df

    def split_columns(self):
        """Splits the dataframe into categorical, numerical, and target columns based on the specified target column name.

        - `categorical_df`: categorical columns (dtype != number)
        - `numerical_df`: numeric columns (dtype == number)
        - `target`: target column (dtype can be either number or non-number)
        """
        target_column = self.target_column
        df = self.df
        if target_column not in df.columns:
            raise KeyError(
                f"Target column '{target_column}' not found in dataframe columns. Available columns: {list(df.columns)}"
            )

        # target
        target = df[target_column].copy()
        target = target.astype(str).str.strip().str.lower()
        mapping = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
        target = target.map(mapping)

        # on split_columns, after .map(...)
        mask = target.notna()
        if not mask.all():
            print(
                f"Deleting {(~mask).sum()} rows with invalid target : {df.loc[~mask, target_column].unique()}"
            )
            df = df.loc[mask].copy()
            target = target[mask].copy()

        # x
        df = df.drop(columns=target_column)
        numerical_df = df.select_dtypes(include=["number"]).copy()
        categorical_df = df.select_dtypes(exclude=["number"]).copy()

        self.cat_df = categorical_df
        self.num_df = numerical_df
        self.y = target

    def intputting(self):
        self.cat_df = self.cat_df.fillna("unknown")
        self.num_df = self.num_df.fillna(self.num_df.median())

    def preprocessor_setup(self):
        cat_df = self.cat_df
        num_df = self.num_df
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_df.columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_df.columns),
            ]
        )

    def join_cat_num(self):
        self.df = self.cat_df.join(self.num_df)

    def split(self):
        train_size = 1 - self.split_ratio
        test_size = self.split_ratio / 2
        val_size = self.split_ratio / 2
        df = self.df
        y = self.y

        x_train, x_temp, y_train, y_temp = train_test_split(
            df, y, train_size=train_size, random_state=self.random_state, stratify=y
        )
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
        self.preprocessor.fit(self.x_train)

    def preprocess_transform(self):
        self.x_train = self.preprocessor.transform(self.x_train)
        self.x_test = self.preprocessor.transform(self.x_test)
        self.x_val = self.preprocessor.transform(self.x_val)

    def rebuild_dataframes(self):
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
    x = Preprocess(data_source, dataset_info)
    x.run_and_save()
    # usage test for preprocessor python src/data/preprocess.py --test
