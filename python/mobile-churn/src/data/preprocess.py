import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class preprocess():
    def __init__(self, datasource:dict, dataset_info:dict):
        self.datasource = datasource
        self.dataset_info = dataset_info
        self.df=pd.read_csv(self.datasource["data_path"]["dataset_path"])
        # normalize column names (trim whitespace / invisible chars) to avoid KeyError
        try:
            self.df.columns = self.df.columns.str.strip()
        except Exception:
            pass
        
        cat_x, num_x, self.y = self.split_columns(self.df, self.dataset_info["features"]["target"])
        cat_x, num_x = self.cleaning(cat_x, num_x)
        self.preprocessor = self.preprocessor_setup(cat_x, num_x)
        # TODO: save preprocessor to disk (e.g., using joblib) for later use in inference pipeline
    @staticmethod
    def cleaning(cat_df, num_df):
        cat_df = cat_df.fillna("unknown")
        num_df = num_df.fillna(num_df.mean())
        return cat_df, num_df
    
    @staticmethod
    def preprocessor_setup(cat_df:pd.DataFrame, num_df:pd.DataFrame):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_df.columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_df.columns),
            ]
        )
        return preprocessor.fit(cat_df.join(num_df))
    
    @staticmethod
    def split_columns(df:pd.DataFrame, target_column:str):
        """return (categorical_df, numerical_df).

        - `categorical_df`: categorical columns (dtype != number)
        - `numerical_df`: numeric columns (dtype == number)
        - `target`: target column (dtype can be either number or non-number)
        """
        if target_column not in df.columns:
            raise KeyError(
                f"Target column '{target_column}' not found in dataframe columns. Available columns: {list(df.columns)}"
            )
        
        target = df[target_column].copy()
        
        df = df.drop(columns=target_column)
        numerical_df = df.select_dtypes(include=["number"]).copy()
        categorical_df = df.select_dtypes(exclude=["number"]).copy()
        return categorical_df, numerical_df, target
        
if __name__ == "__main__":
    # To execute this file test mode, run:
    # from the root of the project:
    # python src/data/preprocess.py --test
    # Parser for command line arguments
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description="Run test.")
        parser.add_argument("--test", action="store_true", help="Run test.")
        return parser.parse_args()
    args = parse_args()
    # Run test mode if --test flag is provided
    if args.test:
        print("Running test mode.")
    # show the columns and unique values of the dataset (--test flag is required to show unique values)
    dict_test={
        "data_path": {
            "dataset_path": "./../../datasets/mobile-churn-customers.csv"
        }
    }
    dataset_info_test={
        "features": {
            "target": "customer_dropped"
        }
    }
    x = preprocess(dict_test, dataset_info_test)
    cat, num, target = x.split_columns(x.df,"customer_dropped")
    cat, num = x.cleaning(cat, num)
    print("Categorical columns (count={}):".format(len(cat.columns)), cat.columns.tolist())
    print("Numerical columns (count={}):".format(len(num.columns)), num.columns.tolist())
    if args.test:
        for col in cat.columns:
            print(f"Unique values in {col}: {cat[col].unique()}")
            print("value counts: ", cat[col].value_counts())
    print("\nCategorical sample:\n", cat.iloc[0:10, :10])
    print("\nNumerical sample:\n", num.iloc[0:10, :10])
    if args.test:
        for col in num.columns:
            print("{}: {} null values".format(col, num[col].isnull().sum()))  