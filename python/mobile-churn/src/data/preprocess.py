import joblib
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
        #save preprocessor to disk (e.g., using joblib) for later use in inference pipeline
        for i in datasource.items():
            print(f"{i[0]}: {i[1]}")
        joblib_name=datasource["data_path"]["preprocessor_filename"]
        joblib.dump(self.preprocessor, "artifacts/preprocess/{joblib_name}".format(joblib_name=joblib_name))
        #log
        print("Preprocessor saved to artifacts/preprocess/{joblib_name}".format(joblib_name=joblib_name))
        # TODO: return preprocessed data (X) for training and validation. And (y) for training and validation.
        
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
    # python src/data/preprocess.py 
    # Parser for command line arguments
    import os 
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from utils.config import Config, parse_args
    config_path = parse_args().config
    config=Config(config_path)
    data=config.data
    data_source=data["data_source"]
    dataset_info=data["dataset_info"]
    x = preprocess(data_source, dataset_info)
    #usage test for preprocessor python src/data/preprocess.py --test