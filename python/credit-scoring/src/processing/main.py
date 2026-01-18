import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessing:
    def __init__(
        self, cache_file, path: str, test_size=0.15, val_size=0.15, random_state=42
    ):
        self.df = pd.read_csv(path)
        self.joblib_validation(cache_file)

    def joblib_validation(
        self,
        cache_file,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    ):
        if os.path.exists("./preprocess/" + cache_file):
            print(".joblib existente, se usara")
            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
            ) = joblib.load("./preprocess/" + cache_file)
        else:
            print(".joblib inexistente, se va a creara")
            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
            ) = self.process_data(test_size, val_size, random_state)
            joblib.dump(
                self.process_data(test_size, val_size, random_state),
                "./preprocess/" + cache_file,
            )

    def process_data(self, test_size, val_size, random_state) -> tuple:
        # Target
        self.y = self.df["Risk"].map({"good": 1, "bad": 0})

        # Features
        X = self.df.drop(columns="Risk")

        # NA handling
        X = X.replace("NA", np.nan)
        X["Saving accounts"] = X["Saving accounts"].fillna("none")
        X["Checking account"] = X["Checking account"].fillna("none")

        numerical_cols = ["Age", "Job", "Credit amount", "Duration"]
        categorical_cols = [
            "Sex",
            "Housing",
            "Saving accounts",
            "Checking account",
            "Purpose",
        ]

        # One-hot
        X_cat = pd.get_dummies(X[categorical_cols], drop_first=False)

        # Split ANTES de escalar
        X_train, X_temp, y_train, y_temp = train_test_split(
            X[numerical_cols],
            self.y,
            test_size=test_size + val_size,
            stratify=self.y,
            random_state=random_state,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=test_size / (test_size + val_size),
            stratify=y_temp,
            random_state=random_state,
        )

        # Scaler con train
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Reconstruir DataFrames
        X_train_num = pd.DataFrame(
            X_train_scaled, columns=numerical_cols, index=X_train.index
        )
        X_val_num = pd.DataFrame(
            X_val_scaled, columns=numerical_cols, index=X_val.index
        )
        X_test_num = pd.DataFrame(
            X_test_scaled, columns=numerical_cols, index=X_test.index
        )

        # Alinear categóricas
        X_cat = X_cat.loc[X_train.index.union(X_val.index).union(X_test.index)]

        X_train = pd.concat([X_train_num, X_cat.loc[X_train.index]], axis=1)
        X_val = pd.concat([X_val_num, X_cat.loc[X_val.index]], axis=1)
        X_test = pd.concat([X_test_num, X_cat.loc[X_test.index]], axis=1)

        y_train = y_train
        y_val = y_val
        y_test = y_test

        return (X_train, X_val, X_test, y_train, y_val, y_test)

    def get_train(self):
        return self.X_train, self.y_train

    def get_val(self):
        return self.X_val, self.y_val

    def get_test(self):
        return self.X_test, self.y_test
