import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessing:
    def __init__(
        self,
        cache_file: str,
        path: str,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    ):
        self.df = pd.read_csv(path)
        self.cache_path = "./preprocess/" + cache_file

        self.joblib_validation(test_size, val_size, random_state)

    # --------------------------------------------------
    # JOBLIB
    # --------------------------------------------------
    def joblib_validation(self, test_size, val_size, random_state):
        if os.path.exists(self.cache_path):
            print(".joblib existente, se usará")

            data = joblib.load(self.cache_path)

            self.X_train = data["X_train"]
            self.X_val = data["X_val"]
            self.X_test = data["X_test"]
            self.y_train = data["y_train"]
            self.y_val = data["y_val"]
            self.y_test = data["y_test"]

            self.scaler = data["scaler"]
            self.feature_columns = data["feature_columns"]

        else:
            print(".joblib inexistente, se va a crear")

            self.process_data(test_size, val_size, random_state)

            joblib.dump(
                {
                    "X_train": self.X_train,
                    "X_val": self.X_val,
                    "X_test": self.X_test,
                    "y_train": self.y_train,
                    "y_val": self.y_val,
                    "y_test": self.y_test,
                    "scaler": self.scaler,
                    "feature_columns": self.feature_columns,
                },
                self.cache_path,
            )

    # --------------------------------------------------
    # DATA PROCESSING
    # --------------------------------------------------
    def process_data(self, test_size, val_size, random_state):
        # Target
        y = self.df["Risk"].map({"good": 1, "bad": 0})

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

        # One-hot completo (ANTES del split)
        X_cat = pd.get_dummies(X[categorical_cols], drop_first=False)

        # Split
        X_num = X[numerical_cols]

        X_train_num, X_temp_num, y_train, y_temp = train_test_split(
            X_num,
            y,
            test_size=test_size + val_size,
            stratify=y,
            random_state=random_state,
        )

        X_val_num, X_test_num, y_val, y_test = train_test_split(
            X_temp_num,
            y_temp,
            test_size=test_size / (test_size + val_size),
            stratify=y_temp,
            random_state=random_state,
        )

        # Fit scaler SOLO con train
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_num)
        X_val_scaled = self.scaler.transform(X_val_num)
        X_test_scaled = self.scaler.transform(X_test_num)

        # Reconstruir numéricos
        X_train_num = pd.DataFrame(
            X_train_scaled, columns=numerical_cols, index=X_train_num.index
        )
        X_val_num = pd.DataFrame(
            X_val_scaled, columns=numerical_cols, index=X_val_num.index
        )
        X_test_num = pd.DataFrame(
            X_test_scaled, columns=numerical_cols, index=X_test_num.index
        )

        # Concatenar categóricas
        X_train = pd.concat([X_train_num, X_cat.loc[X_train_num.index]], axis=1)
        X_val = pd.concat([X_val_num, X_cat.loc[X_val_num.index]], axis=1)
        X_test = pd.concat([X_test_num, X_cat.loc[X_test_num.index]], axis=1)

        # Guardar columnas finales (CRÍTICO para predicción)
        self.feature_columns = X_train.columns

        # Asegurar mismo orden
        X_val = X_val.reindex(columns=self.feature_columns)
        X_test = X_test.reindex(columns=self.feature_columns)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    # --------------------------------------------------
    # GETTERS (SIN CAMBIOS)
    # --------------------------------------------------
    def get_train(self):
        return self.X_train, self.y_train

    def get_val(self):
        return self.X_val, self.y_val

    def get_test(self):
        return self.X_test, self.y_test
