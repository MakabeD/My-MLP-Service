import logging as log
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.server.schemas import CreditScoringInput
from src.train.training import creditScoringModel

device_ = "cpu"

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CreditScoringRiskPredictor:
    """
    Orchestrates the loading of artifacts and the execution of inference
    """

    def __init__(self, model_path, preprocessor_path, model_config):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model_config = model_config
        self.preprocessor = None
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self):
        """
        load the artifacts from the path
        """
        try:
            self.preprocessor = joblib.load(self.preprocessor_path)
            log.info(f"Archivo preprocesador cargado desde: {self.preprocessor_path}")
        except FileNotFoundError:
            log.error(
                f"No se encontro el archivo preprocesado en: {self.preprocessor_path}"
            )
            raise

        try:
            # recreate the architecture of the model
            self.model = creditScoringModel(
                num_features=26,
                hidden_layers=self.model_config["hidden_layers"],
                dropout_rate=self.model_config["dropout_rate"],
                use_batch_norm=self.model_config["use_batch_norm"],
                activation_funct=self.model_config["activation_fn"],
            )
            # load weights trained
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device(device_))
            )
            self.model.eval()
            log.info(f" Pesos del modelo cargados desde: {self.model_path}")
            log.info(" Modelo y preprocesador cargados exitosamente.")
        except FileNotFoundError:
            log.error(f"Archivo no encontrado en {self.model_path}")
            raise
        except Exception as e:
            log.error(f"Error al cargar el modelo: {e}")
            raise

    def predict(self, input_data: CreditScoringInput):
        """Make the prediction"""
        # pydantic to dataframe
        input_df = pd.DataFrame([input_data.model_dump(by_alias=True)])
        # dataframe to ---->>>>  preprocessing /// tensor
        input_tensor = self.preprocess_for_inference(input_df, self.preprocessor)

        # predition
        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()
            # 5. format outout
        prediction = "good" if probability >= 0.5 else "bad"
        log.info(
            f"Predicción generada: {prediction} con probabilidad: {probability:.4f}"
        )

        return {"prediction": prediction, "probability": probability}

    def preprocess_for_inference(
        self,
        df: pd.DataFrame,
        preprocessor,
    ) -> torch.Tensor:
        """
        Recibe un DataFrame crudo y un joblib de preprocessing.
        Devuelve un tensor listo para inferencia.
        """

        # -------------------------
        # Cargar preprocessing
        # -------------------------
        data = preprocessor

        scaler = data["scaler"]
        feature_columns = data["feature_columns"]

        numerical_cols = ["Age", "Job", "Credit amount", "Duration"]
        categorical_cols = [
            "Sex",
            "Housing",
            "Saving accounts",
            "Checking account",
            "Purpose",
        ]

        # -------------------------
        # Limpieza NA (MISMA lógica)
        # -------------------------
        df = df.replace("NA", np.nan)
        df["Saving accounts"] = df["Saving accounts"].fillna("none")
        df["Checking account"] = df["Checking account"].fillna("none")

        # -------------------------
        # Numéricas
        # -------------------------
        X_num = scaler.transform(df[numerical_cols])
        X_num = pd.DataFrame(X_num, columns=numerical_cols, index=df.index)

        # -------------------------
        # Categóricas
        # -------------------------
        X_cat = pd.get_dummies(df[categorical_cols], drop_first=False)

        # Alinear columnas EXACTAS del entrenamiento
        X_cat = X_cat.reindex(columns=feature_columns, fill_value=0)

        # Eliminar columnas numéricas duplicadas
        X_cat = X_cat.drop(columns=numerical_cols, errors="ignore")

        # -------------------------
        # Concatenar
        # -------------------------
        X_final = pd.concat([X_num, X_cat], axis=1)

        # Orden final (CRÍTICO)
        X_final = X_final[feature_columns]

        # -------------------------
        # A tensor
        # -------------------------
        X_tensor = torch.tensor(X_final.values, dtype=torch.float32)

        return X_tensor


MODEL_CONFIG = {
    "hidden_layers": [128, 64],
    "use_batch_norm": True,
    "activation_fn": "ReLU",
    "dropout_rate": 0.15,
}
MODEL_PATH = "./models/mlp_service_Credit_scoring_model_v001d.pt"
PREPROCESSOR_PATH = "./preprocess/german_credit_preprocessor.joblib"

predictor_instance = CreditScoringRiskPredictor(
    model_path=MODEL_PATH,
    preprocessor_path=PREPROCESSOR_PATH,
    model_config=MODEL_CONFIG,
)


if __name__ == "__main__":
    from src.server.schemas import *

    sample = CreditScoringInput(
        **{
            "Age": 30,
            "Sex": "male",
            "Job": 0,
            "Housing": "free",
            "Saving accounts": "NA",
            "Checking account": "NA",
            "Credit amount": 1000,
            "Duration": 1,
            "Purpose": "car",
        }
    )
    result = predictor_instance.predict(sample)
    print(result)
