import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch

# Dynamically resolve the absolute path to the project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.server.schemas import CreditScoringInput, CreditScoringOutPut
from src.train.training import creditScoringModel

# --- CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = "cpu"
NUM_FEATURES = 26
RISK_THRESHOLD = 0.5


class CreditScoringRiskPredictor:
    """
    Service class responsible for loading machine learning artifacts
    (model weights and preprocessor) and executing inference for credit scoring.
    """

    def __init__(
        self, model_path: Path, preprocessor_path: Path, model_config: Dict[str, Any]
    ):
        """
        Initializes the predictor, storing paths and config, and loads artifacts.
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model_config = model_config
        self.preprocessor_data = None
        self.model = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Loads the joblib preprocessor dictionary and the PyTorch model weights.
        Raises an exception if files are missing or corrupted.
        """
        logger.info(f"Loading preprocessor from: {self.preprocessor_path}")
        try:
            self.preprocessor_data = joblib.load(self.preprocessor_path)
            logger.info("Preprocessor loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Preprocessor file not found at: {self.preprocessor_path}")
            raise

        logger.info(f"Loading PyTorch model weights from: {self.model_path}")
        try:
            # Recreate model architecture
            self.model = creditScoringModel(
                num_features=NUM_FEATURES,
                hidden_layers=self.model_config["hidden_layers"],
                dropout_rate=self.model_config["dropout_rate"],
                use_batch_norm=self.model_config["use_batch_norm"],
                activation_funct=self.model_config["activation_fn"],
            )

            # Load weights
            self.model.load_state_dict(
                torch.load(
                    self.model_path,
                    map_location=torch.device(DEVICE),
                    weights_only=True,
                )
            )
            self.model.eval()
            logger.info("Model weights loaded and set to evaluation mode successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found at: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading the model architecture or weights: {e}")
            raise

    def _preprocess_for_inference(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Applies the training preprocessing logic (handling NaNs, scaling, OHE)
        to the raw input dataframe.

        Args:
            df (pd.DataFrame): Raw input dataframe.

        Returns:
            torch.Tensor: Float32 tensor ready for network inference.
        """
        # Extract components from the loaded joblib dictionary
        scaler = self.preprocessor_data["scaler"]
        feature_columns = self.preprocessor_data["feature_columns"]

        numerical_cols = ["Age", "Job", "Credit amount", "Duration"]
        categorical_cols = [
            "Sex",
            "Housing",
            "Saving accounts",
            "Checking account",
            "Purpose",
        ]

        # --- 1. Clean Missing Values (NA Handling) ---
        df = df.replace("NA", np.nan)
        df["Saving accounts"] = df["Saving accounts"].fillna("none")
        df["Checking account"] = df["Checking account"].fillna("none")

        # --- 2. Numerical Features (Scaling) ---
        # Transform returns a numpy array, we convert it back to a DataFrame
        X_num_array = scaler.transform(df[numerical_cols])
        X_num = pd.DataFrame(X_num_array, columns=numerical_cols, index=df.index)

        # --- 3. Categorical Features (One-Hot Encoding) ---
        X_cat = pd.get_dummies(df[categorical_cols], drop_first=False)
        X_cat = X_cat.astype(float)  # Ensure floats for PyTorch

        # --- 4. Alignment & Concatenation ---
        # Concat first, then reindex to match the EXACT training columns
        X_combined = pd.concat([X_num, X_cat], axis=1)

        # This reindex forces the dataframe to have the exact columns as the training set.
        # Missing dummy columns (e.g., if a category wasn't in the single inference request) are filled with 0.
        X_final = X_combined.reindex(columns=feature_columns, fill_value=0.0)

        # --- 5. Convert to Tensor ---
        return torch.tensor(X_final.values, dtype=torch.float32)

    def predict(self, input_data: CreditScoringInput) -> CreditScoringOutPut:
        """
        Executes the full inference pipeline for credit scoring.
        """
        # 1. Convert Pydantic schema to DataFrame
        input_dict = input_data.model_dump(by_alias=True, mode="json")
        input_df = pd.DataFrame([input_dict])

        # 2. Preprocess data into a tensor
        input_tensor = self._preprocess_for_inference(input_df)

        # 3. Neural Network Forward Pass
        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()

        # 4. Format Output
        prediction = "good" if probability >= RISK_THRESHOLD else "bad"

        logger.info(
            f"Prediction generated: {prediction} | Probability: {probability:.4f}"
        )

        return CreditScoringOutPut(prediction=prediction, probability=probability)


if __name__ == "__main__":
    MODEL_CONFIG = {
        "hidden_layers": [128, 64],
        "use_batch_norm": True,
        "activation_fn": "ReLU",
        "dropout_rate": 0.15,
    }

    # Use pathlib for dynamic resolution
    MODEL_PATH = ROOT_DIR / "models" / "mlp_service_Credit_scoring_model_v001d.pt"
    PREPROCESSOR_PATH = ROOT_DIR / "preprocess" / "german_credit_preprocessor.joblib"

    # Instantiate ONLY when running directly
    predictor_instance = CreditScoringRiskPredictor(
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        model_config=MODEL_CONFIG,
    )

    # Test Sample
    sample_data = {
        "Age": 20,
        "Sex": "female",
        "Job": 0,
        "Housing": "rent",
        "Saving accounts": "NA",
        "Checking account": "NA",
        "Credit amount": 10000,
        "Duration": 13,
        "Purpose": "car",
    }

    sample_input = CreditScoringInput(**sample_data)
    result = predictor_instance.predict(sample_input)
    print("\n--- INFERENCE RESULT ---")
    print(result.model_dump(by_alias=True, mode="json"))
