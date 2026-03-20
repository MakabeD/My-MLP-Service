import logging
from pathlib import Path

import joblib
import pandas as pd
import torch

# Dynamically resolve the absolute path to the project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from src.model.factory import ModelFactory
from src.server.schemas import ChurnInput, ChurnOutput
from utils.config import Config

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = Config(2)
NUM_FEATURES = 17
CHURN_THRESHOLD = 0.3

PREPROCESS_PATH = (
    ROOT_DIR / "artifacts" / "preprocess" / "teleco-churn_preprocessor.joblib"
)
MODEL_PATH = ROOT_DIR / "artifacts" / "models" / "mlp_service_telecom-churn_v3.pt"


class TelecomChurnPredictor:
    """
    Service class responsible for loading machine learning artifacts
    (model and preprocessor) and executing inference for telecom churn.
    """

    def __init__(self) -> None:
        """Initializes the predictor and loads necessary artifacts into memory."""
        self.model = None
        self.preprocessor = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Loads the scikit-learn preprocessor and the PyTorch model weights.
        Raises an exception if files are missing or corrupted.
        """
        logger.info(f"Loading preprocessor from {PREPROCESS_PATH}...")
        try:
            self.preprocessor = joblib.load(PREPROCESS_PATH)
        except FileNotFoundError:
            logger.error(f"Preprocessor artifact not found at {PREPROCESS_PATH}")
            raise

        logger.info(f"Loading PyTorch model from {MODEL_PATH}...")
        try:
            self.model = ModelFactory.build(NUM_FEATURES, CONFIG)
            self.model.load_state_dict(
                torch.load(
                    MODEL_PATH, map_location=torch.device("cpu"), weights_only=True
                )
            )
            self.model.eval()  # Set model to evaluation mode
            logger.info("Artifacts loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model artifact not found at {MODEL_PATH}")
            raise

    def __clean_binaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts boolean-like string values into strict 1s and 0s.

        Args:
            df (pd.DataFrame): Raw input dataframe.

        Returns:
            pd.DataFrame: Dataframe with binary columns mapped to integers.
        """
        mapping = {"Yes": 1, "No": 0, "True": 1, "False": 0}
        binary_columns = ["International_plan", "Voice_mail_plan"]

        for col in binary_columns:
            # map values, fill unexpected NaNs with 0, and ensure integer type
            df[col] = df[col].map(mapping).fillna(0).astype(int)

        return df

    def _preprocessing_for_inference(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Applies the loaded scikit-learn preprocessing pipeline to the dataframe
        and converts the output to a PyTorch tensor.

        Args:
            df (pd.DataFrame): Cleaned dataframe.

        Returns:
            torch.Tensor: Float32 tensor ready for the neural network.
        """
        transformed_data = self.preprocessor.transform(df)
        return torch.tensor(transformed_data, dtype=torch.float32)

    def predict(self, input_data: ChurnInput) -> ChurnOutput:
        """
        Executes the full inference pipeline: data dumping, cleaning,
        preprocessing, and neural network forward pass.

        Args:
            input_data (ChurnInput): Validated Pydantic schema from the API.

        Returns:
            ChurnOutput: Pydantic schema containing prediction and probability.
        """
        # 1. Extract data to a dictionary strictly as JSON/Primitives
        input_dict = input_data.model_dump(by_alias=True, mode="json")
        input_df = pd.DataFrame([input_dict])

        # 2. Clean business logic features
        input_df = self.__clean_binaries(input_df)

        # 3. Apply mathematical preprocessing (Scaling)
        input_tensor = self._preprocessing_for_inference(input_df)

        # 4. Neural Network Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()

            # Determine class based on threshold
            pred_label = "Churn" if probability > CHURN_THRESHOLD else "No Churn"

            logger.info(f"Prediction: {pred_label} | Probability: {probability:.4f}")

            return ChurnOutput(prediction=pred_label, probability=probability)


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Instantiate only when running as a script to prevent blocking imports
    predictor_instance = TelecomChurnPredictor()

    sample = {
        "Accountlength": 111,
        "International_plan": "No",
        "Voice_mail_plan": "No",
        "Number_vmail_messages": 0,
        "Total_day_minutes": 110.4,
        "Total_day_calls": 103,
        "Total_day_charge": 18.77,
        "Total_eve_minutes": 137.3,
        "Total_eve_calls": 102,
        "Total_eve_charge": 11.67,
        "Total_night_minutes": 189.6,
        "Total_night_calls": 105,
        "Total_night_charge": 8.53,
        "Total_intl_minutes": 7.7,
        "Total_intl_calls": 6,
        "Total_intl_charge": 2.08,
        "Customer_service_calls": 2,
    }

    # Simulate an API request payload
    pydantic_input = ChurnInput(**sample)

    # Run prediction
    result = predictor_instance.predict(pydantic_input)
    print("\n--- FINAL RESULT ---")
    print(result.model_dump_json(indent=2))
