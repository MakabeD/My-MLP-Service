from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.inference.predictor import CreditScoringRiskPredictor



class PredictorFactory:
    """Factory class to build and configure ML predictor"""

    @staticmethod
    def build():
        model_config = {
            "hidden_layers": [128, 64],
            "use_batch_norm": True,
            "activation_fn": "ReLU",
            "dropout_rate": 0.15,
        }
        model_path = ROOT_DIR / "models" / "mlp_service_Credit_scoring_model_v001d.pt"
        preprocessor_path = ROOT_DIR / "preprocess" / "german_credit_preprocessor.joblib"
        
        return CreditScoringRiskPredictor(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            model_config=model_config,
        )
        
if __name__=="__main__":
    x=PredictorFactory.build()