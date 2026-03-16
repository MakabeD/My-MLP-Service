import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.inference.predictor import TelecomChurnPredictor


class PredictorFactory:
    """Factory class to build and configure ML predictor"""

    @staticmethod
    def build():

        return TelecomChurnPredictor()


if __name__ == "__main__":
    x = PredictorFactory.build()
