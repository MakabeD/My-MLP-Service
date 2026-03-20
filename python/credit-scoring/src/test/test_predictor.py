import sys
from pathlib import Path

import pytest

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_PATH))
from src.inference.predictor import CreditScoringRiskPredictor
from src.inference.predictor_factory import PredictorFactory
from src.server.schemas import CreditScoringInput, CreditScoringOutPut

SAMPLE = CreditScoringInput(
    **{
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
)


@pytest.fixture(scope="session")
def instance():
    instance = PredictorFactory().build()
    assert isinstance(instance, CreditScoringRiskPredictor)
    return instance


def test_predictor(instance):
    pred = instance.predict(SAMPLE)
    assert isinstance(pred, CreditScoringOutPut)


def test_prediction_stability(instance):
    pred1 = instance.predict(SAMPLE)
    pred2 = instance.predict(SAMPLE)

    assert pred1 == pred2
