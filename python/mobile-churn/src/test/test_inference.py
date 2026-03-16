import sys
from pathlib import Path

import pytest

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_PATH))
from src.inference.predictor import TelecomChurnPredictor
from src.inference.predictor_factory import PredictorFactory
from src.server.schemas import ChurnInput, ChurnOutput

SAMPLE = ChurnInput(
    **{
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
)


@pytest.fixture(scope="session")
def instance():
    instance = PredictorFactory().build()
    assert isinstance(instance, TelecomChurnPredictor)
    return instance


def test_predictor(instance):
    pred = instance.predict(SAMPLE)
    assert isinstance(pred, ChurnOutput)


def test_prediction_stability(instance):
    pred1 = instance.predict(SAMPLE)
    pred2 = instance.predict(SAMPLE)

    assert pred1 == pred2
