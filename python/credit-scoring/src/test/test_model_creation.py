import os
import sys
import torch
import torch.nn as nn
import pytest
import logging as log
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ('../..'))))

try:
    from src.train.model import creditScoringModel
except ImportError:
    log.error("asegurate que la estructura del proyecto sea la correcta.")
    log.error("Este script espera estar en /test y el modelo en src/train/model.py.")
    sys.exit(1)

def set_up_logging(level=log.INFO, log_file: str | None = None):
    handlers =[log.StreamHandler(sys.stdout)]
    if log_file:
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))
    log.basicConfig(
        level=level, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("mlflow"):
        log.getLogger(noisy).setLevel(log.WARNING)

set_up_logging()

#1 config
@pytest.fixture(scope="module")
def model_config_fixture():
    return {
 "num_features": 25,
 "hidden_layers": [128, 64],
 "dropout_rate": 0.2,
 "use_batch_norm": True,
 "activation_funct": "ReLU"
 }

#2 tests

def test_model_installiation(model_config_fixture):
    log.info("TEST: verificando la instancia del modelo")
    try:
        model=creditScoringModel(**model_config_fixture)
        assert model is not None, "El modelo no deberia ser None"
        assert isinstance(model, nn.Module), "El modelo debe ser una instancia de torch.nn.Module"
        log.info("Exito al instancir el modelo")
    except Exception as e:
        pytest.fail(f"La instancia del modelo fallo: {e}.")

