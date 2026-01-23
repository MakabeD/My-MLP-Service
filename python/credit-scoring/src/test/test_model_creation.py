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

def test_model_architecture(model_config_fixture):
    log.info("Test: verificando la arquitectura de la red neuronal")
    model= creditScoringModel(**model_config_fixture)
    net= model.network
     # La arquitectura esperada es: (Linear -> BatchNorm -> ReLU -> Dropout) -> (Linear -> BatchNorm -> ReLU -> Dropout) -> Linear
    expected_layers_count = len(model_config_fixture["hidden_layers"])* 4 +1
    assert len(net)== expected_layers_count, f"Se esperaba {expected_layer_count}, habia {len(net)}."
    #hidden layer 1
    assert isinstance(net[0], nn.Linear) and net[0].in_features==25 and net[0].out_features==128
    assert isinstance(net[1], nn.BatchNorm1d) and net[1].num_features==128
    assert isinstance(net[2], nn.ReLU)
    assert isinstance(net[3], nn.Dropout) and net[3].p==0.2

    #hidden layer 2
    assert isinstance(net[4], nn.Linear) and net[4].in_features==128 and net[4].out_features==64
    assert isinstance(net[5], nn.BatchNorm1d) and net[5].num_features==64
    assert isinstance(net[6], nn.ReLU)
    assert isinstance(net[7], nn.Dropout) and net[7].p==0.2

    #hidden layer 3
    assert isinstance(net[8], nn.Linear) and net[8].in_features ==64 and net[8].out_features==1

    log.info("exito al verificar la arquitectura y las capas  del modelo")

def test_forward_pass(model_config_fixture):
    log.info("Test: verificacion del forward pass")
    model =creditScoringModel(model_config_fixture)
    model.eval()

    batch_size=10
    input_tensor=torch.randn(batch_size, model_config_fixture["num_features"])
    with torch.no_grad():
        output=model(input_tensor)
    expected_shape = (batch_size, 1)
    assert output.shape==expected_shape, f"la forma del tensor de salida es incorrecto, esperado: {expected_shape}. recivido: {output.shape}"
    log.info("el forward pass ha sido completado de manera exitosa")


    
    
