"""
Unit tests for the ChurnMLP model.
"""
import sys
import os
import torch
import torch.nn as nn
import pytest

# Adding project root to sys.path (Consider moving this to a conftest.py in the future)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.mlp import ChurnMLP


@pytest.fixture
def model_config_fixture():
    """
    Fixture providing a mock configuration dictionary for the ChurnMLP model.
    Returns:
        dict: A sample configuration matching the expected YAML structure.
    """
    return {
        "num_features": 25,
        "config": {
            "model_config": {
                "model_name": "pytest_model",
                "model_version": "v999",
                "architecture": {
                    "hidden_layers": [64, 32],
                    "dropout_rate": 0.2,
                    "use_batch_norm": True,
                    "activation_fn": "ReLU"
                }
            }
        }
    }

@pytest.fixture
def churn_model(model_config_fixture):
    """
    Fixture that instantiates and returns the ChurnMLP model to avoid code repetition.
    """
    config = model_config_fixture['config']
    features = model_config_fixture['num_features']
    return ChurnMLP(features, config)


def test_model_initialization(churn_model):
    """
    Test if the model initializes correctly and inherits from nn.Module.
    """
    assert churn_model is not None, "Model failed to initialize."
    assert isinstance(churn_model, nn.Module), "Model is not a valid PyTorch nn.Module."


def test_model_architecture(churn_model):
    """
    Verify that the dynamically generated PyTorch sequential network matches 
    the specifications provided in the configuration dictionary.
    """
    net = churn_model.network
    
    # Layer 1: Linear -> BatchNorm -> ReLU -> Dropout
    assert isinstance(net[0], nn.Linear)
    assert net[0].in_features == 25 and net[0].out_features == 64
    assert isinstance(net[1], nn.BatchNorm1d) and net[1].num_features == 64
    assert isinstance(net[2], nn.ReLU)
    assert isinstance(net[3], nn.Dropout) and net[3].p == 0.2
    
    # Layer 2: Linear -> BatchNorm -> ReLU -> Dropout
    assert isinstance(net[4], nn.Linear)
    assert net[4].in_features == 64 and net[4].out_features == 32
    assert isinstance(net[5], nn.BatchNorm1d) and net[5].num_features == 32
    assert isinstance(net[6], nn.ReLU)
    assert isinstance(net[7], nn.Dropout) and net[7].p == 0.2
    
    # Layer 3: Final Output Layer
    assert isinstance(net[8], nn.Linear)
    assert net[8].in_features == 32 and net[8].out_features == 1


def test_forward_pass(churn_model, model_config_fixture):
    """
    Ensure the model can process a batch of random tensors and return the correct output shape.
    """
    batch_size = 10
    features = model_config_fixture['num_features']
    
    input_tensor = torch.randn(batch_size, features)
    
    output = churn_model(input_tensor)
    
    assert output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, but got {output.shape}"


def test_forward_pass_invalid_shape(churn_model):
    """Test that the model raises a RuntimeError if input has wrong feature count."""
    invalid_tensor = torch.randn(10, 20) # 20 features instead of 25
    with pytest.raises(RuntimeError):
        churn_model(invalid_tensor)


def test_evaluation_mode(churn_model, model_config_fixture):
    """Ensure dropout is disabled and behavior is deterministic in eval mode."""
    input_tensor = torch.randn(5, model_config_fixture['num_features'])
    
    churn_model.eval() # Set to evaluation mode
    output1 = churn_model(input_tensor)
    output2 = churn_model(input_tensor)
    
    # In eval mode, outputs for the exact same input should be identical (no dropout randomness)
    assert torch.allclose(output1, output2)