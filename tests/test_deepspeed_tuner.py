import pytest
from unittest.mock import patch, MagicMock, Mock
from myllm.engines.deepspeed import DeepSpeedConfigTuner


def test_tune_zero3():
    mock_cfg = Mock()
    mock_model = MagicMock()
    mock_model.config.hidden_size = 512
    mock_model.config.num_hidden_layers = 12
    tuner = DeepSpeedConfigTuner(mock_cfg, mock_model)
    tuner.ds_cfg = {
        "zero_optimization": {
            "stage": 3,  # This is required for _tune_zero3 to run
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
        }
    }
    tuner._tune_zero3()  # This method modifies the instance's ds_cfg
    tuned = tuner.ds_cfg
    assert tuned["zero_optimization"]["stage3_prefetch_bucket_size"] > 0
    assert tuned["zero_optimization"]["stage3_param_persistence_threshold"] > 0


def test_validate_config():
    mock_cfg = Mock()
    mock_model = MagicMock()
    tuner = DeepSpeedConfigTuner(mock_cfg, mock_model)
    tuner.ds_cfg = {"some_key": "auto"}  # Set a config with an 'auto' value
    with pytest.raises(ValueError):
        tuner._validate()  # Correct method name is _validate 