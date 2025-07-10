import pytest
from pathlib import Path
from myllm.config.argparser import SmartParser
from myllm.config.schema import Config

@pytest.fixture
def sample_yaml(tmp_path):
    yaml_content = """
    model:
      name: test-model
      dtype: bf16
    training:
      epochs: 1
      lr: 1e-5
    """
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path

def test_load_config(sample_yaml):
    cfg = SmartParser.load_from_file(sample_yaml)
    assert isinstance(cfg, Config)
    assert cfg.model.name == "test-model"
    assert cfg.training.epochs == 1

def test_load_with_overrides(sample_yaml):
    overrides = ["model.name=new-model", "training.epochs=2"]
    cfg = SmartParser.load_from_file(sample_yaml, overrides)
    assert cfg.model.name == "new-model"
    assert cfg.training.epochs == 2

def test_invalid_override(sample_yaml):
    overrides = ["invalid.key=val"]
    cfg = SmartParser.load_from_file(sample_yaml, overrides)
    # Assert that invalid override is stored in raw but doesn't affect known fields
    assert "invalid" in cfg.raw
    assert cfg.model.name == "test-model"  # unchanged
    assert not hasattr(cfg, 'invalid')  # not in dataclass