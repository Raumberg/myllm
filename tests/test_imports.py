import pytest

ALGOS = ("sft", "grpo", "ppo", "distill")


def test_package_import():
    """myllm must be import-able without side-effects."""
    import importlib  # noqa: WPS433

    assert importlib.import_module("myllm")


@pytest.mark.parametrize("name", ALGOS)
def test_get_algorithm(name):
    """Every algorithm string returns a module."""
    from myllm.algorithms import get_algorithm

    mod = get_algorithm(name)
    assert mod is not None, f"{name} resolved to None"


def test_config_roundtrip():
    """Config dataclass should reconstruct from dict without errors."""
    from myllm.config.schema import Config

    cfg = Config()
    cfg2 = Config.from_dict({})
    assert cfg2.training.micro_batch_size == cfg.training.micro_batch_size 

# Additional import tests for key modules
def test_import_engines():
    from myllm.engines import get_engine
    from myllm.engines.deepspeed import DeepSpeedConfigTuner
    assert get_engine
    assert DeepSpeedConfigTuner

def test_import_algorithms():
    from myllm.algorithms import get_algorithm
    from myllm.algorithms.base import BaseTrainer
    assert get_algorithm
    assert BaseTrainer

def test_import_config():
    from myllm.config.argparser import SmartParser
    from myllm.config.schema import Config
    assert SmartParser
    assert Config

def test_import_data():
    from myllm.data import DataModule
    assert DataModule 