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