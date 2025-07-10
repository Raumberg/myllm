import pytest
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict
from myllm.config.schema import Config
from myllm.algorithms.sft import SFTTrainer
from myllm.data import DataModule
from myllm.models import ModelWrapper


@pytest.mark.integration
def test_sft_tiny_workflow():
    # Minimal config
    cfg_dict = {
        "model": {"name": "EleutherAI/gpt-neo-125M", "dtype": "float32"},  # Tiny model on CPU
        "training": {"micro_batch_size": 1, "epochs": 1, "lr": 1e-5, "output_dir": "tmp_test"},
        "data": {
            "name": "dummy-dataset",
            "text_field": "text",
            "max_length": 32,
            "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            "pad_token": "<|endoftext|>", # Explicitly set pad_token for this test
        },
        "engine": "accelerate",
    }
    cfg = Config.from_dict(cfg_dict)

    # Tiny dataset
    tiny_data = Dataset.from_dict({"text": ["Hello world", "Test sentence"] * 10})
    mock_dataset_dict = DatasetDict({"train": tiny_data})

    # Load model and tokenizer on CPU
    wrapper = ModelWrapper(cfg.model.name, dtype="float32")
    wrapper.model.to("cpu")
    wrapper.tokenizer.model_max_length = cfg.data.max_length

    # DataModule
    dm = DataModule(cfg.data, cfg.training, tokenizer_name=cfg.model.name)

    # Mock _load_dataset to avoid hitting the network/filesystem
    with patch("myllm.data.module.DataModule._load_dataset", return_value=mock_dataset_dict):
        dm.setup()

    train_dataloader = dm.get_train_dataloader()

    # SFT Trainer
    trainer = SFTTrainer(
        model=wrapper.model,
        engine=None,  # No engine for this test
        cfg=cfg,
    )

    # Minimal train step
    trainer.train(dataloader=train_dataloader, resume_from=None)