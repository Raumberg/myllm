from __future__ import annotations

"""Command-line interface for myllm.

Examples
--------
$ myllm train --config configs/alpaca_sft.yaml --algo sft --engine deepspeed
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List

import typer

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

# Apply minimal logging config early
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):  # noqa: D401
    """myllm – lightweight LLM fine-tuning framework."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def train(
    config: Path = typer.Option(..., help="Path to YAML/TOML config file."),
    algo: str = typer.Option("sft", help="Training algorithm (sft|ppo|dpo|grpo)."),
    engine: str = typer.Option("deepspeed", help="Backend engine (deepspeed|accelerate)."),
    overrides: List[str] = typer.Argument(None, help="Override config values: key=value"),
    resume_from: Optional[Path] = typer.Option(None, help="Path to checkpoint to resume from."),
):
    """Launch training run."""
    # Lazy imports to speed up CLI display
    from myllm.config.argparser import SmartParser
    from myllm.engines import get_engine
    from myllm.algorithms import get_algorithm
    from myllm.data import DataModule
    from myllm.models import ModelWrapper

    # parse via SmartParser (reuse CLI already)
    cfg_obj = SmartParser.load_from_file(config, overrides)

    # Apply logging directives early
    from myllm.utils.logging_utils import apply_logging_cfg
    apply_logging_cfg(cfg_obj.logging)

    logger.info("Starting training: algo=%s, engine=%s", algo, cfg_obj.engine.name)

    # Load real model & tokenizer
    wrapper = ModelWrapper(
        cfg_obj.model.name,
        dtype=_dtype_from_str(cfg_obj.model.dtype),
        attn_implementation=cfg_obj.model.attn_implementation,
        use_4bit=getattr(cfg_obj.model, "use_4bit", False),
        use_8bit=getattr(cfg_obj.model, "use_8bit", False),
        bnb_compute_dtype=getattr(cfg_obj.model, "bnb_compute_dtype", "fp16"),
    )
    model = wrapper.model

    # Algorithm
    algo_mod = get_algorithm(algo)
    # Generic access: prefer 'Trainer', else fall back to '{AlgoNameUpper}Trainer', else specific names
    if hasattr(algo_mod, "Trainer"):
        trainer_cls = getattr(algo_mod, "Trainer")
    else:
        camel = f"{algo.upper()}Trainer"
        trainer_cls = getattr(algo_mod, camel, None)
        if trainer_cls is None:
            raise RuntimeError(f"Algorithm module {algo_mod.__name__} does not expose a Trainer class")

    # For TRL SFT we delegate deepspeed handling to HF Trainer, so skip manual engine init
    engine = None
    if algo not in {"sft", "grpo", "ppo", "distill", "dpo"}:
        engine_mod = get_engine(cfg_obj.engine.name)
        engine, _, _, _ = engine_mod.prepare(cfg_obj, model)

    trainer = trainer_cls(model, engine, cfg_obj)

    dm = DataModule(cfg_obj.data, cfg_obj.training, tokenizer_name=cfg_obj.model.name)
    dm.setup()
    dl = dm.train_dataloader()

    ckpt_path = resume_from or cfg_obj.training.resume_from_checkpoint
    trainer.train(dl, resume_from=str(ckpt_path) if ckpt_path else None)

    typer.echo("Training loop finished.")


@app.command()
def merge(
    source: Path = typer.Option(..., help="Path to fine-tuned model directory containing LoRA adapters (HF format)."),
    output: Path = typer.Option(..., help="Output directory for merged full-precision model."),
    is_clf: bool = typer.Option(False, "--is-clf", help="Treat model as sequence-classification (AutoPeftModelForSequenceClassification) instead of causal LM."),
    dtype: str = typer.Option("bf16", help="Torch dtype for loading weights: f32 | f16 | bf16"),
):
    """Merge LoRA adapters with base weights and save full model to *output*.

    The *source* directory should be a PEFT checkpoint produced by training with
    ``use_peft=True``. The command will:

    1. Load PEFT model with the specified ``dtype``.
    2. Save raw adapter weights to ``output/original_adapter`` (for archival).
    3. Merge adapters into the base model and save the final model to ``output``.
    """

    import os
    import torch

    try:
        from peft import AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification  # type: ignore
    except ImportError as e:  # pragma: no cover
        typer.secho("peft is not installed – run `pip install peft` first.", fg=typer.colors.RED)
        raise typer.Exit(1) from e

    from transformers import AutoTokenizer  # lazy import to speed up CLI

    dtype = dtype.lower()
    dtype_mapping = {
        "f32": torch.float32,
        "fp32": torch.float32,
        "32": torch.float32,
        "f16": torch.float16,
        "fp16": torch.float16,
        "16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype not in dtype_mapping:
        typer.secho(f"Unknown dtype '{dtype}' (choose from f32|f16|bf16).", fg=typer.colors.RED)
        raise typer.Exit(1)
    torch_dtype = dtype_mapping[dtype]

    logger.info("Loading PEFT model from %s (dtype=%s)", source, torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(source)

    if not is_clf:
        adapter_model = AutoPeftModelForCausalLM.from_pretrained(source, torch_dtype=torch_dtype)
    else:
        adapter_model = AutoPeftModelForSequenceClassification.from_pretrained(source, torch_dtype=torch_dtype, num_labels=1)

    adapter_save_path = output / "original_adapter"
    adapter_save_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving raw adapter weights to %s", adapter_save_path)
    adapter_model.save_pretrained(adapter_save_path)

    logger.info("Merging adapters into base model… this may take a while")
    merged_model = adapter_model.merge_and_unload()

    logger.info("Saving merged model to %s", output)
    output.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    typer.secho(f"Merged model saved to {output}", fg=typer.colors.GREEN)


@app.command()
def eval(
    config: Path = typer.Option(..., help="Path to YAML/TOML config file with dataset & model info."),
    model_path: str = typer.Option(..., help="HF model name or local path to model checkpoint (merged or with adapters)."),
    split: str = typer.Option("eval", help="Dataset split to evaluate (train|eval|test)"),
    overrides: List[str] = typer.Argument(None, help="Override config values: key=value"),
):
    """Run evaluation on a dataset and report metrics.

    Currently computes only *perplexity*, but CLI stays stable when we add more
    metrics later.
    """

    from myllm.config.argparser import SmartParser
    from myllm.data import DataModule
    from myllm.models import ModelWrapper
    from myllm.metrics import Evaluator, Perplexity

    cfg_obj = SmartParser.load_from_file(config, overrides)

    # Apply logging config
    from myllm.utils.logging_utils import apply_logging_cfg
    apply_logging_cfg(cfg_obj.logging)

    # Load model (no training, so no DeepSpeed init required)
    wrapper = ModelWrapper(
        model_path,
        dtype=_dtype_from_str(cfg_obj.model.dtype),
        attn_implementation=cfg_obj.model.attn_implementation,
        use_4bit=getattr(cfg_obj.model, "use_4bit", False),
        use_8bit=getattr(cfg_obj.model, "use_8bit", False),
        bnb_compute_dtype=getattr(cfg_obj.model, "bnb_compute_dtype", "fp16"),
    )
    model = wrapper.model

    dm = DataModule(cfg_obj.data, cfg_obj.training, tokenizer_name=cfg_obj.model.name)
    dm.setup()

    if split == "train":
        dl = dm.train_dataloader()
    elif split in {"eval", "valid"}:
        dl = dm.eval_dataloader()
    elif split == "test":
        dl = dm.test_dataloader()
    else:
        raise typer.BadParameter("split must be one of train|eval|test", param_hint="split")

    if dl is None:
        typer.secho(f"Split '{split}' not available in dataset.", fg=typer.colors.RED)
        raise typer.Exit(1)

    evaluator = Evaluator(model, metrics=[Perplexity()])
    scores = evaluator.evaluate(dl)

    for k, v in scores.items():
        typer.secho(f"{k}: {v:.4f}", fg=typer.colors.GREEN)

    typer.echo("Evaluation finished.")


def _dtype_from_str(s: str):  # noqa: D401
    import torch
    s = s.lower()
    if s in {"fp16", "float16", "16"}:
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    return torch.float32


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main() 