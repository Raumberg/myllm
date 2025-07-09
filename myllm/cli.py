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

from myllm.utils.std import infer_dtype
from myllm.enums import AlgorithmType, EngineType

app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):  # noqa: D401
    """myllm – lightweight LLM fine-tuning framework."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def train(
    config: Path = typer.Option(..., help="Path to YAML/TOML config file."),
    algo: AlgorithmType = typer.Option("sft", help="Training algorithm.", case_sensitive=False),
    engine: EngineType = typer.Option("deepspeed", help="Backend engine.", case_sensitive=False),
    overrides: List[str] = typer.Argument(None, help="Override config values: key=value"),
    resume_from: Optional[Path] = typer.Option(None, help="Path to checkpoint to resume from."),
):
    """Launch training run."""
    # Lazy imports to speed up CLI display
    from myllm.config.argparser import SmartParser
    from myllm.engines import get_engine
    from myllm.algorithms import get_algorithm, get_trainer_class
    from myllm.data import DataModule
    from myllm.models import ModelWrapper
    from myllm.utils.logging_utils import apply_logging_cfg

    # Config
    cfg_obj = SmartParser.load_from_file(config, overrides)
    apply_logging_cfg(cfg_obj.logging)

    # Log training start
    logger.info("Starting training: algo=%s, engine=%s", algo.value, engine.value)

    # Model
    wrapper = ModelWrapper(
        model_name=cfg_obj.model.name,
        dtype=infer_dtype(cfg_obj.model.dtype),
        attn_implementation=cfg_obj.model.attn_implementation,
        use_4bit=getattr(cfg_obj.model, "use_4bit", False),
        use_8bit=getattr(cfg_obj.model, "use_8bit", False),
        bnb_compute_dtype=getattr(cfg_obj.model, "bnb_compute_dtype", "bf16"),
    )
    model = wrapper.model

    # Algorithm
    algo_mod = get_algorithm(algo)
    trainer_cls = get_trainer_class(algo_mod)

    # Datasets and dataloaders
    dm = DataModule(cfg_obj.data, cfg_obj.training, tokenizer_name=cfg_obj.model.name)
    dm.setup()
    
    # Sync tokenizer with model config to prevent training issues
    dm.tokenizer_wrapper.sync_with_model(model)
    
    # Select training dataloader
    train_dataloader = dm.get_train_dataloader()

    # Engine
    engine_mod = get_engine(engine)
    engine, _, _, _ = engine_mod.prepare(cfg_obj, model, dataloader_len=len(train_dataloader))

    # Trainer
    trainer = trainer_cls(model, engine, cfg_obj)

    # Training
    ckpt_path = resume_from or cfg_obj.training.resume_from_checkpoint
    trainer.train(train_dataloader, resume_from=str(ckpt_path) if ckpt_path else None)

    typer.echo("Training loop finished.")


@app.command()
def merge(
    source: Path = typer.Option(..., help="Path to LoRA/PEFT checkpoint directory."),
    output: Optional[Path] = typer.Option(None, help="Where to save merged model (defaults to <source>-merged)."),
    task: str = typer.Option("causal-lm", "--task", help="Model head type: causal-lm | seq-clf"),
    dtype: str = typer.Option("bf16", "--dtype", case_sensitive=False, help="Tensor dtype: f32 | f16 | bf16"),
    keep_adapter: bool = typer.Option(True, help="Save original adapter weights to <output>/original_adapter."),
    overwrite: bool = typer.Option(False, help="Overwrite *output* directory if it exists."),
):
    """Merge LoRA adapters into base weights and export *full* model.

    Steps:
    1. Load PEFT checkpoint from *source* with the requested ``dtype``.
    2. Optionally archive raw adapter to ``<output>/original_adapter``.
    3. Call ``merge_and_unload`` and save ready-to-serve weights.
    """

    import torch
    from transformers import AutoTokenizer

    try:
        from peft import AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification  # type: ignore
    except ImportError as e:
        typer.secho("peft not installed – run `pip install peft`.", fg=typer.colors.RED)
        raise typer.Exit(1) from e

    from myllm.utils.logging_utils import apply_logging_cfg
    from myllm.config.schema import LoggingCfg

    apply_logging_cfg(LoggingCfg(level="info"))  # minimal logger, user can override via env

    torch_dtype = infer_dtype(dtype)  # reuse helper at bottom

    if output is None:
        output = source.with_name(source.name + "-merged")

    if output.exists() and not overwrite:
        typer.secho(f"Output dir {output} exists; pass --overwrite to replace.", fg=typer.colors.RED)
        raise typer.Exit(1)

    logger.info("Loading adapter checkpoint from %s (dtype=%s)", source, torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)

    task = task.lower()
    if task in {"causal-lm", "lm", "clm"}:
        adapter_model = AutoPeftModelForCausalLM.from_pretrained(source, torch_dtype=torch_dtype)
    elif task in {"seq-clf", "sequence-classification", "clf"}:
        adapter_model = AutoPeftModelForSequenceClassification.from_pretrained(source, torch_dtype=torch_dtype, num_labels=1)
    else:
        typer.secho("--task must be causal-lm|seq-clf", fg=typer.colors.RED)
        raise typer.Exit(1)

    if keep_adapter:
        adapter_archive = output / "original_adapter"
        logger.info("Archiving raw adapter weights → %s", adapter_archive)
        adapter_archive.mkdir(parents=True, exist_ok=True)
        adapter_model.save_pretrained(adapter_archive)

    logger.info("Merging adapters… (this can take a while on CPU)")
    merged_model = adapter_model.merge_and_unload()

    logger.info("Saving merged model → %s", output)
    output.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    typer.secho(f"✅  Merged model saved to {output}", fg=typer.colors.GREEN)


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
    from myllm.utils.logging_utils import apply_logging_cfg
    apply_logging_cfg(cfg_obj.logging)

    cfg_obj = SmartParser.load_from_file(config, overrides)

    # Model
    wrapper = ModelWrapper(
        model_path,
        dtype=infer_dtype(cfg_obj.model.dtype),
        attn_implementation=cfg_obj.model.attn_implementation,
        use_4bit=getattr(cfg_obj.model, "use_4bit", False),
        use_8bit=getattr(cfg_obj.model, "use_8bit", False),
        bnb_compute_dtype=getattr(cfg_obj.model, "bnb_compute_dtype", "fp16"),
    )
    model = wrapper.model

    # Datasets and dataloaders
    dm = DataModule(cfg_obj.data, cfg_obj.training, tokenizer_name=cfg_obj.model.name)
    dm.setup()
    
    # Sync tokenizer with model config
    dm.tokenizer_wrapper.sync_with_model(model)

    # Evaluation
    if split == "train":
        dl = dm.get_train_dataloader()
    elif split in {"eval", "valid"}:
        dl = dm.get_eval_dataloader()
    elif split == "test":
        dl = dm.get_test_dataloader()
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



def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main() 