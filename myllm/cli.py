from __future__ import annotations

"""Command-line interface for myllm.

Examples
--------
$ myllm train --config configs/alpaca_sft.yaml --algo sft --engine deepspeed
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List, Any
import os
import subprocess
import socket

import typer

from myllm.utils.std import infer_dtype
from myllm.enums import AlgorithmType, EngineType
from myllm.utils.io import ConfigDumper

from accelerate import PartialState



app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):  # noqa: D401
    """myllm – lightweight LLM fine-tuning framework."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())

def _init_and_dump(cfg: Any, output_dir: Path) -> Optional[ConfigDumper]:
    """Initialize a ConfigDumper for dumping configs to output_dir."""
    if PartialState().is_main_process:
        dumper = ConfigDumper(output_dir=output_dir)
        dumper.dump(cfg, "main_config")
        return dumper
    return None

def _dump_from_trainer(trainer: Any, dumper: ConfigDumper) -> None:
    """Dump configs from trainer to output_dir."""
    if PartialState().is_main_process:
        if hasattr(trainer, "ta"):  # TrainingArguments from BaseTrainer
            dumper.dump(getattr(trainer, "ta"), "training_args")
        if hasattr(trainer, "sft_args"):  # SFTConfig from SFTTrainer
            dumper.dump(getattr(trainer, "sft_args"), "sft_config")  # type: ignore[attr-defined]
        if hasattr(trainer, "_peft_cfg"):  # LoraConfig from BaseTrainer
            dumper.dump(getattr(trainer, "_peft_cfg"), "peft_config")  # type: ignore[attr-defined]


def _maybe_relaunch_with_accelerate(accelerate_config_path: Optional[str] = None):
    """Relaunches the script with `accelerate launch` if not already running under it."""
    # Check for environment variables set by `accelerate`
    if "ACCELERATE_PROCESS_ID" not in os.environ and "RANK" not in os.environ:
        logger.info("Not running under `accelerate launch`, relaunching...")

        command = ["accelerate", "launch"]
        
        # Use the provided config path, otherwise fallback to default.
        config_path = accelerate_config_path or "configs/accelerate/stage3_config.yaml"
        
        if Path(config_path).exists():
            command.extend(["--config_file", config_path])
        else:
            logger.warning(
                "Accelerate config not found at %s. "
                "You might need to specify it with --backend_config. "
                "Running `accelerate launch` with its defaults.", 
                config_path
            )
        
        command.extend(sys.argv)

        logger.info("Relaunching with command: %s", " ".join(command))

        try:
            # Replace the current process with the accelerate launch command.
            # This is cleaner than subprocess.run() as it avoids creating a nested
            # process that can interfere with torch.distributed.
            os.execvp(command[0], command)
            
        except FileNotFoundError:
            logger.error(
                "`accelerate` command not found. "
                "Please ensure Hugging Face's `accelerate` is installed and in your PATH."
            )
            raise typer.Exit(1)
        except Exception as e:
            # This part will likely not be reached if execvp is successful,
            # but it's good practice to have a catch-all.
            logger.error("Failed to relaunch with `accelerate launch`: %s", e)
            raise typer.Exit(1)


@app.command()
def train(
    config: Path = typer.Option(..., help="Path to YAML/TOML config file."),
    algo: AlgorithmType = typer.Option("sft", help="Training algorithm.", case_sensitive=False),
    engine: EngineType = typer.Option("deepspeed", help="Backend engine.", case_sensitive=False),
    backend_config: Optional[Path] = typer.Option(None, help="Path to an accelerate config file."),
    overrides: List[str] = typer.Argument(None, help="Override config values: key=value"),
    dump: bool = typer.Option(False, help="Dump all configs to output_dir."),
    resume_from: Optional[Path] = typer.Option(None, help="Path to checkpoint to resume from."),
):
    """Launch training run."""
    # We pass the accelerate_config path to the relauncher.
    # It's a bit of a hack to parse it here before the main parser,
    # but it's the cleanest way to solve the chicken-and-egg problem.
    accel_config_path = None
    if "--backend-config" in sys.argv:
        try:
            index = sys.argv.index("--backend-config")
            accel_config_path = sys.argv[index + 1]
        except (ValueError, IndexError):
            pass  # Let typer handle the error later
            
    _maybe_relaunch_with_accelerate(accel_config_path)

    # Lazy imports to speed up CLI display
    from myllm.config.argparser import SmartParser
    from myllm.engines import get_engine
    from myllm.algorithms import get_algorithm, get_trainer_class
    from myllm.data import DataModule
    from myllm.models import ModelWrapper
    from myllm.utils.logging_utils import apply_logging_cfg
    from myllm.kernels.patching import patch_model

    # Config
    cfg_obj = SmartParser.load_from_file(config, overrides)
    apply_logging_cfg(cfg_obj.logging)

    # Log training start
    logger.info("Starting training: algo=%s, engine=%s", algo.value, engine.value)

    # Model
    model = ModelWrapper(
        model_cfg=cfg_obj.model,
        attn_implementation=cfg_obj.model.attn_implementation,
    ).model

    # Patch model with fused kernels if enabled
    if getattr(cfg_obj.training, "use_fused_kernels", False):
        logger.info("⚡️ Patching model with fused kernels...")
        patch_model(model)

    # Algorithm
    algo_mod = get_algorithm(algo)
    trainer_cls = get_trainer_class(algo_mod)

    # Datasets and dataloaders
    dm = (
        DataModule(
            data_cfg=cfg_obj.data,
            training_cfg=cfg_obj.training,
            tokenizer_name=cfg_obj.model.name,
        )
        .setup()
        .sync_with_model(model)
    )

    # Select training dataloader
    train_dataloader = dm.get_train_dataloader()

    # Engine
    engine_mod = get_engine(engine)
    engine, _, _, _ = engine_mod.prepare(cfg_obj, model, dataloader_len=len(train_dataloader))

    # Trainer
    trainer = trainer_cls(model, engine, cfg_obj)

    # Dump all other configs
    if dump:
        output_dir = Path(cfg_obj.training.output_dir)
        dumper = _init_and_dump(cfg_obj, output_dir)  # type: ignore[assignment]
        _dump_from_trainer(trainer, dumper)

    # Training
    ckpt_path = resume_from or cfg_obj.training.resume_from_checkpoint
    trainer.train(train_dataloader, resume_from=str(ckpt_path) if ckpt_path else None)

    typer.echo("Training loop finished.")


@app.command()
def estimate(
    model_name: str = typer.Argument(..., help="The model name on the Hugging Face Hub."),
    dtypes: List[str] = typer.Option(
        ["float32", "float16", "int8", "int4"],
        "--dtype",
        help="The dtypes to use for the model.",
    ),
    with_trust: bool = typer.Option(
        False, help="Allow custom models defined on the Hub."
    ),
):
    """Estimate model memory usage for inference and training."""
    # Suppress loud loggers
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    
    from rich.console import Console
    from accelerate.utils import calculate_maximum_sizes, convert_bytes
    from myllm.utils.memory_estimator import (
        create_empty_model,
        rich_table,
        estimate_training_usage,
    )

    try:
        model = create_empty_model(
            model_name,
            library_name='transformers',
            trust_remote_code=with_trust
        )
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)

    total_size, largest_layer = calculate_maximum_sizes(model)
    data = []

    for dtype in dtypes:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]
        
        if dtype == "float16":
            dtype_total_size /= 2
            dtype_largest_layer /= 2
            training_size = estimate_training_usage(total_size)["float16"]
        elif dtype == "int8":
            dtype_total_size /= 4
            dtype_largest_layer /= 4
            training_size = "N/A"
        elif dtype == "int4":
            dtype_total_size /= 8
            dtype_largest_layer /= 8
            training_size = "N/A"
        else: # float32
            training_size = estimate_training_usage(total_size)["float32"]

        data.append([
            dtype,
            convert_bytes(dtype_largest_layer),
            convert_bytes(dtype_total_size),
            convert_bytes(training_size) if isinstance(training_size, (int, float)) else training_size
        ])

    headers = ["dtype", "Largest Layer", "Total Size", "Training using Adam"]
    title = f"Memory Usage for loading `{model_name}`"
    table = rich_table(headers, data, title)
    
    console = Console()
    console.print(table)


@app.command()
def inspect(
    model_name: str = typer.Argument(..., help="The model name on the Hugging Face Hub or local path."),
    max_depth: int = typer.Option(2, help="Max depth to inspect the model."),
    with_trust: bool = typer.Option(
        False, help="Allow custom models defined on the Hub."
    ),
):
    """Display a detailed summary of a model's architecture."""
    # Suppress loud loggers
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    
    from rich.console import Console
    from transformers import AutoModelForCausalLM
    from myllm.utils.model_inspector import model_summary

    console = Console()

    try:
        console.print(f"Loading model `{model_name}`...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=with_trust,
            low_cpu_mem_usage=True,
        )
        console.print("Model loaded.")
    except Exception as e:
        typer.secho(f"Error loading model: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
        
    summary_table = model_summary(model, max_depth=max_depth)
    console.print(summary_table)


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

    from myllm.utils.lazy import peft
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
        adapter_model = peft.AutoPeftModelForCausalLM.from_pretrained(source, torch_dtype=torch_dtype)
    elif task in {"seq-clf", "sequence-classification", "clf"}:
        adapter_model = peft.AutoPeftModelForSequenceClassification.from_pretrained(source, torch_dtype=torch_dtype, num_labels=1)
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