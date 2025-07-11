<div align="center">
  <h1 style="background: linear-gradient(to right, black, white); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
        MyLLM
    </h1>
    <div style="border: 10px solid; border-image: linear-gradient(to right, black, white) 1; padding: 10px; display: inline-block;">
        <img src="assets/myllmv2.png" alt="gallery" width="600"/>
    </div>
    <br>
    <br>
    <p align="center">
        <img src="https://img.shields.io/github/issues/Raumberg/myllm?style=for-the-badge">
        <img src="https://img.shields.io/github/languages/count/Raumberg/myllm?style=for-the-badge">
        <img src="https://img.shields.io/github/repo-size/Raumberg/myllm?style=for-the-badge">
        <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000">
        <br>
    </p>
</div>

**An advanced, config-driven, and high-performance toolkit for fine-tuning LLMs.** Built on Hugging Face (`transformers`, `trl`, `peft`) and modern distribution frameworks (`deepspeed`, `accelerate`), `myllm` simplifies the complex orchestration of LLM training into a clean, declarative, and reproducible workflow.

---

## ✨ Highlights

*   **Declarative, Unified Config**: Manage your entire experiment—from model and data to engine and logging—through a single, clean YAML file. No more scattered scripts or CLI flag hell.
*   **Intelligent DeepSpeed Engine**: Features a cutting-edge, auto-tuning DeepSpeed configuration system. Automatically enables **Flash Attention 2**, `FusedAdam`, and other modern optimizations for H100/A100 GPUs. Dynamically calculates optimal parameters based on your model's architecture.
*   **Stable FP8 Training**: Out-of-the-box support for FP8 training on NVIDIA H100/Ada GPUs, powered by **Transformer Engine**. `myllm` handles the low-level details, so you can focus on your model. Watch out! May be a nightly (beta) version
*   **Full Reproducibility**: Every run automatically saves a snapshot of all resolved configurations (`TrainingArguments`, `SFTConfig`, `LoraConfig`, etc.) to a timestamped directory. Never lose track of what parameters were used.
*   **Modern Algorithms via `trl`**: Leverages Hugging Face's `trl` library to support popular fine-tuning algorithms like SFT, PPO, and distillation.
*   **Robust & Clean Codebase**:
    *   **Fluent, Chainable APIs**: Methods on core classes like `DataModule` are chainable (`.setup().sync_with_model(...)`), leading to more readable and expressive code.
    *   **Lazy Imports**: Eliminates `ImportError` headaches for optional dependencies. Libraries are only imported when they are actually used.
*   **Quantization & PEFT**: Full support for 4/8-bit quantization via `bitsandbytes` and parameter-efficient fine-tuning with LoRA.
*   **Powerful CLI**: A `typer`-based command-line interface provides `train`, `merge`, and `eval` commands for a streamlined workflow.
*   **Developer-Friendly**: Comes with a self-documenting `Makefile` for common tasks like installation, linting, and testing.

---

## 🔧 Quick Start

### 1. Installation

Clone the repository and use the `Makefile` for an editable installation. This will also install all development dependencies.

```bash
git clone https://github.com/Raumberg/myllm.git
cd myllm
make install    # python uv needed
```

### 2. Configure Your Run

Create a single YAML file (e.g., `sft_run.yaml`) to define your experiment.

> [!Note:]
> You can find more complex train cfg examples in configs/ directory of the repo 


```yaml
# sft_run.yaml
model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  dtype: bf16
  attn_implementation: "flash_attention_2" # Use "sdpa" for non-NVIDIA or older GPUs
  
  # PEFT / LoRA configuration
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

  # Optional: 4/8-bit quantization (mutually exclusive with FP8)
  # use_4bit: true
  # bnb_compute_dtype: "bf16"

data:
  name: "HuggingFaceH4/ultrachat_200k"
  processor_type: "default"
  split: "train_sft[:5%]"
  test_size: 0.05
  max_length: 2048
  collator:
    type: "completion_only"
    template: "### Assistant:" # Response template for completion-only loss

training:
  output_dir: "experiments/llama2-7b-sft"
  epochs: 1
  micro_batch_size: 2
  gradient_accumulation_steps: 8
  lr: "2.0e-5" # Can be a string or float
  gradient_checkpointing: true

engine:
  name: "deepspeed" # Or "accelerate"
  # For DeepSpeed, the config is auto-generated! No JSON file needed.
  # Key parameters are calculated at runtime based on your model.

wandb:
  enable: true
  project: "myllm-sft-runs"
  name: "llama2-7b-sft-ultrachat"

logging:
  level: "info"
  disable_tqdm: true
```

### 3. Launch Training

Use the `myllm` CLI to launch the training run.

```bash
# With our intelligent DeepSpeed engine
myllm train --config sft_run.yaml --algo sft --engine deepspeed

# Or with Hugging Face Accelerate
# First, create a minimal Accelerate config: accelerate config / or use existing ones in the repo
accelerate launch myllm train --config sft_run.yaml --algo sft --engine accelerate --dump
```

After the run, check `experiments/llama2-7b-sft/.run/` for the dumped configuration files.

---

## 🗂 Project Structure

```
myllm/
  algorithms/      # SFT, PPO, Distill trainers (wrappers around TRL)
  callbacks/       # Rich progress, WandB, and other callbacks
  config/          # Pydantic schema for config validation
  data/            # DataModule, collators, and text processors
  engines/         # DeepSpeed and Accelerate backend logic
  models/          # Model and tokenizer wrappers
  utils/           # Lazy importer, config dumper, and other helpers
  cli.py           # Entry-point for the `myllm` CLI
```

---

## 🛠 Development

The project uses `make` for common development tasks. Run `make help` to see all available commands.

```bash
make help   # List all available commands
make lint   # Run ruff linter and formatter
make test   # Run tests with pytest
make ci     # Run the full CI pipeline (lint + test)
```

The CI workflow is defined in `.github/workflows/ci.yml`.

---

## ⚙️ Architecture Overview

`myllm` follows a modular, object-oriented design that prioritizes composition and clear separation of concerns.

```
┌──────────────────── CLI (`myllm train`) ──────────────────┐
│                                                           │
│  YAML Config ──► SmartParser ──► Trainer Initialization   │
│                          │                                │
│                          └─────► DataModule.setup()       │
│                                          │                │
│                                          ▼                │
│        HuggingFace Trainer (TRL) ◄── Engine Backend       │
│         (manages training loop)       DeepSpeed/Accelerate│
│                                                           │
└───────────────────────────────────────────────────────────┘
```

*   **CLI & SmartParser**: The `typer`-based CLI parses the command and the YAML config path. The `SmartParser` loads the YAML and resolves it into a structured configuration object.
*   **Engine Backend**: The selected engine (`deepspeed` or `accelerate`) prepares the model and optimizer for distributed training. The DeepSpeed engine dynamically generates its configuration.
*   **Trainer**: The algorithm-specific `Trainer` (e.g., `SFTTrainer`) is initialized with the model, engine, and config. It constructs the necessary components like `TrainingArguments` and `SFTConfig`.
*   **DataModule**: Handles loading, processing, and serving data via `DataLoader`s. It uses a fluent API for a clean setup process.
*   **TRL Integration**: The core training loop is delegated to a Hugging Face `trl` trainer, which reliably handles the complexities of distributed training, gradient accumulation, and callbacks.

---

## License

Apache 2.0 – do what you want, just keep the notices.

--- 
> [!IMPORTANT]
> Thank you for your interest in MyLLM! We look forward to your contributions and feedback! 🚀

