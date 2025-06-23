<div align="center">
  <h1 style="background: linear-gradient(to right, black, white); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
        MyLLM
    </h1>
    <div style="border: 10px solid; border-image: linear-gradient(to right, black, white) 1; padding: 10px; display: inline-block;">
        <img src="assets/myllm.png" alt="gallery" width="600"/>
    </div>
    <br>
    <br>
    <p align="center">
        <img src="https://img.shields.io/github/issues/Raumberg/myllm?style=for-the-badge">
        <br>
        <img src="https://img.shields.io/github/languages/count/Raumberg/myllm?style=for-the-badge">
        <img src="https://img.shields.io/github/repo-size/Raumberg/myllm?style=for-the-badge">
        <br>
    </p>
</div>

**Modular RL-fine-tuning toolkit for LLMs** ― SFT, PPO, GRPO, KL-Distillation out of the box, DeepSpeed/Accelerate engines, reward registry and plain-YAML configs.

---

## ✨ Highlights

* **Algorithms** – `sft`, `grpo`, `ppo`, `distill` (KL-KD)
* **Engines** – DeepSpeed ZeRO-3 or vanilla Accelerate (easy switch)
* **LoRA / PEFT** – one flag, any algorithm
* **Reward system** – registry + YAML specs (`correctness_reward:2`, `{ ngram_penalty: { ngram_size: 4 } }`)
* **Tiny-to-Huge** – runs on a laptop (TinyLlama) or on multi-GPU A100 cluster
* **CLI first** – `myllm train`, `myllm merge`, `myllm eval`
* **Config > Code** – everything configurable from a single YAML
* **CI ready** – Ruff, PyTest, GitHub Actions

---

## 🔧 Quick start

### Install

```bash
# editable dev-install (+ruff/pytest/black)
make -C myllmv2 install-dev
# or classic
uv pip install myllm-v2
```

### Myllm v1 (old version):
> Available in `legacy` branch

### Minimal SFT

```yaml
model:
  name: attn-signs/GPTR-8b-v2
  dtype: bf16
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  lora_task_type: "CAUSAL_LM"

  # Quantisation – either 4bit or 8bit (mutually exclusive)
  use_4bit: false
  use_8bit: false  # load in 8-bit via bitsandbytes
  bnb_compute_dtype: "bf16"  # compute dtype for 4-bit if enabled

  # Attention optimisation backend (optional): xformers | flash_attention_2
  attn_implementation: flash_attention_2

wandb:
  enable: true
  project: myllm-experiments
  name: gptr8b-sft-run1

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  epochs: 1
  lr: 2.0e-5

logging:
  level: warning
  suppress: ["transformers", "datasets", "deepspeed"]
  warnings_ignore: ["use_cache=True", "TORCH_CUDA_ARCH_LIST"]
  disable_tqdm: true

engine:
  name: deepspeed
  config: configs/deepspeed/stage_3.json

dataset:
  name: attn-signs/kolmogorov-3
  text_field: conversation
  split: train[:5%]
  test_size: 0.01        # will derive eval split automatically
  max_length: 1024

  # Advanced processing
  processor_type: default          # default | history | grpo
  system_prompt: "You are a helpful assistant."
  model_support_system_role: true

  # misc
  offline: false                  # set true to forbid HF downloads
  chat_template: null             # provide custom template string if needed

collator:
  type: completion_only
  template: "<s>assistant" 
  ignore_index: -100
  strict: false
```

```bash
myllm train --config configs/sft.yaml --algo sft --engine deepspeed
```

### or via Accelerate (prefered in MultiGPU setup):
```bash
accelerate launch --config_file <path-to-accelerate-cfg.yaml> myllm train --config configs/sft.yaml --algo sft
```

## 🗂 Directory layout

```
myllm/
  algorithms/      # sft, grpo, ppo, distill (KD)
  engines/         # deepspeed, accelerate wrapper
  rewards/         # registry + concrete classes
  data/            # processors & datamodule
  callbacks/       # rich progress, wandb
  config/          # dataclass schema + smart parser
  utils/           # logging helpers, etc.
  cli.py           # entry-point for `myllm` CLI
```

---

## 🛠 Development

```bash
make lint   # ruff check
make fmt    # black format
make test   # pytest
make ci     # lint + tests (same as GitHub CI)
```

### GitHub Actions
CI workflow lives in `.github/workflows/ci.yml` – runs ruff & pytest on Python 3.12 using **uv** for dependency resolver.

---

## 📝 Migration notes (v1 → v2)

| Area            | v1                                    | v2 (this repo)                                   |
|-----------------|---------------------------------------|--------------------------------------------------|
| Config          | scattered `.ini` + argparse           | single YAML → dataclass → everything             |
| Algorithms      | SFT, PPO, GRPO (hand-rolled)          | SFT (TRL), GRPO (TRL), PPO (TRL), KD             |
| Rewards         | hard-coded python lists               | registry + YAML spec + auto-import               |
| Engine          | only DeepSpeed, manual JSON hacks     | DeepSpeed OR Accelerate, auto-tune buckets       |
| CLI             | `python train.py ...`                 | `myllm train|merge|eval`                         |


---

## ⚙️  Architecture overview

```
┌──────────────────── CLI (`myllm`) ───────────────────┐
│                                                      │
│  YAML Config  →  Config Dataclass  →  Algorithm ♻    │
│                                     │                │
│                                     ▼                │
│  DataModule  ←  Processor  ←  Dataset (HF)           │
│         │                                            │
│         ▼                                            │
│    HuggingFace Trainer (TRL / vanilla)               │
│             ▲                ▲                       │
│             │                │                       │
│        Reward Registry   Engine (DeepSpeed / Acc)    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

* **OOP-first** – every training algorithm is its own `Trainer` subclass that reuses shared utilities from `BaseTrainer` (LoRA plumbing, WandB env-vars, callbacks, TrainingArguments factory).
* **Reward plug-ins** – adding a new reward has never been easier:

```python
@register_reward
class MyReward(BaseReward):
    name = "my_reward"

    def __call__(self, *, prompts, completions, **_):
        return [len(c) * 0.01 for c in completions]
```

and then, in YAML:

```yaml
training:
  grpo:
    reward_funcs:
      - my_reward:2
```

* **Wrappers instead of hardcode** – engine, Collator, Processors, Callbacks are all customizable classes, you can easily override and copy without breaking the lib.

---

➡️  **Bottom line:**  `pip install -e .` is all you need —  CLI, engine, LoRA, rewards. None of manual bs.

---

## License
Apache 2.0 – do what you want, just keep the notices.

--- 
> [!IMPORTANT]
> Thank you for your interest in MyLLM! We look forward to your contributions and feedback! 🚀
